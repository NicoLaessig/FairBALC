import numpy as np
import random
import time
from math import gcd
import networkx as nx
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from imblearn.under_sampling import RandomUnderSampler

from FCF.utils import distance


class SparseFairletDecomposition:
    """
    Fairlet decomposition with multiple operating modes.

    Modes
    -----
    sparse            – min-cost PERFECT matching on kNN graph (may be infeasible)
    sparse_fallback   – (SFFD) single-round max-cardinality Hopcroft–Karp, drop residuals
    sparse_label      – max-cardinality matching on same-label kNN edges
    mcmm              – single-round MCMM via dummy-augmented graph (O(N^2) edges)
    mcmm_flow         – single-round MCMM via sparse min-cost flow
    mcmm_dummy        – alias for mcmm
    mcmm_nutb         – mcmm without undersampling
    greedy            – greedy cost-sorted matching
    greedy_then_flow  – greedy first pass, then MCMM-flow on residuals
    shfd              – (SHFD / Option B) progressive kNN schedule + residual solver
    """

    def __init__(
        self,
        a,
        data,
        y=None,
        k=50,
        bal="under",
        random_state=42,
        cost_scale=1_000_000,
        slack_penalty=None,
    ):
        self.a = np.asarray(a).ravel()
        self.data = np.asarray(data)
        self.y = np.asarray(y)
        self.k = int(k)
        self.bal = bal
        self.random_state = int(random_state)
        self.cost_scale = int(cost_scale)
        self.slack_penalty = slack_penalty

        self.used_indices_ = None
        self.dropped_majority_indices_ = None
        self.dropped_by_slack_ = None
        self.dropped_by_matching_ = None


    def _filter_edges_same_label(self, edges, y_left, y_right):
        y_left = np.asarray(y_left).ravel()
        y_right = np.asarray(y_right).ravel()
        filtered = [(i, j, c) for (i, j, c) in edges if y_left[i] == y_right[j]]
        max_c = max((c for _, _, c in filtered), default=0.0)
        return filtered, float(max_c)

    def _binary_remap_majority_minority(self):
        vals, counts = np.unique(self.a, return_counts=True)
        if len(vals) != 2:
            raise ValueError(f"Requires binary a; got {vals}.")
        maj = vals[np.argmax(counts)]
        a01 = np.where(self.a == maj, 0, 1).astype(int)
        return a01

    def _undersample_to_balance(self, a01):
        rus = RandomUnderSampler(sampling_strategy="auto",
                                 random_state=self.random_state)
        rus.fit_resample(self.data, a01)
        used_idx = np.asarray(rus.sample_indices_, dtype=int)
        self.used_indices_ = used_idx
        mask = np.ones(len(a01), dtype=bool)
        mask[used_idx] = False
        self.dropped_majority_indices_ = np.where(mask)[0]
        return used_idx

    def _build_bidirectional_knn_edges(self, X_left, X_right, k):
        n_left, n_right = X_left.shape[0], X_right.shape[0]
        if n_left == 0 or n_right == 0:
            return [], set(), 0.0

        kL = min(max(1, k), n_right)
        kR = min(max(1, k), n_left)
        edges, present, max_c = [], set(), 0.0

        nnR = NearestNeighbors(n_neighbors=kL, algorithm="auto")
        nnR.fit(X_right)
        _, idxsR = nnR.kneighbors(X_left, return_distance=True)
        for i in range(n_left):
            for pos in range(kL):
                j = int(idxsR[i, pos])
                if (i, j) not in present:
                    c = float(distance(X_left[i], X_right[j]))
                    edges.append((i, j, c))
                    present.add((i, j))
                    max_c = max(max_c, c)

        nnL = NearestNeighbors(n_neighbors=kR, algorithm="auto")
        nnL.fit(X_left)
        _, idxsL = nnL.kneighbors(X_right, return_distance=True)
        for j in range(n_right):
            for pos in range(kR):
                i = int(idxsL[j, pos])
                if (i, j) not in present:
                    c = float(distance(X_left[i], X_right[j]))
                    edges.append((i, j, c))
                    present.add((i, j))
                    max_c = max(max_c, c)

        return edges, present, max_c


    def _solve_min_cost_perfect_matching(self, N, edges):
        G = nx.DiGraph()
        s, t = "s", "t"
        G.add_node(s, demand=-N)
        G.add_node(t, demand=+N)
        for i in range(N):
            G.add_node(f"L{i}", demand=0)
            G.add_node(f"R{i}", demand=0)
            G.add_edge(s, f"L{i}", capacity=1, weight=0)
            G.add_edge(f"R{i}", t, capacity=1, weight=0)
        for i, j, c in edges:
            G.add_edge(f"L{i}", f"R{j}", capacity=1,
                       weight=int(round(self.cost_scale * c)))
        flow = nx.min_cost_flow(G)
        match = [-1] * N
        for i in range(N):
            for node, fval in flow[f"L{i}"].items():
                if fval == 1 and node.startswith("R"):
                    match[i] = int(node[1:])
                    break
            if match[i] < 0:
                raise RuntimeError("Could not extract perfect matching from flow.")
        return match

    def _solve_min_cost_with_slack(self, N, edges, max_edge_cost):
        P_dist = max(1.0, 3.0 * float(max_edge_cost)) \
                 if self.slack_penalty is None else float(self.slack_penalty)
        P = int(round(self.cost_scale * P_dist))
        G = nx.DiGraph()
        s, t = "s", "t"
        G.add_node(s, demand=-(2 * N))
        G.add_node(t, demand=+(2 * N))
        for i in range(N):
            for tag in (f"L{i}", f"R{i}", f"DL{i}", f"DR{i}"):
                G.add_node(tag, demand=0)
            G.add_edge(s, f"L{i}", capacity=1, weight=0)
            G.add_edge(s, f"DL{i}", capacity=1, weight=0)
            G.add_edge(f"R{i}", t, capacity=1, weight=0)
            G.add_edge(f"DR{i}", t, capacity=1, weight=0)
        for i, j, c in edges:
            G.add_edge(f"L{i}", f"R{j}", capacity=1,
                       weight=int(round(self.cost_scale * c)))
        for i in range(N):
            for k in range(N):
                G.add_edge(f"L{i}", f"DR{k}", capacity=1, weight=P)
                G.add_edge(f"DL{k}", f"R{i}", capacity=1, weight=P)
                G.add_edge(f"DL{i}", f"DR{k}", capacity=1, weight=0)
        flow = nx.min_cost_flow(G)
        real_pairs, dropped_left, dropped_right = [], [], []
        matched_real_right = set()
        for i in range(N):
            chosen = next((nd for nd, fv in flow[f"L{i}"].items() if fv == 1), None)
            if chosen is None:
                raise RuntimeError("Slack flow extraction failed.")
            if chosen.startswith("R"):
                real_pairs.append((i, int(chosen[1:])))
                matched_real_right.add(int(chosen[1:]))
            else:
                dropped_left.append(i)
        dropped_right = [j for j in range(N) if j not in matched_real_right]
        return real_pairs, dropped_left, dropped_right

    def _solve_max_cardinality_matching_old(self, N, edges):
        B = nx.Graph()
        left_nodes = [f"L{i}" for i in range(N)]
        right_nodes = [f"R{j}" for j in range(N)]
        B.add_nodes_from(left_nodes, bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)
        for i, j, _c in edges:
            B.add_edge(f"L{i}", f"R{j}")
        matching = nx.algorithms.bipartite.matching.maximum_matching(
            B, top_nodes=left_nodes)
        pairs, matched_left, matched_right = [], set(), set()
        for Li in left_nodes:
            if Li in matching and matching[Li].startswith("R"):
                i, j = int(Li[1:]), int(matching[Li][1:])
                pairs.append((i, j))
                matched_left.add(i)
                matched_right.add(j)
        dropped_left  = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right

    def _solve_max_cardinality_matching(self, N_left, N_right, edges):
        B = nx.Graph()
        left_nodes  = [f"L{i}" for i in range(N_left)]
        right_nodes = [f"R{j}" for j in range(N_right)]
        B.add_nodes_from(left_nodes, bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)
        for i, j, _c in edges:
            B.add_edge(f"L{i}", f"R{j}")
        matching = nx.algorithms.bipartite.matching.maximum_matching(
            B, top_nodes=left_nodes)
        pairs, matched_left, matched_right = [], set(), set()
        for Li in left_nodes:
            if Li in matching and matching[Li].startswith("R"):
                pairs.append((int(Li[1:]), int(matching[Li][1:])))
                matched_left.add(int(Li[1:]))
                matched_right.add(int(matching[Li][1:]))
        dropped_left  = [i for i in range(N_left)  if i not in matched_left]
        dropped_right = [j for j in range(N_right) if j not in matched_right]
        return pairs, dropped_left, dropped_right

    def _solve_min_cost_maximum_matching(self, N, edges):
        left_real   = [f"L{i}" for i in range(N)]
        right_real  = [f"R{j}" for j in range(N)]
        left_dummy  = [f"DL{i}" for i in range(N)]
        right_dummy = [f"DR{j}" for j in range(N)]
        top_nodes   = left_real + left_dummy
        B = nx.Graph()
        B.add_nodes_from(top_nodes, bipartite=0)
        B.add_nodes_from(right_real + right_dummy, bipartite=1)
        max_c = max((float(c) for _, _, c in edges if c is not None), default=0.0)
        BIG_M = (N + 1) * (max_c + 1.0) + 1.0
        for i, j, c in edges:
            B.add_edge(f"L{i}", f"R{j}", weight=float(c))
        for i in range(N):
            for j in range(N):
                B.add_edge(f"L{i}",  f"DR{j}", weight=BIG_M)
                B.add_edge(f"DL{i}", f"R{j}",  weight=BIG_M)
                B.add_edge(f"DL{i}", f"DR{j}", weight=0.0)
        matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
            B, top_nodes=top_nodes, weight="weight")
        pairs, matched_left, matched_right = [], set(), set()
        for Li in left_real:
            mate = matching.get(Li)
            if mate and mate.startswith("R"):
                i, j = int(Li[1:]), int(mate[1:])
                pairs.append((i, j))
                matched_left.add(i)
                matched_right.add(j)
        dropped_left  = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right

    def _solve_greedy_matching(self, N, edges):
        edges_sorted = sorted(edges, key=lambda x: x[2])
        usedL, usedR, pairs = set(), set(), []
        for i, j, _c in edges_sorted:
            if i not in usedL and j not in usedR:
                usedL.add(i); usedR.add(j)
                pairs.append((i, j))
        dropped_left  = [i for i in range(N) if i not in usedL]
        dropped_right = [j for j in range(N) if j not in usedR]
        return pairs, dropped_left, dropped_right

    def _solve_mcmm_with_dummies(self, N, edges):
        left_real   = [f"L{i}" for i in range(N)]
        right_real  = [f"R{j}" for j in range(N)]
        left_dummy  = [f"DL{i}" for i in range(N)]
        right_dummy = [f"DR{j}" for j in range(N)]
        top_nodes   = left_real + left_dummy
        B = nx.Graph()
        B.add_nodes_from(top_nodes, bipartite=0)
        B.add_nodes_from(right_real + right_dummy, bipartite=1)
        max_c = max((float(c) for _, _, c in edges), default=0.0)
        BIG_M = (N + 1) * (max_c + 1.0) + 1.0
        for i, j, c in edges:
            B.add_edge(f"L{i}", f"R{j}", weight=float(c))
        for i in range(N):
            for j in range(N):
                B.add_edge(f"L{i}",  f"DR{j}", weight=BIG_M)
                B.add_edge(f"DL{i}", f"R{j}",  weight=BIG_M)
                B.add_edge(f"DL{i}", f"DR{j}", weight=0.0)
        matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
            B, top_nodes=top_nodes, weight="weight")
        pairs, matched_left, matched_right = [], set(), set()
        for Li in left_real:
            mate = matching.get(Li)
            if mate and mate.startswith("R"):
                i, j = int(Li[1:]), int(mate[1:])
                pairs.append((i, j))
                matched_left.add(i)
                matched_right.add(j)
        dropped_left  = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right

    def _solve_greedy_then_flow(self, N, edges):
        pairs_g, droppedL, droppedR = self._solve_greedy_matching(N, edges)
        if not droppedL or not droppedR:
            return pairs_g, droppedL, droppedR
        keptL  = droppedL[:min(len(droppedL), len(droppedR))]
        keptR  = droppedR[:min(len(droppedL), len(droppedR))]
        Nsub   = len(keptL)
        mapL   = {old: new for new, old in enumerate(keptL)}
        mapR   = {old: new for new, old in enumerate(keptR)}
        edges_sub = [(mapL[i], mapR[j], c)
                     for i, j, c in edges if i in mapL and j in mapR]
        pairs_sub, _, _ = self._solve_mcmm_via_flow(Nsub, edges_sub)
        pairs = pairs_g + [(keptL[i], keptR[j]) for i, j in pairs_sub]
        matchedL = {i for i, _ in pairs}
        matchedR = {j for _, j in pairs}
        return pairs, \
               [i for i in range(N) if i not in matchedL], \
               [j for j in range(N) if j not in matchedR]

    def _solve_mcmm_via_flow(self, N_left, N_right, edges):
        B = nx.Graph()
        left_nodes  = [f"L{i}" for i in range(N_left)]
        right_nodes = [f"R{j}" for j in range(N_right)]
        B.add_nodes_from(left_nodes,  bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)
        for i, j, _c in edges:
            B.add_edge(f"L{i}", f"R{j}")
    
        hk = nx.algorithms.bipartite.matching.maximum_matching(
            B, top_nodes=left_nodes)
    
        pairs_hk, usedL, usedR = [], set(), set()
        for Li in left_nodes:
            if Li in hk and hk[Li].startswith("R"):
                i = int(Li[1:])
                j = int(hk[Li][1:])
                pairs_hk.append((i, j))
                usedL.add(i)
                usedR.add(j)
    
        m_star = len(pairs_hk)
    
        if m_star == 0:
            return (
                [],
                list(range(N_left)),
                list(range(N_right)),
            )
    
        G = nx.DiGraph()
        s, t = "s", "t"
        G.add_node(s, demand=-m_star)
        G.add_node(t, demand=+m_star)
    
        for i in range(N_left):
            G.add_node(f"L{i}", demand=0)
            G.add_edge(s, f"L{i}", capacity=1, weight=0)
    
        for j in range(N_right):
            G.add_node(f"R{j}", demand=0)
            G.add_edge(f"R{j}", t, capacity=1, weight=0)
    
        for i, j, c in edges:
            G.add_edge(f"L{i}", f"R{j}",
                    capacity=1,
                    weight=int(1e6 * float(c)))
    
        try:
            _, flow_dict = nx.network_simplex(G)
        except nx.NetworkXUnfeasible:
            return (
                pairs_hk,
                [i for i in range(N_left)  if i not in usedL],
                [j for j in range(N_right) if j not in usedR],
            )
    
        pairs, matched_left, matched_right = [], set(), set()
        for i in range(N_left):
            for node, fv in flow_dict[f"L{i}"].items():
                if fv == 1 and node.startswith("R"):
                    j = int(node[1:])
                    pairs.append((i, j))
                    matched_left.add(i)
                    matched_right.add(j)
                    break   # each L can match at most one R
    
        dropped_left  = [i for i in range(N_left)  if i not in matched_left]
        dropped_right = [j for j in range(N_right) if j not in matched_right]
        return pairs, dropped_left, dropped_right
    
    def _solve_greedy_matching_rect(self, N_left, N_right, edges):
        edges_sorted = sorted(edges, key=lambda x: x[2])
        usedL, usedR, pairs = set(), set(), []
        for i, j, _c in edges_sorted:
            if i not in usedL and j not in usedR:
                usedL.add(i)
                usedR.add(j)
                pairs.append((i, j))
        dropped_left  = [i for i in range(N_left)  if i not in usedL]
        dropped_right = [j for j in range(N_right) if j not in usedR]
        return pairs, dropped_left, dropped_right
 
    
    def _shfd_one_round(
        self,
        X_left_full,
        X_right_full,
        unmatched_left,
        unmatched_right,
        k,
    ):
        if len(unmatched_left) == 0 or len(unmatched_right) == 0:
            return [], unmatched_left, unmatched_right
    
        # Sub-matrices for unmatched points only — NO truncation
        X_sl = X_left_full[unmatched_left]
        X_sr = X_right_full[unmatched_right]
    
        N_sl = len(unmatched_left)
        N_sr = len(unmatched_right)
    
        # kNN edges in sub-local space
        # _build_bidirectional_knn_edges already handles N_left != N_right
        # internally (kL = min(k, N_right), kR = min(k, N_left))
        edges_sub, _, _ = self._build_bidirectional_knn_edges(X_sl, X_sr, k=k)
    
        if not edges_sub:
            return [], unmatched_left, unmatched_right
    
        # Rectangular MCMM — no truncation of either side
        pairs_sub, dropL_sub, dropR_sub = self._solve_mcmm_via_flow(
            N_sl, N_sr, edges_sub
        )
    
        # Lift sub-local → local
        new_pairs = [
            (unmatched_left[i], unmatched_right[j])
            for i, j in pairs_sub
        ]
    
        matched_sub_L = {i for i, _ in pairs_sub}
        matched_sub_R = {j for _, j in pairs_sub}
    
        # Unmatched points on BOTH sides flow into the next round
        still_left  = [unmatched_left[i]
                    for i in range(N_sl)
                    if i not in matched_sub_L]
        still_right = [unmatched_right[j]
                    for j in range(N_sr)
                    if j not in matched_sub_R]
    
        return new_pairs, still_left, still_right

    def decompose(
        self,
        mode="sparse",
        # SHFD-specific parameters
        shfd_schedule=None,
        shfd_residual="greedy",
    ):
        """
        Parameters
        ----------
        mode : str
            One of the mode strings listed in the class docstring.
            Use "shfd" for the Sparse-Hybrid Fairlet Decomposition (Option B).

        shfd_schedule : list[int] | None
            Only used when mode="shfd".
            List of kNN sizes for the progressive matching rounds, e.g.
            [10, 20, 40].  Each round operates only on the residuals from
            the previous round.  Defaults to [10, 20, 40] if None.

        shfd_residual : str
            Only used when mode="shfd".
            Solver for whatever remains unmatched after all schedule rounds.
            Options:
              "greedy"   – greedy cost-sorted matching on a dense kNN graph
                           (k = max(shfd_schedule) * 2, capped at N)
              "mcmm_flow"– MCMM via sparse flow on a denser kNN graph
              "drop"     – do not match residuals; leave them in
                           self.dropped_by_matching_ (same as SFFD behaviour)
            Default: "greedy".
        """
        a01 = self._binary_remap_majority_minority()

        if mode != "mcmm_nutb" and self.bal == "under":
            used_idx = self._undersample_to_balance(a01)
        else:
            used_idx = np.arange(len(self.data))

        X_bal = self.data[used_idx]
        a_bal = a01[used_idx]

        left_local  = np.where(a_bal == 0)[0]
        right_local = np.where(a_bal == 1)[0]

        underrepresented = "left" if len(left_local) < len(right_local) else "right"

        if self.bal == "under":
            N = min(len(left_local), len(right_local))
            left_local  = left_local[:N]
            right_local = right_local[:N]
            N_left = N_right = N
        else:
            N_left  = len(left_local)
            N_right = len(right_local)
            N = min(N_left, N_right)

        X_left  = X_bal[left_local]
        X_right = X_bal[right_local]

        edges, present, max_c = self._build_bidirectional_knn_edges(
            X_left, X_right, k=self.k
        )

        pairs        = None
        dropped_left  = []
        dropped_right = []

        if mode == "shfd":
            if shfd_schedule is None:
                shfd_schedule = [10, 20, 40]

            all_pairs = []

            # unmatched tracks LOCAL indices into left_local / right_local
            unmatched_left  = list(range(N_left))
            unmatched_right = list(range(N_right))

            for k_round in shfd_schedule:
                if not unmatched_left or not unmatched_right:
                    break

                new_pairs, unmatched_left, unmatched_right = \
                    self._shfd_one_round(
                        X_left, X_right,
                        unmatched_left, unmatched_right,
                        k=k_round,
                    )
                all_pairs.extend(new_pairs)

                print(
                    f"[SHFD] round k={k_round}: "
                    f"+{len(new_pairs)} pairs, "
                    f"{len(unmatched_left)} left / "
                    f"{len(unmatched_right)} right still unmatched"
                )

            # residual solver
            if unmatched_left and unmatched_right:
                print(f"[SHFD] residual solver='{shfd_residual}' on "
                      f"{len(unmatched_left)} left, "
                      f"{len(unmatched_right)} right")

                if shfd_residual == "drop":
                    # Leave residuals unmatched
                    dropped_left  = unmatched_left
                    dropped_right = unmatched_right

                else:
                    # Build a denser kNN graph for the residual sub-problem
                    k_res = min(
                        max(shfd_schedule) * 2,
                        len(unmatched_left),
                        len(unmatched_right),
                    )
                    k_res = max(k_res, 1)

                    X_res_left  = X_left[unmatched_left]
                    X_res_right = X_right[unmatched_right]
                    Nres = min(len(unmatched_left), len(unmatched_right))
                    Nres_L = len(unmatched_left)
                    Nres_R = len(unmatched_right)

                    edges_res, _, _ = self._build_bidirectional_knn_edges(
                        X_res_left[:Nres], X_res_right[:Nres], k=k_res
                    )

                    edges_res, _, _ = self._build_bidirectional_knn_edges(
                        X_res_left, X_res_right, k=k_res
                    )
        
                    if shfd_residual == "greedy":
                        pairs_res, dropL_res, dropR_res = \
                            self._solve_greedy_matching_rect(Nres_L, Nres_R, edges_res)

                    elif shfd_residual == "mcmm":
                        pairs_res, dropL_res, dropR_res = \
                            self._solve_mcmm_via_flow(Nres_L, Nres_R, edges_res)

                    else:
                        raise ValueError(
                            f"Unknown shfd_residual='{shfd_residual}'. "
                            "Choose 'greedy', 'mcmm', or 'drop'."
                        )

                    # Lift residual sub-indices back to original local indices
                    res_pairs_lifted = [
                        (unmatched_left[i], unmatched_right[j])
                        for i, j in pairs_res
                    ]
                    all_pairs.extend(res_pairs_lifted)

                    # Any still unmatched after residual solver
                    matched_res_L = {i for i, _ in pairs_res}
                    matched_res_R = {j for _, j in pairs_res}
                    dropped_left  = [unmatched_left[i]
                                     for i in range(Nres)
                                     if i not in matched_res_L]
                    dropped_right = [unmatched_right[j]
                                     for j in range(Nres)
                                     if j not in matched_res_R]
                    # Points beyond Nres (when sizes differ) are also dropped
                    dropped_left  += unmatched_left[Nres:]
                    dropped_right += unmatched_right[Nres:]

            pairs = all_pairs

        else:
            raise ValueError(f"Unknown mode='{mode}'.")

        fairlets        = []
        fairlet_centers = []
        fairlet_costs   = []
        fairlet_centers_idx = []

        for i, j in pairs:
            li     = int(left_local[i])
            rj     = int(right_local[j])
            l_orig = int(used_idx[li])
            r_orig = int(used_idx[rj])

            fairlets.append([l_orig, r_orig])
            c = float(distance(self.data[l_orig], self.data[r_orig]))
            fairlet_costs.append(c)
            fairlet_centers.append((self.data[l_orig] + self.data[r_orig]) / 2)
            if underrepresented == "left":
                fairlet_centers_idx.append(l_orig)
            else:
                fairlet_centers_idx.append(r_orig)

        # Collect dropped indices in original space
        dropped_orig = []
        for i in dropped_left:
            dropped_orig.append(int(used_idx[int(left_local[i])]))
        for j in dropped_right:
            dropped_orig.append(int(used_idx[int(right_local[j])]))

        self.dropped_by_matching_ = np.asarray(dropped_orig, dtype=int)
        return fairlets, fairlet_centers, fairlet_centers_idx, fairlet_costs
        


class VanillaFairletDecomposition(object):
    """
    Computes vanilla fairlet decomposition that ensures fair clusters. It might not give the optimal cost value.
    """

    def __init__(self, p, q, blues, reds, data):
        """
        p (int) : First balance parameter 
        q (int) : Second balance parameter
        blues (list) : Index of the points corresponding to first class
        reds (list) : Index of the points corresponding to second class
        data (list) : Contains actual data points
        """
        self.p = p
        self.q = q
        self.blues = blues
        self.reds = reds
        self.data = data

    def balanced(self, r, b):
        """
        Checks for initial balance and feasibility.

        Args:
            r (int) : Total length of majority class
            b (int) : Total length of minority class

        Returns:
            bool value indicating whether the balance is possible
        """
        if r == 0 and b == 0:
            return True
        if r == 0 or b == 0:
            return False
        return min(float(r / b), float(b / r)) >= float(self.p / self.q)
 
    def make_fairlet(self, points, dataset, fairlets, fairlet_centers, costs):
        """
        Adds fairlet to the fairlet decomposition and returns the k-center cost for the fairlet.
        
        Args:
            points (list) : Index of the points that comprise the fairlet
            dataset (list) : Original data
            fairlets (list)
            fairlet_centers (list)
            costs (list)
        """
        # Finding the point as center whose maximum distance from any point is minimum
        cost_list = [(i, max([distance(dataset[i], dataset[j]) for j in points])) for i in points]
        cost_list = sorted(cost_list, key=lambda x:x[1], reverse=False)
        center, cost = cost_list[0][0], cost_list[0][1]
        
        # Adding the shortlisted points to the fairlets
        fairlets.append(points)
        fairlet_centers.append(center)
        costs.append(cost)
        return
     
    def decompose(self):
        """
        Computes vanilla (p , q) - fairlet decomposition of given points as per Lemma 3 in NeurIPS 2017 paper.
        Assumes that balance parameters are non-negative integers such that gcd(p, q) = 1.
        Also assumes that balance of reds and blues is atleast p/q.

        Returns:
            fairlets (list) 
            fairlet_centers (list)
            costs (list)
        """
        assert gcd(self.p, self.q) == 1, 'Please ensure that the GCD of balance parameters is 1.'
        assert self.p <= self.q, 'Please use balance parameters such that p <= q.'
        
        fairlets = []
        fairlet_centers = []
        fairlet_costs = []
        
        if len(self.reds) < len(self.blues): # We want the reds to be bigger in size as they correspond to 'q' parameter
            temp = self.blues
            self.blues = self.reds
            self.reds = temp
            
        R = len(self.reds)
        B = len(self.blues)
        
        assert self.balanced(R, B), 'Input sets are unbalanced: ' + str(R) + ' , ' + str(B)
     
        # If both reds and blues are empty, return empty results
        if R == 0 and B == 0:
            return fairlets, fairlet_centers, fairlet_costs
     
        b = 0
        r = 0
        
        #random.seed(42)
        random.shuffle(self.reds)
        random.shuffle(self.blues)
            
        while ((R - r) - (B - b)) >= (self.q - self.p) and (R - r) >= self.q and (B - b) >= self.p:
            self.make_fairlet(self.reds[r: (r + self.q)] + self.blues[b: (b + self.p)], self.data, fairlets, fairlet_centers, fairlet_costs)
            r += self.q
            b += self.p
        if ((R - r) + (B - b)) >= 1 and ((R - r) + (B - b)) <= (self.p + self.q):
            self.make_fairlet(self.reds[r:] + self.blues[b:], self.data, fairlets, fairlet_centers, fairlet_costs)
            r = R
            b = B
        elif ((R - r) != (B - b)) and ((B - b) >= self.p):
            self.make_fairlet(self.reds[r: r + (R - r) - (B - b) + self.p] + self.blues[b: (b + self.p)], self.data, fairlets,
                                     fairlet_centers, fairlet_costs)
            r += (R - r) - (B - b) + self.p
            b += self.p
        assert (R - r) == (B - b), 'Error in computing fairlet decomposition.'
        for i in range(R - r):    
            self.make_fairlet([self.reds[r + i], self.blues[b + i]], self.data, fairlets, fairlet_centers, fairlet_costs)

        print("%d fairlets have been identified."%(len(fairlet_centers)))
        assert len(fairlets) == len(fairlet_centers)
        assert len(fairlet_centers) == len(fairlet_costs)
        
        #return fairlets, fairlet_centers, fairlet_costs
        #NEW
        centroids = np.array([self.data[i] for i in fairlet_centers])
        return fairlets, centroids, fairlet_costs

class MCFFairletDecomposition(object):
    """
    Computes the optimized version of fairlet decomposition using minimum-cost flow.
    """

    def __init__(self, blues, reds, t, distance_threshold, data):
        """
        blues (list) : Index of the points corresponding to first class
        reds (list) : Index of the points corresponding to second class
        t (int) : (1, t) is the fairness ratio to be enforced
        distance_threshold (int) : Value to be used for pushing cost to infinity
        data (list) : Contains actual data points
        """
        self.blues = blues
        self.blue_nodes = len(blues)
        self.reds = reds
        self.red_nodes = len(reds)

        assert self.blue_nodes >= self.red_nodes

        self.t = t
        self.distance_threshold = distance_threshold
        self.data = data

        # Initializing the Graph
        self.G = nx.DiGraph()

    def compute_distances(self):
        """
        Compute distances between every pair of blue and red nodes.
        """
        random.seed(42)
        random.shuffle(self.blues)
        random.shuffle(self.reds)

        self.distances = {}
        for idx, i in enumerate(self.blues):
            for idx2, j in enumerate(self.reds):
                self.distances['B_%d_R_%d'%(idx+1, idx2+1)] = distance(self.data[i], self.data[j])

    def build_graph(self, plot_graph=False, weight_limit=10000000):
        """
        Builds the graph i.e. nodes and edges.

        Args:
            plot_graph (bool) : Indicates whether the graph needs to be plotted
            weight_limit (int) : Big value to be used in place of infinity for cost definition
        """

        self.G.add_node('beta', pos=(0, 4+(1+max(self.blue_nodes, self.red_nodes))/2), demand=(-1*self.red_nodes))
        self.G.add_node('ro', pos=(5, 4+(1+max(self.blue_nodes, self.red_nodes))/2), demand=(self.blue_nodes))
        self.G.add_edge('beta', 'ro', weight=0, capacity=min(self.blue_nodes, self.red_nodes))

        for i in range(self.blue_nodes):
            self.G.add_node('B%d'%(i+1), pos=(1, i+1), demand=-1)
            self.G.add_edge('beta', 'B%d'%(i+1), weight=0, capacity=self.t-1)
        for i in range(self.red_nodes):
            self.G.add_node('R%d'%(i+1), pos=(4, i+1), demand=1)
            self.G.add_edge('R%d'%(i+1), 'ro', weight=0, capacity=self.t-1)
            
        # Latent nodes
        for i in range(self.blue_nodes):
            for j in range(self.t):
                position = (i+1) + ((i+1 - i) / self.t)*j
                self.G.add_node('B%d_%d'%(i+1, j+1), pos=(2, position), demand=0)
                self.G.add_edge('B%d'%(i+1), 'B%d_%d'%(i+1, j+1), weight=0, capacity=1)
        for i in range(self.red_nodes):
            for j in range(self.t):
                position = (i+1) + ((i+1 - i) / self.t)*j
                self.G.add_node('R%d_%d'%(i+1, j+1), pos=(3, position), demand=0)
                self.G.add_edge('R%d_%d'%(i+1, j+1), 'R%d'%(i+1), weight=0, capacity=1)
                
        # Adding edges between latent nodes
        for i in range(self.blue_nodes):
            for j in range(self.t):
                for k in range(self.red_nodes):
                    for l in range(self.t):
                        dist = self.distances['B_%d_R_%d'%(i+1, k+1)]
                        if dist <= self.distance_threshold:
                            self.G.add_edge('B%d_%d'%(i+1, j+1), 'R%d_%d'%(k+1, l+1), weight=1, capacity=1)
                        else:
                            self.G.add_edge('B%d_%d'%(i+1, j+1), 'R%d_%d'%(k+1, l+1), weight=weight_limit, capacity=1)

        if plot_graph:
            if self.blue_nodes > 10:
                print("Graph can't be plotted because the blue nodes exceed 10.")
            else:
                plt.figure(figsize=(10, 8))
                pos = {n : (x, y) for (n, (x, y)) in nx.get_node_attributes(self.G, 'pos').items()}
                nx.draw_networkx_nodes(self.G, pos, node_size=1000, alpha=0.5)
                nx.draw_networkx_labels(self.G, pos, font_size=11)
                nx.draw_networkx_edges(self.G, pos)
                plt.show()

    def decompose(self):
        """
        Calls the network simplex to run the MCF algorithm.
        Computes the fairlets and fairlet centers.

        Returns:
            fairlets (list) 
            fairlet_centers (list)
            costs (list)
        """

        start_time = time.time()
        flow_cost, flow_dict = nx.network_simplex(self.G)
        print("Time taken to compute MCF solution - %.3f seconds."%(time.time() - start_time))

        fairlets = {}
        # Assumes mapping from blue nodes to the red nodes
        for i in flow_dict.keys():
            if 'B' in i and '_' in i:
                if sum(flow_dict[i].values()) == 1:
                    for j in flow_dict[i].keys():
                        if flow_dict[i][j] == 1:
                            if j.split('_')[0] not in fairlets:
                                fairlets[j.split('_')[0]] = [i.split('_')[0]]
                            else:
                                fairlets[j.split('_')[0]].append(i.split('_')[0])
                
        fairlets = [([a] + b) for a, b in fairlets.items()]

        fairlets2 = []
        for i in fairlets:
            curr_fairlet = []
            for j in i:
                if 'R' in j:
                    d = self.reds
                else:
                    d = self.blues
                curr_fairlet.append(d[int(j[1:]) - 1])
            fairlets2.append(curr_fairlet)
        fairlets = fairlets2
        del fairlets2

        # Choosing fairlet centers
        fairlet_centers = []
        fairlet_costs = []

        for f in fairlets:
            cost_list = [(i, max([distance(self.data[i], self.data[j]) for j in f])) for i in f]
            cost_list = sorted(cost_list, key=lambda x:x[1], reverse=False)
            center, cost = cost_list[0][0], cost_list[0][1]
            fairlet_centers.append(center)
            fairlet_costs.append(cost)

        print("%d fairlets have been identified."%(len(fairlet_centers)))
        assert len(fairlets) == len(fairlet_centers)
        assert len(fairlet_centers) == len(fairlet_costs)
        centroids = np.array([self.data[i] for i in fairlet_centers])

        return fairlets, centroids, fairlet_costs