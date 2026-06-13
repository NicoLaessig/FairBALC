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
    1) Undersample majority so |majority| == |minority| = N
    2) Build bidirectional kNN edges between the two groups (k=50 recommended)
    3) Try min-cost PERFECT matching on sparse graph via min-cost flow
    4) If infeasible, fallback to min-cost flow with SLACK (dummy nodes) + penalties:
         - real-left can match dummy-right (penalty P)  => left dropped
         - dummy-left can match real-right (penalty P)  => right dropped
         - dummy-left <-> dummy-right (0) completes the assignment
    5) Return pair fairlets for real-real matches only.
       Dropped nodes are stored in dropped_by_slack_ and can be assigned later via km.predict.
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
        slack_penalty=None,   # if None: auto based on observed knn max distance
    ):
        self.a = np.asarray(a).ravel()
        self.data = np.asarray(data)
        self.y = np.asarray(y)

        self.k = int(k)
        self.bal = bal
        self.random_state = int(random_state)
        self.cost_scale = int(cost_scale)
        self.slack_penalty = slack_penalty  # in *distance units*, not scaled int

        # outputs for downstream use
        self.used_indices_ = None
        self.dropped_majority_indices_ = None
        self.dropped_by_slack_ = None  # indices (in original index space) dropped by slack solver
        self.dropped_by_matching_ = None

    # ----------------------------
    # helpers
    # ----------------------------
    def _filter_edges_same_label(self, edges, y_left, y_right):
        """
        Keep only edges (i,j,_) such that y_left[i] == y_right[j]
        """
        y_left = np.asarray(y_left).ravel()
        y_right = np.asarray(y_right).ravel()
        filtered = [(i, j, c) for (i, j, c) in edges if y_left[i] == y_right[j]]
        max_c = 0.0
        for _i, _j, c in filtered:
            if c > max_c:
                max_c = float(c)
        return filtered, max_c


    def _binary_remap_majority_minority(self):
        vals, counts = np.unique(self.a, return_counts=True)
        if len(vals) != 2:
            raise ValueError(f"Requires binary a; got {vals}.")
        maj = vals[np.argmax(counts)]
        mino = vals[np.argmin(counts)]
        # 0 = majority, 1 = minority
        a01 = np.where(self.a == maj, 0, 1).astype(int)
        return a01

    def _undersample_to_balance(self, a01):
        rus = RandomUnderSampler(sampling_strategy="auto", random_state=self.random_state)
        rus.fit_resample(self.data, a01)

        used_idx = np.asarray(rus.sample_indices_, dtype=int)
        self.used_indices_ = used_idx

        mask = np.ones(len(a01), dtype=bool)
        mask[used_idx] = False
        self.dropped_majority_indices_ = np.where(mask)[0]

        return used_idx

    def _build_bidirectional_knn_edges(self, X_left, X_right, k):
        """
        Build union of:
          - left -> kNN(right)
          - right -> kNN(left) (added as left-edge too)
        Returns:
          edges: list[(i, j, cost_float)]
          present: set[(i,j)]
          max_cost: max cost in edges
        """
        n_left, n_right = X_left.shape[0], X_right.shape[0]
        if n_left == 0 or n_right == 0:
            return [], set(), 0.0

        kL = min(max(1, k), n_right)
        kR = min(max(1, k), n_left)

        edges = []
        present = set()
        max_c = 0.0

        # left -> right
        nnR = NearestNeighbors(n_neighbors=kL, algorithm="auto")
        nnR.fit(X_right)
        _, idxsR = nnR.kneighbors(X_left, return_distance=True)
        for i in range(n_left):
            for pos in range(kL):
                j = int(idxsR[i, pos])
                if (i, j) in present:
                    continue
                c = float(distance(X_left[i], X_right[j]))
                edges.append((i, j, c))
                present.add((i, j))
                if c > max_c:
                    max_c = c

        # right -> left  (add as left-edge)
        nnL = NearestNeighbors(n_neighbors=kR, algorithm="auto")
        nnL.fit(X_left)
        _, idxsL = nnL.kneighbors(X_right, return_distance=True)
        for j in range(n_right):
            for pos in range(kR):
                i = int(idxsL[j, pos])
                if (i, j) in present:
                    continue
                c = float(distance(X_left[i], X_right[j]))
                edges.append((i, j, c))
                present.add((i, j))
                if c > max_c:
                    max_c = c

        return edges, present, max_c

    # ----------------------------
    # perfect matching attempt
    # ----------------------------
    def _solve_min_cost_perfect_matching(self, N, edges):
        """
        Min-cost perfect matching via min-cost flow on an N x N bipartite graph.
        """
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
            w = int(round(self.cost_scale * c))
            G.add_edge(f"L{i}", f"R{j}", capacity=1, weight=w)

        flow = nx.min_cost_flow(G)  # raises NetworkXUnfeasible if no perfect matching

        match = [-1] * N
        for i in range(N):
            out = flow[f"L{i}"]
            for node, fval in out.items():
                if fval == 1 and node.startswith("R"):
                    match[i] = int(node[1:])
                    break
            if match[i] < 0:
                raise RuntimeError("Could not extract perfect matching from flow.")
        return match

    # ----------------------------
    # fallback: slack / penalties
    # ----------------------------
    def _solve_min_cost_with_slack(self, N, edges, max_edge_cost):
        """
        Build an always-feasible assignment of size 2N with dummy nodes.

        Left side:  real L0..L(N-1)    + dummy DL0..DL(N-1)
        Right side: real R0..R(N-1)    + dummy DR0..DR(N-1)

        We send 2N units of flow:
          s -> each left (real + dummy) cap 1
          each right (real + dummy) -> t cap 1

        Edges:
          real Li -> real Rj   : cost = dist (only provided by kNN sparsification)
          real Li -> dummy DRk : cost = P      (unmatch a left)
          dummy DLk -> real Rj : cost = P      (unmatch a right)
          dummy DLk -> dummy DRm : cost = 0    (complete the rest)

        This is guaranteed feasible and penalizes dropping.
        """

        # Choose penalty P (in integer cost units)
        if self.slack_penalty is None:
            # "big enough" to prefer any available real-real edge
            # P := 3 * max_kNN_edge_cost (distance units), at least 1.0
            P_dist = max(1.0, 3.0 * float(max_edge_cost))
        else:
            P_dist = float(self.slack_penalty)

        P = int(round(self.cost_scale * P_dist))

        G = nx.DiGraph()
        s, t = "s", "t"
        G.add_node(s, demand=-(2 * N))
        G.add_node(t, demand=+(2 * N))

        # nodes
        for i in range(N):
            G.add_node(f"L{i}", demand=0)     # real left
            G.add_node(f"R{i}", demand=0)     # real right
            G.add_node(f"DL{i}", demand=0)    # dummy left
            G.add_node(f"DR{i}", demand=0)    # dummy right

            G.add_edge(s, f"L{i}", capacity=1, weight=0)
            G.add_edge(s, f"DL{i}", capacity=1, weight=0)

            G.add_edge(f"R{i}", t, capacity=1, weight=0)
            G.add_edge(f"DR{i}", t, capacity=1, weight=0)

        # real-real (sparse)
        for i, j, c in edges:
            w = int(round(self.cost_scale * c))
            G.add_edge(f"L{i}", f"R{j}", capacity=1, weight=w)

        # real-left -> dummy-right (drop left) : connect to all DRk (dense but fallback only)
        for i in range(N):
            for k in range(N):
                G.add_edge(f"L{i}", f"DR{k}", capacity=1, weight=P)

        # dummy-left -> real-right (drop right) : connect to all Rj (dense but fallback only)
        for k in range(N):
            for j in range(N):
                G.add_edge(f"DL{k}", f"R{j}", capacity=1, weight=P)

        # dummy-left -> dummy-right : 0 cost complete bipartite (dense but fallback only)
        for k in range(N):
            for m in range(N):
                G.add_edge(f"DL{k}", f"DR{m}", capacity=1, weight=0)

        flow = nx.min_cost_flow(G)

        # Extract assignments among 2N left nodes
        # We'll read only real-left -> real-right matches as "pairs".
        real_pairs = []
        dropped_left = []
        dropped_right = []

        matched_real_right = set()

        # real left decisions
        for i in range(N):
            out = flow[f"L{i}"]
            chosen = None
            for node, fval in out.items():
                if fval == 1:
                    chosen = node
                    break
            if chosen is None:
                raise RuntimeError("Slack flow extraction failed for real left.")

            if chosen.startswith("R"):
                j = int(chosen[1:])
                real_pairs.append((i, j))
                matched_real_right.add(j)
            else:
                # matched to dummy right => left dropped
                dropped_left.append(i)

        # rights not matched by any real left will be matched by dummy-left (and pay penalty)
        for j in range(N):
            if j not in matched_real_right:
                dropped_right.append(j)

        return real_pairs, dropped_left, dropped_right

    # ----------------------------
    # fallback: max-cardinality matching (no costs)
    # ----------------------------
    def _solve_max_cardinality_matching_old(self, N, edges):
        """
        Returns:
          pairs: list[(i_left, j_right)]
          dropped_left: list[i_left]
          dropped_right: list[j_right]
        """
        B = nx.Graph()
        left_nodes = [f"L{i}" for i in range(N)]
        right_nodes = [f"R{j}" for j in range(N)]
        B.add_nodes_from(left_nodes, bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)

        for i, j, _c in edges:
            B.add_edge(f"L{i}", f"R{j}")

        matching = nx.algorithms.bipartite.matching.maximum_matching(B, top_nodes=left_nodes)

        pairs = []
        matched_left = set()
        matched_right = set()

        for Li in left_nodes:
            if Li in matching:
                Rj = matching[Li]
                if Rj.startswith("R"):
                    i = int(Li[1:])
                    j = int(Rj[1:])
                    pairs.append((i, j))
                    matched_left.add(i)
                    matched_right.add(j)

        dropped_left = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right

    def _solve_max_cardinality_matching(self, N_left, N_right, edges):
        B = nx.Graph()
        left_nodes = [f"L{i}" for i in range(N_left)]
        right_nodes = [f"R{j}" for j in range(N_right)]
        B.add_nodes_from(left_nodes, bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)

        for i, j, _c in edges:
            B.add_edge(f"L{i}", f"R{j}")

        matching = nx.algorithms.bipartite.matching.maximum_matching(B, top_nodes=left_nodes)

        pairs = []
        matched_left = set()
        matched_right = set()
        for Li in left_nodes:
            if Li in matching:
                Rj = matching[Li]
                if Rj.startswith("R"):
                    pairs.append((int(Li[1:]), int(Rj[1:])))
                    matched_left.add(int(Li[1:]))
                    matched_right.add(int(Rj[1:]))

        dropped_left = [i for i in range(N_left) if i not in matched_left]
        dropped_right = [j for j in range(N_right) if j not in matched_right]
        return pairs, dropped_left, dropped_right

    def _solve_min_cost_maximum_matching(self, N, edges):
        """
        Min-Cost Maximum Matching (MCMM) on a sparse bipartite graph.

        Returns:
          pairs: list[(i_left, j_right)]  # real-real matches
          dropped_left: list[i_left]      # unmatched real left indices
          dropped_right: list[j_right]    # unmatched real right indices
        """
        # --- Build augmented bipartite graph ---
        # Real nodes
        left_real  = [f"L{i}" for i in range(N)]
        right_real = [f"R{j}" for j in range(N)]

        # Dummy nodes (same count so a perfect matching always exists)
        left_dummy  = [f"DL{i}" for i in range(N)]
        right_dummy = [f"DR{j}" for j in range(N)]

        top_nodes = left_real + left_dummy
        bottom_nodes = right_real + right_dummy

        B = nx.Graph()
        B.add_nodes_from(top_nodes, bipartite=0)
        B.add_nodes_from(bottom_nodes, bipartite=1)

        # Compute BIG_M large enough so the solver prefers any real edge over any dummy edge
        # Safe choice: BIG_M > maximum possible total real cost of any matching.
        # If max edge cost is max_c, then total real cost <= N*max_c, so choose (N+1)*max_c + 1.
        max_c = 0.0
        for _i, _j, c in edges:
            if c is None:
                continue
            if c > max_c:
                max_c = float(c)
        BIG_M = (N + 1) * (max_c + 1.0) + 1.0

        # Add sparse real edges with their costs
        for i, j, c in edges:
            B.add_edge(f"L{i}", f"R{j}", weight=float(c))

        # Add dummy edges:
        # - Any real left can match some dummy right (means "left is dropped") with penalty BIG_M
        for i in range(N):
            Li = f"L{i}"
            for j in range(N):
                DRj = f"DR{j}"
                B.add_edge(Li, DRj, weight=BIG_M)

        # - Any dummy left can match any real right (means "right is dropped") with penalty BIG_M
        for i in range(N):
            DLi = f"DL{i}"
            for j in range(N):
                Rj = f"R{j}"
                B.add_edge(DLi, Rj, weight=BIG_M)

        # - Dummy lefts match dummy rights at ~0 cost to fill remaining slots
        for i in range(N):
            DLi = f"DL{i}"
            for j in range(N):
                DRj = f"DR{j}"
                B.add_edge(DLi, DRj, weight=0.0)

        # --- Solve min-cost perfect matching on augmented graph ---
        # This guarantees:
        # 1) maximum number of real-real matches (minimizes #dummy edges)
        # 2) among those, minimum total real cost
        matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
            B, top_nodes=top_nodes, weight="weight"
        )

        # --- Extract real-real pairs and dropped nodes ---
        pairs = []
        matched_left = set()
        matched_right = set()

        for Li in left_real:
            mate = matching.get(Li, None)
            if mate is None:
                continue
            if mate.startswith("R"):  # real-right matched => keep
                i = int(Li[1:])
                j = int(mate[1:])
                pairs.append((i, j))
                matched_left.add(i)
                matched_right.add(j)
            # else matched to DR* => dropped_left

        dropped_left = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right


    # ----------------------------
    # Greedy min-cost matching on sparse edges
    # ----------------------------
    def _solve_greedy_matching(self, N, edges):
        """
        edges: list of (i, j, cost)
        Returns pairs (i, j), dropped_left, dropped_right
        """
        edges_sorted = sorted(edges, key=lambda x: x[2])
        usedL = set()
        usedR = set()
        pairs = []

        for i, j, c in edges_sorted:
            if i not in usedL and j not in usedR:
                usedL.add(i)
                usedR.add(j)
                pairs.append((i, j))

        dropped_left = [i for i in range(N) if i not in usedL]
        dropped_right = [j for j in range(N) if j not in usedR]
        return pairs, dropped_left, dropped_right


    # ----------------------------
    # MCMM via dummy nodes + minimum_weight_full_matching
    # WARNING: adds O(N^2) dummy edges, only good for modest N
    # ----------------------------
    def _solve_mcmm_with_dummies(self, N, edges):
        """
        Min-Cost Maximum Matching (MCMM) using dummy nodes and
        minimum_weight_full_matching on an augmented graph.

        Returns:
          pairs (real-real), dropped_left, dropped_right
        """
        left_real  = [f"L{i}" for i in range(N)]
        right_real = [f"R{j}" for j in range(N)]
        left_dummy  = [f"DL{i}" for i in range(N)]
        right_dummy = [f"DR{j}" for j in range(N)]

        top_nodes = left_real + left_dummy

        B = nx.Graph()
        B.add_nodes_from(top_nodes, bipartite=0)
        B.add_nodes_from(right_real + right_dummy, bipartite=1)

        max_c = 0.0
        for _i, _j, c in edges:
            max_c = max(max_c, float(c))
        BIG_M = (N + 1) * (max_c + 1.0) + 1.0

        # real edges
        for i, j, c in edges:
            B.add_edge(f"L{i}", f"R{j}", weight=float(c))

        # real-left to dummy-right (drop left)
        for i in range(N):
            Li = f"L{i}"
            for j in range(N):
                B.add_edge(Li, f"DR{j}", weight=BIG_M)

        # dummy-left to real-right (drop right)
        for i in range(N):
            DLi = f"DL{i}"
            for j in range(N):
                B.add_edge(DLi, f"R{j}", weight=BIG_M)

        # dummy-left to dummy-right (fill)
        for i in range(N):
            DLi = f"DL{i}"
            for j in range(N):
                B.add_edge(DLi, f"DR{j}", weight=0.0)

        matching = nx.algorithms.bipartite.matching.minimum_weight_full_matching(
            B, top_nodes=top_nodes, weight="weight"
        )

        pairs = []
        matched_left = set()
        matched_right = set()

        for Li in left_real:
            mate = matching.get(Li, None)
            if mate is None:
                continue
            if mate.startswith("R"):
                i = int(Li[1:])
                j = int(mate[1:])
                pairs.append((i, j))
                matched_left.add(i)
                matched_right.add(j)

        dropped_left = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right


    # ----------------------------
    # MCMM via sparse min-cost flow:
    # 1) compute maximum cardinality m* (Hopcroft–Karp)
    # 2) solve min-cost flow of value m* on sparse edges
    # ----------------------------
    def _solve_mcmm_via_flow(self, N, edges):
        """
        MCMM on sparse graph without dummy-complete O(N^2) edges.

        Returns:
          pairs (i, j), dropped_left, dropped_right
        """
        # 1) max-cardinality size m*
        B = nx.Graph()
        left_nodes = [f"L{i}" for i in range(N)]
        right_nodes = [f"R{j}" for j in range(N)]
        B.add_nodes_from(left_nodes, bipartite=0)
        B.add_nodes_from(right_nodes, bipartite=1)
        for i, j, _c in edges:
            B.add_edge(f"L{i}", f"R{j}")

        hk = nx.algorithms.bipartite.matching.maximum_matching(B, top_nodes=left_nodes)
        pairs_hk = []
        usedL = set()
        usedR = set()
        for Li in left_nodes:
            if Li in hk and hk[Li].startswith("R"):
                i = int(Li[1:])
                j = int(hk[Li][1:])
                pairs_hk.append((i, j))
                usedL.add(i)
                usedR.add(j)
        m_star = len(pairs_hk)

        if m_star == 0:
            # nothing matchable
            return [], list(range(N)), list(range(N))

        # 2) min-cost flow of value m* on sparse edges
        G = nx.DiGraph()
        s = "s"
        t = "t"

        # node demands: supply at s is -m*, demand at t is +m* (network_simplex uses demand sign convention)
        # In NetworkX: demand>0 means net inflow required, demand<0 means net outflow required.
        G.add_node(s, demand=-m_star)
        G.add_node(t, demand=+m_star)

        for i in range(N):
            G.add_node(f"L{i}", demand=0)
        for j in range(N):
            G.add_node(f"R{j}", demand=0)

        # s -> L (cap 1)
        for i in range(N):
            G.add_edge(s, f"L{i}", capacity=1, weight=0)

        # L -> R (cap 1, weight=cost) only for sparse edges
        for i, j, c in edges:
            G.add_edge(f"L{i}", f"R{j}", capacity=1, weight=int(1e6 * float(c)))  # integer weights for stability

        # R -> t (cap 1)
        for j in range(N):
            G.add_edge(f"R{j}", t, capacity=1, weight=0)

        try:
            flow_cost, flow_dict = nx.network_simplex(G)
        except nx.NetworkXUnfeasible:
            # should not happen if m_star computed correctly, but keep safe
            return pairs_hk, [i for i in range(N) if i not in usedL], [j for j in range(N) if j not in usedR]

        pairs = []
        matched_left = set()
        matched_right = set()

        for i in range(N):
            Li = f"L{i}"
            for Rj, f in flow_dict[Li].items():
                if f == 1 and Rj.startswith("R"):
                    j = int(Rj[1:])
                    pairs.append((i, j))
                    matched_left.add(i)
                    matched_right.add(j)

        dropped_left = [i for i in range(N) if i not in matched_left]
        dropped_right = [j for j in range(N) if j not in matched_right]
        return pairs, dropped_left, dropped_right


    # ----------------------------
    # Greedy then MCMM-flow on leftovers
    # ----------------------------
    def _solve_greedy_then_flow(self, N, edges):
        """
        First do greedy min-cost matching.
        Then run MCMM-flow on the leftover induced subproblem.
        """
        pairs_g, droppedL, droppedR = self._solve_greedy_matching(N, edges)
        if not droppedL or not droppedR:
            return pairs_g, droppedL, droppedR

        # remap leftover indices to [0..nL-1], [0..nR-1]
        mapL = {old: new for new, old in enumerate(droppedL)}
        mapR = {old: new for new, old in enumerate(droppedR)}
        edges_sub = []
        for i, j, c in edges:
            if i in mapL and j in mapR:
                edges_sub.append((mapL[i], mapR[j], c))

        nL = len(droppedL)
        nR = len(droppedR)
        Nsub = min(nL, nR)
        # If not equal, cut to Nsub on both sides to keep consistent with your 1:1 setup
        # (alternatively you can allow partial match; MCMM-flow will do that anyway)
        # We’ll just run MCMM-flow on the subgraph as-is by setting Nsub = max(nL,nR) is not valid,
        # so do a consistent truncation:
        keptL = droppedL[:Nsub]
        keptR = droppedR[:Nsub]
        mapL = {old: new for new, old in enumerate(keptL)}
        mapR = {old: new for new, old in enumerate(keptR)}
        edges_sub = [(mapL[i], mapR[j], c) for i, j, c in edges if i in mapL and j in mapR]

        pairs_sub, dropL_sub, dropR_sub = self._solve_mcmm_via_flow(Nsub, edges_sub)

        # lift back to original indices
        pairs = pairs_g + [(keptL[i], keptR[j]) for i, j in pairs_sub]

        matchedL = {i for i, _ in pairs}
        matchedR = {j for _, j in pairs}
        dropped_left = [i for i in range(N) if i not in matchedL]
        dropped_right = [j for j in range(N) if j not in matchedR]
        return pairs, dropped_left, dropped_right


    # ----------------------------
    # main API
    # ----------------------------
    def decompose(self, mode="sparse"):
        a01 = self._binary_remap_majority_minority()
        if mode != "mcmm_nutb" and self.bal == "under":
            used_idx = self._undersample_to_balance(a01)
        else:
            used_idx = np.arange(len(self.data))

        X_bal = self.data[used_idx]
        a_bal = a01[used_idx]

        # left = majority(0), right = minority(1)
        left_local = np.where(a_bal == 0)[0]
        right_local = np.where(a_bal == 1)[0]

        if self.bal == "under":
            N = min(len(left_local), len(right_local))
            left_local = left_local[:N]
            right_local = right_local[:N]
        else:
             N_left = len(left_local)
             N_right = len(right_local)

        X_left = X_bal[left_local]
        X_right = X_bal[right_local]

        # 1) sparse bidirectional kNN edges (reduced k)
        edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=self.k)

        # 2) try min-cost perfect matching
        pairs = None
        dropped_left = []
        dropped_right = []

        #try:
        if mode == "sparse":
            match = self._solve_min_cost_perfect_matching(N, edges)
            pairs = [(i, match[i]) for i in range(N)]
        elif mode == "sparse_label":
            # 1) Filter to same-label edges only
            if self.y is None:
                raise ValueError("mode='sparse_label' requires y (labels) passed to the constructor.")

            # labels aligned with balanced arrays
            y_bal = self.y[used_idx]
            y_left = y_bal[left_local]
            y_right = y_bal[right_local]
            edges_lbl, max_c_lbl = self._filter_edges_same_label(edges, y_left, y_right)

            pairs, dropped_left, dropped_right = self._solve_max_cardinality_matching(N, edges_lbl)
            # If graph is too sparse / empty, you’ll almost surely be infeasible.
            # We'll try perfect matching, and if it fails, use slack fallback on label-filtered edges.
            #try:
            #match = self._solve_min_cost_perfect_matching(N, edges_lbl)
            #pairs = [(i, match[i]) for i in range(N)]
            #except nx.NetworkXUnfeasible:
                # label-aware slack fallback: keeps same-label edges only, drops otherwise
             #   pairs, dropped_left, dropped_right = self._solve_min_cost_with_slack(
              #      N, edges_lbl, max_edge_cost=max_c_lbl
               # )
        #except nx.NetworkXUnfeasible:
        elif mode == "sparse_fallback":
            # 3) fallback: max-cardinality matching + drop leftovers
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=20)
            if self.bal == "under":
                pairs, dropped_left, dropped_right = self._solve_max_cardinality_matching_old(N, edges)
            else:
                pairs, dropped_left, dropped_right = self._solve_max_cardinality_matching(N_left, N_right, edges)
        elif mode == "shfd":
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=40)
            pairs, dropped_left, dropped_right = self._solve_min_cost_maximum_matching(N, edges)
        elif mode in ("mcmm", "mcmm_nutb"):
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=40)
            pairs, dropped_left, dropped_right = self._solve_min_cost_maximum_matching(N, edges)
        elif mode == "mcmm_dummy":
            # MCMM via dummies (simple but O(N^2) dummy edges)
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=20)
            pairs, dropped_left, dropped_right = self._solve_mcmm_with_dummies(N, edges)

        elif mode == "mcmm_flow":
            # MCMM via sparse min-cost flow (recommended for larger N)
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=20)
            pairs, dropped_left, dropped_right = self._solve_mcmm_via_flow(N, edges)

        elif mode == "greedy":
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=20)
            pairs, dropped_left, dropped_right = self._solve_greedy_matching(N, edges)

        elif mode == "greedy_then_flow":
            edges, present, max_c = self._build_bidirectional_knn_edges(X_left, X_right, k=20)
            pairs, dropped_left, dropped_right = self._solve_greedy_then_flow(N, edges)


        # Convert to fairlets in ORIGINAL index space
        fairlets = []
        fairlet_centers = []
        fairlet_costs = []

        for i, j in pairs:
            li = int(left_local[i])
            rj = int(right_local[j])

            l_orig = int(used_idx[li])
            r_orig = int(used_idx[rj])

            fairlets.append([l_orig, r_orig])
            c = float(distance(self.data[l_orig], self.data[r_orig]))
            fairlet_costs.append(c)
            fairlet_centers.append((self.data[l_orig] + self.data[r_orig]) / 2)

        # dropped because max-cardinality < N (original indices)
        dropped_orig = []
        for i in dropped_left:
            li = int(left_local[i])
            dropped_orig.append(int(used_idx[li]))
        for j in dropped_right:
            rj = int(right_local[j])
            dropped_orig.append(int(used_idx[rj]))

        self.dropped_by_matching_ = np.asarray(dropped_orig, dtype=int)
        return fairlets, fairlet_centers, fairlet_costs


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