"""
Scalable Fair Clustering (SFC) with k-means macro-clustering.

This is a corrected port of Backurs et al. (2019) "Scalable Fair Clustering",
combining the original three-phase fairlet decomposition (Chierichetti et al.
2017 / Backurs et al. 2019) with a k-means macro-clustering step instead of
k-medoids. The k-means step yields fixed cluster centers in feature space,
which support stateless nearest-centroid inference for new samples via
standard sklearn `predict` semantics.

Classes
-------
TreeNode
    Quadtree node used by the SFC fairlet decomposition. Holds the cluster
    of point indices and the per-color sub-lists populated by
    `populate_colors`.

FairletDecomposition
    Implements the three-phase fairlet decomposition:
      Phase 1 (must-remove):  push up unbalanceable excess from each child
      Phase 2 (may-remove):   pull additional points until parent is balanced
      Phase 3 (unsaturated):  dissolve unsaturated fairlets if still unbalanced
    Returns a list of fairlets (each a list of point indices), the medoid
    index per fairlet, and the total fairlet decomposition cost.

SFC_KMeans
    End-to-end SFC clustering with sklearn-style fit/predict interface.
    Builds a quadtree, runs fairlet decomposition, computes fairlet centroids
    in feature space, and fits k-means on those centroids. Inference for new
    samples uses the standard nearest-centroid lookup over the fitted k-means
    cluster centers.
"""

import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans


EPSILON = 1e-4


# ------------------------------------------------------------------ #
# Quadtree node                                                      #
# ------------------------------------------------------------------ #

class TreeNode:
    """Quadtree node for the SFC fairlet decomposition."""

    def __init__(self):
        self.children = []
        self.cluster = []
        self.reds = []
        self.blues = []

    def set_cluster(self, cluster):
        self.cluster = list(cluster)

    def add_child(self, child):
        self.children.append(child)

    def populate_colors(self, colors):
        """Populate per-node red/blue lists bottom-up."""
        self.reds = []
        self.blues = []
        if len(self.children) == 0:
            for i in self.cluster:
                if colors[i] == 0:
                    self.reds.append(i)
                else:
                    self.blues.append(i)
        else:
            for child in self.children:
                child.populate_colors(colors)
                self.reds.extend(child.reds)
                self.blues.extend(child.blues)


# ------------------------------------------------------------------ #
# Fairlet decomposition                                              #
# ------------------------------------------------------------------ #

class FairletDecomposition:
    """
    Three-phase fairlet decomposition of Backurs et al. (2019).

    After calling `tree_fairlet_decomposition(p, q, root, dataset, colors)`,
    the following attributes are populated:

        fairlets         : list[list[int]]  — each fairlet's point indices
        fairlet_centers  : list[int]        — medoid index for each fairlet
        cost             : float            — total decomposition cost
    """

    def __init__(self):
        self.fairlets = []
        self.fairlet_centers = []
        self.cost = 0.0
        self.colors = None

    # -------- helpers -------- #

    @staticmethod
    def balanced(p, q, r, b):
        if r == 0 and b == 0:
            return True
        if r == 0 or b == 0:
            return False
        return min(r * 1.0 / b, b * 1.0 / r) >= p * 1.0 / q

    def make_fairlet(self, points, dataset):
        """Append a fairlet, store its medoid index, return its medoid cost."""
        self.fairlets.append(list(points))
        cost_list = [
            sum(np.linalg.norm(dataset[c] - dataset[pt]) for pt in points)
            for c in points
        ]
        cost, idx = min((c, i) for i, c in enumerate(cost_list))
        self.fairlet_centers.append(points[idx])
        return cost

    # -------- decomposition -------- #

    def basic_fairlet_decomposition(self, p, q, blues, reds, dataset):
        """
        (p, q)-fairlet decomposition for a single balanced point set.
        Faithful port of Lemma 3 in Chierichetti et al. (2017).
        """
        assert p <= q, "Balance parameters must satisfy p <= q."
        if len(reds) < len(blues):
            reds, blues = blues, reds
        R, B = len(reds), len(blues)
        assert self.balanced(p, q, R, B), \
            f"Input sets are unbalanced: {R}, {B}"
        if R == 0 and B == 0:
            return 0

        r0 = b0 = 0
        cost = 0
        # Saturated (q reds + p blues) fairlets while we have spare reds.
        while (R - r0) - (B - b0) >= q - p and R - r0 >= q and B - b0 >= p:
            cost += self.make_fairlet(
                reds[r0:r0 + q] + blues[b0:b0 + p], dataset
            )
            r0 += q
            b0 += p

        # Tail handling: either one bundled fairlet, or a partial fairlet
        # plus a row of single (red, blue) pairs.
        if 1 <= (R - r0) + (B - b0) <= p + q:
            cost += self.make_fairlet(reds[r0:] + blues[b0:], dataset)
            r0, b0 = R, B
        elif R - r0 != B - b0 and B - b0 >= p:
            take_r = (R - r0) - (B - b0) + p
            cost += self.make_fairlet(
                reds[r0:r0 + take_r] + blues[b0:b0 + p], dataset
            )
            r0 += take_r
            b0 += p

        assert R - r0 == B - b0, "Error in computing fairlet decomposition."
        for i in range(R - r0):
            cost += self.make_fairlet([reds[r0 + i], blues[b0 + i]], dataset)
        return cost

    def node_fairlet_decomposition(self, p, q, node, dataset, donelist, depth=0):
        """Recursive three-phase decomposition over the quadtree."""

        # ---- LEAF ----
        if len(node.children) == 0:
            node.reds = [r for r in node.reds if donelist[r] == 0]
            node.blues = [b for b in node.blues if donelist[b] == 0]
            assert self.balanced(p, q, len(node.reds), len(node.blues)), \
                "Reached unbalanced leaf."
            return self.basic_fairlet_decomposition(
                p, q, node.blues, node.reds, dataset
            )

        # ---- NON-LEAF ----
        for child in node.children:
            child.reds = [r for r in child.reds if donelist[r] == 0]
            child.blues = [b for b in child.blues if donelist[b] == 0]

        R = [len(child.reds) for child in node.children]
        B = [len(child.blues) for child in node.children]

        if sum(R) == 0 or sum(B) == 0:
            assert sum(R) == 0 and sum(B) == 0, \
                "One color class is empty for this node while the other is not."
            return 0

        NR = 0  # reds the parent pulls upward
        NB = 0  # blues the parent pulls upward

        # ---- Phase 1: must-remove ----
        for i in range(len(node.children)):
            if R[i] >= B[i]:
                must_remove_red = max(0, R[i] - int(np.floor(B[i] * q / p)))
                R[i] -= must_remove_red
                NR += must_remove_red
            else:
                must_remove_blue = max(0, B[i] - int(np.floor(R[i] * q / p)))
                B[i] -= must_remove_blue
                NB += must_remove_blue

        # How many of the under-represented color must we still pull up?
        if NR >= NB:
            missing = max(0, int(np.ceil(NR * p / q)) - NB)
        else:
            missing = max(0, int(np.ceil(NB * p / q)) - NR)

        # ---- Phase 2: may-remove ----
        for i in range(len(node.children)):
            if missing == 0:
                break
            if NR >= NB:
                may_remove_blue = B[i] - int(np.ceil(R[i] * p / q))
                remove_blue = min(may_remove_blue, missing)
                B[i] -= remove_blue
                NB += remove_blue
                missing -= remove_blue
            else:
                may_remove_red = R[i] - int(np.ceil(B[i] * p / q))
                remove_red = min(may_remove_red, missing)
                R[i] -= remove_red
                NR += remove_red
                missing -= remove_red

        # ---- Phase 3: unsaturated fairlets ----
        for i in range(len(node.children)):
            if self.balanced(p, q, NR, NB):
                break
            if R[i] >= B[i]:
                num_saturated = R[i] // q
                excess_red = R[i] - q * num_saturated
                excess_blue = B[i] - p * num_saturated
            else:
                num_saturated = B[i] // q
                excess_red = R[i] - p * num_saturated
                excess_blue = B[i] - q * num_saturated
            R[i] -= excess_red
            B[i] -= excess_blue
            NR += excess_red
            NB += excess_blue

        assert self.balanced(p, q, NR, NB), \
            "Parent-level sets are unbalanced after Phase 3."

        # ---- Aggregate upward-pulled points and mark them done ----
        # Each child keeps its first R[i]/B[i] points; the rest go up.
        reds, blues = [], []
        for i, child in enumerate(node.children):
            for r in child.reds[R[i]:]:
                reds.append(r)
                donelist[r] = 1
            for b in child.blues[B[i]:]:
                blues.append(b)
                donelist[b] = 1

        assert len(reds) == NR and len(blues) == NB, \
            "Mismatch between accumulated counts and aggregated points."

        # Decompose this level's pulled-up points, then recurse into children
        # (each child still owns its first R[i]/B[i] points).
        cost = self.basic_fairlet_decomposition(p, q, blues, reds, dataset)
        cost += sum(
            self.node_fairlet_decomposition(p, q, child, dataset, donelist, depth + 1)
            for child in node.children
        )
        return cost

    def tree_fairlet_decomposition(self, p, q, root, dataset, colors):
        """Run the full fairlet decomposition over a quadtree."""
        assert p <= q, "Balance parameters must satisfy p <= q."
        self.colors = list(colors)
        root.populate_colors(self.colors)
        assert self.balanced(p, q, len(root.reds), len(root.blues)), \
            "Dataset is unbalanced — cannot satisfy (p, q)."
        donelist = [0] * dataset.shape[0]
        self.cost = self.node_fairlet_decomposition(
            p, q, root, dataset, donelist, 0
        )
        return self.cost


# ------------------------------------------------------------------ #
# Quadtree construction                                              #
# ------------------------------------------------------------------ #

def build_quadtree(dataset, max_levels=0, random_shift=True, random_state=None):
    """
    Build a (randomly shifted) quadtree over `dataset`.

    If max_levels=0, the tree partitions until every leaf cluster is a
    singleton (or the cell is below EPSILON in all dimensions).
    """
    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    dimension = dataset.shape[1]
    lower = np.amin(dataset, axis=0).astype(float)
    upper = np.amax(dataset, axis=0).astype(float)

    shift = np.zeros(dimension)
    if random_shift:
        for d in range(dimension):
            spread = upper[d] - lower[d]
            shift[d] = rng.uniform(0, spread)
            upper[d] += spread

    return _build_quadtree_aux(
        dataset, list(range(dataset.shape[0])), lower, upper, max_levels, shift
    )


def _build_quadtree_aux(dataset, cluster, lower, upper, max_levels, shift):
    dimension = dataset.shape[1]

    cell_too_small = all((upper[i] - lower[i]) <= EPSILON for i in range(dimension))

    node = TreeNode()
    if max_levels == 1 or len(cluster) <= 1 or cell_too_small:
        node.set_cluster(cluster)
        return node

    midpoint = 0.5 * (lower + upper)
    subclusters = defaultdict(list)
    for i in cluster:
        key = tuple(
            (dataset[i, d] + shift[d]) <= midpoint[d] for d in range(dimension)
        )
        subclusters[key].append(i)

    for edge, subcluster in subclusters.items():
        sub_lower = np.zeros(dimension)
        sub_upper = np.zeros(dimension)
        for d in range(dimension):
            if edge[d]:
                sub_lower[d] = lower[d]
                sub_upper[d] = midpoint[d]
            else:
                sub_lower[d] = midpoint[d]
                sub_upper[d] = upper[d]
        node.add_child(
            _build_quadtree_aux(
                dataset, subcluster, sub_lower, sub_upper, max_levels - 1, shift
            )
        )
    return node


# ------------------------------------------------------------------ #
# SFC + k-means wrapper                                              #
# ------------------------------------------------------------------ #

class SFC:
    """
    Scalable Fair Clustering with a k-means macro-clustering step.

    Fits in two stages:
      1. Quadtree + three-phase fairlet decomposition (Backurs et al. 2019).
      2. k-means over the fairlet centroids in feature space.

    The fitted k-means model produces fixed cluster centers, so inference
    for new samples reduces to a standard nearest-centroid lookup. This
    makes SFC_KMeans suitable as a drop-in clustering component in
    pipelines that require stateless, single-point inference.

    Parameters
    ----------
    p, q : int
        Balance parameters with p <= q and gcd(p, q) = 1. Each fairlet
        targets a (p:q) ratio between the two protected groups.
    k : int
        Number of macro-clusters (k for k-means).
    random_state : int or None
        Seed for the quadtree shift and k-means initialisation.
    max_levels : int
        Maximum quadtree depth (0 = unlimited until singletons).
    n_init : int
        Number of k-means initialisations (passed through to sklearn).

    Attributes (after fit)
    ----------------------
    fairlets_ : list[list[int]]
        Indices of training points in each fairlet.
    fairlet_centroids_ : ndarray of shape (n_fairlets, d)
        Mean-position centroid of each fairlet (used as k-means input).
    fairlet_medoids_ : list[int]
        Medoid index of each fairlet (preserved for compatibility).
    cluster_centers_ : ndarray of shape (k, d)
        Fitted k-means cluster centers (the fixed model parameters).
    training_labels_ : ndarray of shape (n,)
        Cluster assignments for the training data.
    fairlet_cost_ : float
        Total fairlet decomposition cost.
    """

    def __init__(self, p, q, k, random_state=0, max_levels=0, n_init=10):
        self.p = p
        self.q = q
        self.k = k
        self.random_state = random_state
        self.max_levels = max_levels
        self.n_init = n_init

        self.fairlet_decomposition_ = None
        self.kmeans_ = None
        self.fairlets_ = None
        self.fairlet_centroids_ = None
        self.fairlet_medoids_ = None
        self.training_labels_ = None
        self.fairlet_cost_ = None

    # -------- fit / predict -------- #

    def fit(self, X, colors):
        """
        Fit the SFC clustering on (X, colors).

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Feature matrix.
        colors : array-like of length n
            Binary protected attribute (values in {0, 1}).
        """
        X = np.asarray(X, dtype=float)
        colors = list(colors)

        # 1. Quadtree
        root = build_quadtree(
            X, max_levels=self.max_levels, random_state=self.random_state
        )

        # 2. Fairlet decomposition
        self.fairlet_decomposition_ = FairletDecomposition()
        self.fairlet_cost_ = self.fairlet_decomposition_.tree_fairlet_decomposition(
            self.p, self.q, root, X, colors
        )

        self.fairlets_ = self.fairlet_decomposition_.fairlets
        self.fairlet_medoids_ = self.fairlet_decomposition_.fairlet_centers

        # 3. Compute fairlet centroids (means in feature space, not medoids).
        #    These are what k-means clusters over.
        self.fairlet_centroids_ = np.array([
            X[fairlet].mean(axis=0) for fairlet in self.fairlets_
        ])

        # 4. k-means over fairlet centroids — yields fixed centers.
        self.kmeans_ = KMeans(
            n_clusters=self.k,
            n_init=self.n_init,
            random_state=self.random_state,
        ).fit(self.fairlet_centroids_)

        # 5. Training-time assignments (over the original points, not centroids)
        self.training_labels_ = self.kmeans_.predict(X)
        return self

    def predict(self, X_new):
        """
        Assign new samples to clusters via nearest-centroid lookup.

        Stateless: depends only on `X_new` and the stored cluster centers.
        """
        if self.kmeans_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return self.kmeans_.predict(X_new)

    def fit_predict(self, X, colors):
        """Fit on (X, colors), return training-time cluster assignments."""
        self.fit(X, colors)
        return self.training_labels_

    # -------- accessors -------- #

    @property
    def cluster_centers_(self):
        """Fixed cluster centers, usable for stateless inference."""
        if self.kmeans_ is None:
            raise RuntimeError("Call fit() before accessing cluster_centers_.")
        return self.kmeans_.cluster_centers_

    @property
    def n_fairlets_(self):
        return len(self.fairlets_) if self.fairlets_ is not None else None