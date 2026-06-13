from __future__ import annotations
"""Implements the LOG-Means [1] algorithm for parameter estimation of KMeans.
[1] Fritz, M., Behringer, M., Schwarz, H. "LOG-Means: Efficiently Estimating
    the Number of Clusters in Large Datasets". 2020.
"""
from sklearn.cluster import KMeans

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from sklearn.cluster import KMeans


def log_means(X, k_low, k_high):
    """Uses the LOG-means algorithm to estimate the perfect amount of clusters.

    Parameters
    ----------
    X: DataFrame, shape (n_samples, m_features)
        Dataset on which clustering will be performed.

    k_low: int
        The minimum amount of clusters that should be generated.

    k_high: int
        The maximum amount of clusters that should be generated.


    Returns/Output
    ----------
    parameter_best: int
        Returns the best estimated parameter within the given range [k_low, k_high].
    """
    k_low = k_low - 1
    K = []
    M = {}
    SSE_low = KMeans(k_low).fit(X).inertia_
    K.append((k_low, SSE_low))
    SSE_high = KMeans(k_high).fit(X).inertia_
    K.append((k_high, SSE_high))
    count = 0
    while k_high != k_low + 1:
        k_mid = int((k_high + k_low)/2)
        SSE_mid = KMeans(k_mid).fit(X).inertia_
        K.append((k_mid, SSE_mid))
        ratio_left = SSE_low/SSE_mid
        ratio_right = SSE_mid/SSE_high
        if ratio_left >= ratio_right:
            k_high = k_mid
            #k_low = k_low
        else:
            #k_high = k_high
            k_low = k_mid
        K = sorted(K, key=lambda x: x[0])
        for pos, t in enumerate(K):
            if t[0] == k_high:
                k_high_id = pos
                break
        SSE_high = K[k_high_id][1]
        SSE_low = K[k_high_id-1][1]
        count += 1

    try:
        if float(M[k_high]) >= float(M[k_low]):
            parameter_best = k_high
        else:
            parameter_best = k_low
    except KeyError:
        parameter_best = k_high

    return parameter_best


@dataclass
class LogMeansResult:
    k_est: int
    sse_by_k: Dict[int, float]
    ratio_by_k: Dict[int, float]          # M: key is right endpoint k
    evaluated_ks: List[int]               # sorted keys in K


def log_means_new(
    X,
    k_low: int,
    k_high: int,
    epsilon: int = 0,
    *,
    kmeans_kwargs: Optional[Dict[str, Any]] = None,
) -> LogMeansResult:
    """
    LOG-Means (Fritz et al., 2020/2022) for estimating k for centroid-based clustering.

    Implements Algorithm 1 (LOG-Means), including the optional epsilon-neighborhood step.
    See Algorithm 1, lines 1-37 in the paper. :contentReference[oaicite:2]{index=2}

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    k_low : int
        Minimum desired clusters (min(R)). Must be >= 2.
    k_high : int
        Maximum desired clusters (max(R)). Must be > k_low.
    epsilon : int, default=0
        Number of neighbors to evaluate around the estimated bend (optional step).
        Matches Algorithm 1 lines 21-35. :contentReference[oaicite:3]{index=3}
    kmeans_kwargs : dict, optional
        Extra kwargs for sklearn.cluster.KMeans, e.g.
        {"init":"k-means++", "n_init":20, "random_state":0, "max_iter":300}

    Returns
    -------
    LogMeansResult
        k_est plus caches (SSE and ratios) that can be inspected / reused.
    """
    if kmeans_kwargs is None:
        # Good defaults for ratio-based decisions: more stable SSE.
        kmeans_kwargs = {"init": "k-means++", "n_init": 20, "random_state": 0}

    if k_low < 2:
        raise ValueError("k_low must be >= 2 (LOG-Means uses k_low-1 internally).")
    if k_high <= k_low:
        raise ValueError("k_high must be > k_low.")
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0.")

    # Algorithm 1 line 1: klow <- klow - 1 :contentReference[oaicite:4]{index=4}
    klow = k_low - 1
    khigh = k_high

    # K: store SSE for executed k; M: store ratios keyed by right endpoint k :contentReference[oaicite:5]{index=5}
    sse_by_k: Dict[int, float] = {}
    ratio_by_k: Dict[int, float] = {}

    def sse(k: int) -> float:
        if k < 1:
            raise ValueError("Internal k became < 1; check k_low.")
        if k not in sse_by_k:
            km = KMeans(n_clusters=k, **kmeans_kwargs)
            km.fit(X)
            sse_by_k[k] = float(km.inertia_)
        return sse_by_k[k]

    def sorted_ks() -> List[int]:
        return sorted(sse_by_k.keys())

    def left_adjacent(k: int) -> int:
        ks = sorted_ks()
        idx = ks.index(k)
        if idx == 0:
            raise ValueError("No left-adjacent value exists for the smallest k in K.")
        return ks[idx - 1]

    def update_ratio_for_right_endpoint(k_right: int) -> None:
        """M stores ratio between k_right and its left adjacent k_prev."""
        k_prev = left_adjacent(k_right)
        ratio_by_k[k_right] = sse(k_prev) / sse(k_right)

    def argmax_ratio_in_range(k_min: int, k_max: int) -> int:
        """Pick k with highest ratio among keys in ratio_by_k within [k_min, k_max]."""
        candidates = [(k, r) for k, r in ratio_by_k.items() if k_min <= k <= k_max]
        if not candidates:
            # Fallback: pick current khigh if no ratios yet (shouldn't happen after first mid)
            return k_max
        return max(candidates, key=lambda t: t[1])[0]

    # Init (Algorithm 1 lines 4-7) :contentReference[oaicite:6]{index=6}
    sse(klow)
    sse(khigh)

    # Main loop (Algorithm 1 lines 8-20) :contentReference[oaicite:7]{index=7}
    while khigh != klow + 1:
        kmid = (khigh + klow) // 2  # line 9 :contentReference[oaicite:8]{index=8}
        if kmid == klow:  # safety (should not happen when khigh != klow+1)
            kmid = klow + 1
        if kmid == khigh:
            kmid = khigh - 1

        sse(kmid)  # lines 10-11 :contentReference[oaicite:9]{index=9}

        # After inserting kmid, ratios to update are:
        # (klow, kmid) stored at key kmid  and (kmid, khigh) stored at key khigh
        # lines 12-15 :contentReference[oaicite:10]{index=10}
        update_ratio_for_right_endpoint(kmid)
        update_ratio_for_right_endpoint(khigh)

        # line 16: khigh <- k with highest ratio from M :contentReference[oaicite:11]{index=11}
        khigh = max(ratio_by_k.items(), key=lambda t: t[1])[0]

        # line 17: klow <- left adjacent value of khigh from K :contentReference[oaicite:12]{index=12}
        klow = left_adjacent(khigh)

        # lines 18-19 are implicit since we retrieve SSE via sse(...)
        # next iteration uses the updated klow/khigh

    # Optional epsilon refinement (Algorithm 1 lines 21-35) :contentReference[oaicite:13]{index=13}
    if epsilon > 0:
        kbend = argmax_ratio_in_range(klow + 1, khigh)  # line 22 concept :contentReference[oaicite:14]{index=14}

        half = epsilon // 2
        k_min = max(k_low, kbend - half)     # keep within user range
        k_max = min(k_high, kbend + half)

        # Ensure we can compute ratios (need k-1 available). We’ll compute sequentially.
        for k in range(k_min, k_max + 1):
            sse(k)

        # Update ratios for all k in [k_min, k_max] using adjacent predecessor
        for k in range(k_min, k_max + 1):
            # Ensure predecessor exists in K:
            # If k is the smallest in the window and predecessor not evaluated, evaluate it too.
            if k - 1 >= 1 and (k - 1) not in sse_by_k:
                sse(k - 1)
            # Now compute ratio between left-adjacent evaluated k and k
            # Algorithm 1 line 33 uses SSE_{k_prev} / SSE_k, where k_prev is previous in the loop. :contentReference[oaicite:15]{index=15}
            # We'll use literal k-1 if available; otherwise use left_adjacent from K.
            if (k - 1) in sse_by_k:
                ratio_by_k[k] = sse_by_k[k - 1] / sse_by_k[k]
            else:
                update_ratio_for_right_endpoint(k)

        k_est = argmax_ratio_in_range(k_min, k_max)  # line 36 :contentReference[oaicite:16]{index=16}
    else:
        # With epsilon=0, final estimate is the k with max ratio in M (Algorithm 1 note). :contentReference[oaicite:17]{index=17}
        k_est = max(ratio_by_k.items(), key=lambda t: t[1])[0] if ratio_by_k else khigh

    return LogMeansResult(
        k_est=k_est,
        sse_by_k=sse_by_k,
        ratio_by_k=ratio_by_k,
        evaluated_ks=sorted_ks(),
    )
