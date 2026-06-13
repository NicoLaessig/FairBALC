import math
import numpy as np
from typing import List, Optional, Union
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class FairBoost(BaseEstimator, ClassifierMixin):
    """
    information [...]

    References:
        ...
    """
    def __init__(
        self,
        df_dict,
        classifier,
        estimators: int = 50,
        eps: float = 0.1,
        k: int = 30,
        standardize: bool = True,
        use_proba: bool = True,
        ensure_mixed: bool = True,
        k_max: Optional[int] = None,
        debug: bool = False,
    ):
        self.df_dict = df_dict
        self.classifier = classifier
        self.estimators = estimators
        self.eps = eps
        self.k = k
        self.standardize = standardize
        self.use_proba = use_proba
        self.ensure_mixed = ensure_mixed
        self.k_max = k_max
        self.debug = debug

        # learned/public attributes
        self.trained_estimator: List = []
        self.alphas: List[float] = []

        # internals
        self._sens_name: Optional[str] = None
        self._favored_val: Optional[Union[int, float, str]] = None
        self._scaler: Optional[StandardScaler] = None
        self._all_neighbors: Optional[np.ndarray] = None  # (n, Kmax) positional indices

    def _scores_01(self, est, X: "pd.DataFrame") -> np.ndarray:
        """Return a score per sample in [0,1]-ish for PPR and ensemble."""
        if self.use_proba and hasattr(est, "predict_proba"):
            return est.predict_proba(X)[:, 1].astype(float)
        if hasattr(est, "decision_function"):
            s = est.decision_function(X).astype(float)
            # per-model min-max scale
            smin = np.min(s)
            sptp = np.ptp(s)
            return (s - smin) / (sptp + 1e-12)
        return est.predict(X).astype(float)

    def _build_knn_space(self, X: "pd.DataFrame") -> np.ndarray:
        """Non-sensitive, optionally standardized matrix for k-NN."""
        sens_cols = self.df_dict["sens_attrs"]
        if len(sens_cols) != 1:
            raise ValueError("FairBoost expects exactly one sensitive attribute.")
        X_nosens = X.drop(columns=sens_cols).to_numpy()
        if self.standardize:
            self._scaler = StandardScaler().fit(X_nosens)
            return self._scaler.transform(X_nosens)
        self._scaler = None
        return X_nosens

    def _precompute_neighbors(self, Xn: np.ndarray):
        """Precompute neighbors up to Kmax once; drop self if present."""
        n = Xn.shape[0]
        base_k = min(self.k, max(1, n - 1))
        Kmax = self.k_max if self.k_max is not None else min(n - 1, max(5 * base_k, base_k + 1))
        # +1 for potential self removal
        nn = NearestNeighbors(n_neighbors=min(Kmax + 1, n), algorithm="auto", metric="euclidean").fit(Xn)
        idx = nn.kneighbors(Xn, return_distance=False)  # (n, Kmax+1 or n)
        rows = np.arange(n)
        # drop self if it's the first neighbor
        if idx.shape[1] > 0 and np.all(idx[:, 0] == rows):
            idx = idx[:, 1:]
        # ensure we have exactly Kmax columns (pad last neighbor if needed)
        if idx.shape[1] < Kmax:
            pad = np.tile(idx[:, -1:], (1, Kmax - idx.shape[1]))
            idx = np.hstack([idx, pad])
        self._all_neighbors = idx[:, :Kmax]  # (n, Kmax)

    def _mixed_slice(self, i: int, sens: np.ndarray, favored_val, base_k: int) -> np.ndarray:
        """Return the smallest prefix of neighbors for i that contains both groups (>= base_k if possible)."""
        row = self._all_neighbors[i]
        Kmax = row.shape[0]
        # at least base_k
        k_needed = max(base_k, 1)
        # grow until both groups present or reach Kmax
        while k_needed <= Kmax:
            neigh = row[:k_needed]
            sn = sens[neigh]
            if (sn == favored_val).any() and (sn != favored_val).any():
                return neigh
            k_needed += 1 if self.ensure_mixed else 1e9  # break if ensure_mixed=False
        # fallback: return base_k even if not mixed (δ will stay 0)
        return row[:base_k]

    def fit(self, X, y):
        """Train FairBoost ensemble."""
        X = X.copy()
        y = np.asarray(y)
        n = len(X)

        self._sens_name = self.df_dict["sens_attrs"][0]
        self._favored_val = self.df_dict["favored"]
        sens = X[self._sens_name].to_numpy()

        # initial uniform weights
        sample_weight = np.full(n, 1.0 / max(n, 1), dtype=float)

        # build k-NN space and neighbors once
        Xn = self._build_knn_space(X)
        self._precompute_neighbors(Xn)
        base_k = min(self.k, max(1, n - 1))

        # reset ensemble
        self.trained_estimator = []
        self.alphas = []

        for t in range(self.estimators):
            est = clone(self.classifier)
            # IMPORTANT: pass keyword sample_weight
            est.fit(X, y, sample_weight=sample_weight)

            yh = self._scores_01(est, X)  # [0,1]-ish scores

            # k-NN situation testing to set δ per point
            delta = np.zeros(n, dtype=int)
            c = 0
            for i in range(n):
                neigh = self._mixed_slice(i, sens, self._favored_val, base_k)
                sn = sens[neigh]
                if (sn == self._favored_val).any() and (sn != self._favored_val).any():
                    yh_nei = yh[neigh]
                    m_f = (sn == self._favored_val)
                    m_d = ~m_f
                    ppr_fav = float(yh_nei[m_f].mean())
                    ppr_dis = float(yh_nei[m_d].mean())
                    if abs(ppr_fav - ppr_dis) > self.eps:
                        delta[i] = 1
                        c += 1
                # else: delta[i] stays 0 (no mixed neighborhood up to Kmax)

            # weighted unfairness rate and alpha
            eps_t = float(np.dot(sample_weight, delta) / (sample_weight.sum() + 1e-12))
            eps_t = min(max(eps_t, 1e-12), 1.0 - 1e-12)
            alpha_t = math.log((1.0 - eps_t) / eps_t)

            # store round
            self.trained_estimator.append(est)
            self.alphas.append(alpha_t)

            # unfairness-based weight update + normalize
            sample_weight *= np.exp(alpha_t * delta)
            sw_sum = sample_weight.sum()
            sample_weight = (sample_weight / sw_sum) if sw_sum > 0 else np.full(n, 1.0 / max(n, 1))

            #if self.debug:
             #   print(f"[t={t}] mean(delta)={delta.mean():.4f}  eps_t={eps_t:.4g}  alpha={alpha_t:.3f}  "
              #        f"w_min={sample_weight.min():.2e}  w_max={sample_weight.max():.2e}")

        return self

    def predict_proba(self, X):
        """Weighted-average probability across base learners (paper uses averaging)."""
        if not self.trained_estimator:
            raise RuntimeError("Call fit(...) before predict_proba(...).")

        alphas = np.asarray(self.alphas, dtype=float)
        denom = alphas.sum() if alphas.size else 1.0

        # prefer predict_proba
        if all(hasattr(e, "predict_proba") for e in self.trained_estimator):
            P = np.vstack([e.predict_proba(X)[:, 1] for e in self.trained_estimator])
            ens = (alphas @ P) / denom
            ens = np.clip(ens, 0.0, 1.0)
            return np.c_[1.0 - ens, ens]

        # decision_function fallback
        if all(hasattr(e, "decision_function") for e in self.trained_estimator):
            S = np.vstack([e.decision_function(X).astype(float) for e in self.trained_estimator])
            S = (S - S.min(axis=1, keepdims=True)) / (S.ptp(axis=1, keepdims=True) + 1e-12)
            ens = (alphas @ S) / denom
            ens = np.clip(ens, 0.0, 1.0)
            return np.c_[1.0 - ens, ens]

        # last resort: average hard labels
        P = np.vstack([e.predict(X).astype(float) for e in self.trained_estimator])
        ens = (alphas @ P) / denom
        ens = np.clip(ens, 0.0, 1.0)
        return np.c_[1.0 - ens, ens]

    def predict(self, X):
        """Threshold the probability ensemble at 0.5."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    def staged_predict(self, X):
        """
        Generator yielding ensemble predictions after each round (1..T).
        Useful for diagnostics and plots.
        """
        alphas = []
        ens_scores = None
        for t, est in enumerate(self.trained_estimator, start=1):
            a = self.alphas[t - 1]
            alphas.append(a)
            denom = sum(alphas)

            if hasattr(est, "predict_proba"):
                s = est.predict_proba(X)[:, 1].astype(float)
            elif hasattr(est, "decision_function"):
                d = est.decision_function(X).astype(float)
                s = (d - d.min()) / (np.ptp(d) + 1e-12)
            else:
                s = est.predict(X).astype(float)

            if ens_scores is None:
                ens_scores = a * s
            else:
                ens_scores = ens_scores + a * s

            yield (ens_scores / denom >= 0.5).astype(int)


    def fit_old(self, X_train, y_train):
        dataset_length = len(X_train)
        sample_weight = [1/dataset_length for i in range(dataset_length)]
        estimator = self.classifier
        self.trained_estimator = []
        self.alphas = []
        X_train_nosens = X_train.drop(self.df_dict["sens_attrs"], axis=1)

        neighbor_dict = dict()
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree').fit(X_train_nosens.values)
        for i in range(len(X_train_nosens)):
            indices = nbrs.kneighbors(X_train_nosens.iloc[i].values.reshape(1, -1), return_distance=False)
            real_indices = X_train_nosens.index[indices.flatten()].tolist()
            neighbor_dict[i] = real_indices
        for i in range(self.estimators):
            error = math.inf
            X_train = X_train.loc[:, X_train.columns != "prediction"]
            current_estimator = estimator.fit(X_train, y_train, sample_weight=sample_weight)
            estimator_predictions = current_estimator.predict(X_train)
            X_train["prediction"] = estimator_predictions
            full_error = 0
            error_fair = 0
            wrong_classified = []
            for j in range(len(estimator_predictions)):
                full_error += sample_weight[j]
                nearest_neighbors_df = X_train.loc[neighbor_dict[j]]
                favored_val = 0
                discriminated = 0
                grouped_df = nearest_neighbors_df.groupby(self.df_dict["sens_attrs"])
                if len(grouped_df) == 2:
                    for key, item in grouped_df:
                        if len(self.df_dict["sens_attrs"]) == 1:
                            key = key[0]
                        part_df = grouped_df.get_group(key)
                        if key == self.df_dict["favored"]:
                            total_fav = len(part_df)
                            for l, row in part_df.iterrows():
                                favored_val += row["prediction"]
                        else:
                            total_discr = len(part_df)
                            for l, row in part_df.iterrows():
                                discriminated += row["prediction"]
                    ppr_fav = favored_val/total_fav
                    ppr_discr = discriminated/total_discr
                    if abs(ppr_fav - ppr_discr) > self.eps:
                        error_fair += sample_weight[j]
                        wrong_classified.append(j)

            error = error_fair/(full_error + 1e-24)
            alpha = np.log((1-error)/(error + 1e-24))

            self.trained_estimator.append(current_estimator)
            self.alphas.append(alpha)

            for j in wrong_classified:
                sample_weight[j] = sample_weight[j] * math.exp(alpha)

        return self