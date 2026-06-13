"""FALCC variants with fairlet-based clustering and ensembling."""

import copy
import math
import itertools
import joblib
import ast
import os
import re
import hashlib
import numpy as np
import pandas as pd
from collections import Counter

from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from algorithm.FALCCClassic_files import RunTraining
from algorithm.FALCCAddons import Metrics, Metrics2, Metrics_Reg, Metrics_Multi
from algorithm.FALCCAddons.parameter_estimation import log_means
from FCF.fairlet_decomposition import (
    VanillaFairletDecomposition,
    MCFFairletDecomposition,
    SparseFairletDecomposition
)

from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset
from typing import List, Tuple
from sklearn.pipeline import Pipeline


def _binary_group_codes(series):
    codes, uniques = pd.factorize(series)
    return codes.astype(int)


class FairletConstruction:
    """
    Fairlet-based clustering
    """
    def __init__(
        self,
        mode="vanilla",          # "vanilla", "mcf", or "shfd" => "shfd" is our implementation"
        p=1,                     # vanilla (p,q) parameters (see VanillaFairletDecomposition)
        q=2,
        t=2,                     # MCF (1,t)-fairness parameter (see MCFFairletDecomposition)
        distance_threshold=1e7,  # MCF threshold for infinity-like cost
        k=50,                    # sparse fairlet decomposition majority group chunk size
        n_clusters=None,         # optionally fix number of clusters; otherwise use log_means heuristic
        cluster_strat=("kmeans", "classic"),
        bal="under",             # for SHFD
        random_state=0,
        n_init="auto",
        max_iter=300,
    ):
        self.mode = mode
        self.p = p
        self.q = q
        self.t = t
        self.distance_threshold = distance_threshold
        self.k = k
        self.n_clusters = n_clusters
        self.bal = bal

        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

        self.cluster_centers_ = None
        self.labels_ = None
        self.clustersize = None
        self.cluster_strat = cluster_strat

        # Optional: keep fairlet info for inspection/debugging
        self.fairlets_ = None
        self.fairlet_center_indices_ = None
        self.fairlet_costs_ = None

        self._dropped_majority_indices = None
        self._dropped_by_matching = None

        self._fairlet_center_to_cluster = None    # np.ndarray shape (n_fairlets,) cluster id
        self._fairlet_center_vectors = None       # np.ndarray shape (n_fairlets, d)
        self._fairlet_nbrs = None                 # NearestNeighbors fitted on center vectors

    def _build_fairlets(self, X, a, y=None):
        """Build the fairlets via the chosen decomposition."""
        fairlet_centers_idx = None
        X = np.asarray(X)
        a = np.asarray(a).ravel()
        if y is not None:
            y = np.asarray(y).ravel()
        n = len(a)

        if n == 0:
            return [], [], []

        # Detect the majority/minority group and remap
        unique_vals, counts = np.unique(a, return_counts=True)
        if len(unique_vals) != 2:
            raise ValueError(
                f"FairletConstruction requires a binary protected attribute, "
                f"got values {unique_vals}."
            )

        majority_label = unique_vals[np.argmax(counts)]
        minority_label = unique_vals[np.argmin(counts)]

        a_remapped = np.where(a == majority_label, 0, 1)

        # Map the sensitive attribute to blues (majority) and reds (minority)
        blues = np.where(a_remapped == 0)[0].tolist()  # majority
        reds  = np.where(a_remapped == 1)[0].tolist()  # minority

        bal = int(np.ceil(len(blues) / len(reds)))

        print("start fairlets..")
        if self.mode == "vanilla":
            # (p,q)-fairlet decomposition
            decomp = VanillaFairletDecomposition(self.p, bal, blues, reds, X)
            fairlets, fairlet_centers, fairlet_costs = decomp.decompose()

        elif self.mode == "mcf":
            # Minimum-cost-flow-based (1,t)-fairlet decomposition
            decomp = MCFFairletDecomposition(
                blues=blues,
                reds=reds,
                t=bal,
                distance_threshold=self.distance_threshold,
                data=X,
            )
            decomp.compute_distances()
            decomp.build_graph(plot_graph=False)
            fairlets, fairlet_centers, fairlet_costs = decomp.decompose()

        elif "shfd" in self.mode:
            decomp = SparseFairletDecomposition(a=a, data=X, y=y, k=self.k, bal=self.bal, random_state=self.random_state)
            if self.mode == "shfd":
                mode_residual = ["shfd", "drop"]
            else:
                mode_residual = re.split("_", self.mode)
            fairlets, fairlet_centers, fairlet_centers_idx, fairlet_costs = decomp.decompose(mode=mode_residual[0], shfd_residual=mode_residual[1])
            self._dropped_majority_indices = decomp.dropped_majority_indices_
            self._dropped_by_matching = decomp.dropped_by_matching_
            self.fairlet_centers_idx = fairlet_centers_idx

        else:
            raise ValueError(f"mode must be 'vanilla', 'mcf' or 'shfd', got {self.mode!r}")
        
        # Store for inspection
        self.fairlets_ = fairlets
        self.fairlet_center_indices_ = fairlet_centers
        self.fairlet_costs_ = fairlet_costs

        return fairlets, fairlet_centers, fairlet_centers_idx, fairlet_costs


    def fit(self, X, a, y=None):
        """
        Information

        Args:

        Returns:
        """
        X = np.asarray(X)

        # Fairlet decomposition
        fairlets, fairlet_centers, fairlet_centers_idx, fairlet_costs = self._build_fairlets(X, a, y)

        if len(fairlets) == 0:
            # degenerate case
            self.labels_ = np.zeros(len(X), dtype=int)
            self.cluster_centers_ = np.mean(X, axis=0, keepdims=True)
            self.clustersize = 1
            return self

        # Representative points: the centers chosen by the decomposition
        try:
            centroids_median = X[np.asarray(fairlet_centers_idx, dtype=int)]
        except:
            pass
        centroids = np.array(fairlet_centers)

        # Choose the number of clusters
        if self.n_clusters is not None:
            self.clustersize = self.n_clusters
        elif self.n_clusters is not None and self.n_clusters < centroids.shape[0]:
            k_before = int(self.n_clusters)
            k_after = log_means(
                centroids,
                2,
                int(min(50, np.sqrt(centroids.shape[0]))),
            )
            self.clustersize = int(round(np.sqrt(k_before * k_after)))
        else:
            # LOGMeans heuristic on the centroids
            self.clustersize = log_means(
                centroids,
                2,
                int(min(50, np.sqrt(centroids.shape[0]))),
            )

        # Corner case: clustersize cannot exceed the number of fairlets or be < 1
        self.clustersize = max(1, min(self.clustersize, centroids.shape[0]))

        # Run the chosen clustering on the fairlet centers
        if self.cluster_strat[0] == "kmeans":
            km = KMeans(
                n_clusters=self.clustersize,
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
        elif self.cluster_strat[0] == "kmedoids":
            km = KMedoids(
                n_clusters=self.clustersize,
                random_state=self.random_state,
                max_iter=self.max_iter,
                metric='euclidean',
                method='alternate',
                init='k-medoids++',
            )

        if self.cluster_strat[1] == "classic":
            fl_center_labels = km.fit_predict(centroids)
        elif self.cluster_strat[1] == "median":
            fl_center_labels = km.fit_predict(centroids_median)
        self._fairlet_center_vectors = centroids
        self._fairlet_center_to_cluster = np.asarray(fl_center_labels, dtype=int)

        # NN index for post-hoc assignment
        self._fairlet_nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
        self._fairlet_nbrs.fit(self._fairlet_center_vectors)

        self.cluster_centers_ = km.cluster_centers_

        # Propagate the cluster labels back to all points through the fairlets
        labels = np.empty(X.shape[0], dtype=int)
        for j, fairlet in enumerate(fairlets):
            labels[fairlet] = fl_center_labels[j]

        # Assign labels to the majority points dropped by undersampling
        dropped = []
        if (
            hasattr(self, "_dropped_majority_indices")
            and self._dropped_majority_indices is not None
            and len(self._dropped_majority_indices) > 0
        ):
            dropped.append(self._dropped_majority_indices)
        if (
            hasattr(self, "_dropped_by_matching")
            and self._dropped_by_matching is not None
            and len(self._dropped_by_matching) > 0
        ):
            dropped.append(self._dropped_by_matching)
            
        if dropped:
            dropped = np.unique(np.concatenate(dropped)).astype(int)
            labels[dropped] = km.predict(X[dropped])

        self.labels_ = labels
        return self

    def fit_predict(self, X, a, y=None):
        self.fit(X, a, y)
        return self.labels_

    def predict(self, X):
        X = np.asarray(X)
        centers = self.cluster_centers_
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        return np.argmin(dists, axis=1)
    

    def predict_nn(self, X):
        X = np.asarray(X)

        # Fallback in case something wasn't fitted
        if self._fairlet_nbrs is None or self._fairlet_center_to_cluster is None:
            centers = self.cluster_centers_
            dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            return np.argmin(dists, axis=1)

        idx = self._fairlet_nbrs.kneighbors(X, return_distance=False).ravel()  # fairlet-center row ids
        return self._fairlet_center_to_cluster[idx]


    def return_clustersize(self):
        return self.clustersize


class FairBALC:
    """
    Implementation of our FairBALC submission

    References:
        ...
    """

    # "no" and "reweigh" are taken from FALCC approach. "shfd" is the option we use in our paper
    _ALLOWED_PROXIES = ("no", "reweigh", "vanilla", "mcf", "shfd")


    def __init__(
            self,
            link, df, input_file, df_dict, metric,
            lam=0.5,
            training="DiversePool",
            validationsize=0.4,
            proxy="no",
            cluster_algorithm="LOGmeans",
            ccr="[-1,-1]",
            k=20,
            bal="under",
            single_model_per_cluster="no",
            cluster_strat=0,
            remove=False,
            eval_strategy="comb",
            iteration=-1,
            ensembling_strategy="single",
            ensemble_size=None,
        ):
        # Restricted-value checks
        if training in ("no", "fair") or "fair" in training:
            raise ValueError(
                f"training={training!r} is not allowed; "
                f"the regression and 'fair' wrappers and the 'no' "
                f"(use-trained-models) path have been removed."
            )
        if proxy not in self._ALLOWED_PROXIES:
            raise ValueError(
                f"proxy={proxy!r} is not allowed; must be one of "
                f"{self._ALLOWED_PROXIES}."
            )

        self.link = link
        self.df = df
        self.input_file = input_file
        self.df_dict = df_dict
        self.metric = metric
        self.df_dict["metric"] = re.sub("bea_", "", metric)
        self.lam = lam
        self.training = training
        self.validationsize = validationsize
        self.proxy = proxy
        self.cluster_algorithm = cluster_algorithm
        self.ccr = ast.literal_eval(ccr) if isinstance(ccr, str) else ccr
        self.cluster_strat = {
            0: ("kmeans",   "classic"),
            1: ("kmeans",   "median"),
            2: ("kmedoids", "classic"),
            3: ("kmedoids", "median"),
        }.get(cluster_strat, ("kmeans", "classic"))
        self.remove = remove
        self.iteration = iteration
        self.k = k
        self.bal = bal
        self.single_model_per_cluster = single_model_per_cluster
        self.eval_strategy = eval_strategy

        # Ensembling
        self.ensembling_strategy = ensembling_strategy
        if ensemble_size is not None:
            self.ensemble_size = ensemble_size
        elif ensembling_strategy in (
            "majority_vote", "weighted_vote", "weighted_vote_proba"
        ):
            self.ensemble_size = 9
        else:
            self.ensemble_size = 1

        # Clusterer handle
        self.kmeans = None


    def _config_signature(self):
        sig = (
            self.training, self.proxy,
            tuple(self.ccr) if isinstance(self.ccr, (list, tuple)) else self.ccr,
            self.k, self.bal, self.cluster_strat,
            self.lam, self.eval_strategy, self.ensembling_strategy,
            self.ensemble_size, self.single_model_per_cluster, self.remove,
        )
        return hashlib.sha256(repr(sig).encode()).hexdigest()[:16]


    def _build_subdict_entry(self, i, comb_list_topm, all_comb_list_topm):
        if self.ensembling_strategy in (
            "majority_vote", "weighted_vote", "weighted_vote_proba"
        ):
            score_lookup = {tuple(c): s for c, s in all_comb_list_topm}
            ranked = []
            for comb in comb_list_topm:
                if i < len(comb):
                    ranked.append((comb[i], score_lookup.get(tuple(comb))))
            return ranked
        return comb_list_topm[0][i]

    def _predict_one(self, model_info, x_row):
        if not isinstance(model_info, list):
            return joblib.load(model_info).predict(x_row.reshape(1, -1))[0]
        if len(model_info) == 1:
            entry = model_info[0]
            path = entry[0] if isinstance(entry, tuple) else entry
            return joblib.load(path).predict(x_row.reshape(1, -1))[0]

        # Hard majority vote, multiplicity-preserved (no score weights)
        if self.ensembling_strategy == "majority_vote":
            preds = [
                int(joblib.load(e[0] if isinstance(e, tuple) else e)
                    .predict(x_row.reshape(1, -1))[0])
                for e in model_info
            ]
            return int(sum(preds) >= len(preds) / 2.0)

        # Combination-score-weighted vote (hard or soft)
        if self.ensembling_strategy in ("weighted_vote", "weighted_vote_proba"):
            use_proba = (self.ensembling_strategy == "weighted_vote_proba")
            weighted_sum = 0.0
            weight_total = 0.0
            for entry in model_info:
                if isinstance(entry, tuple):
                    path, comb_score = entry
                else:
                    path, comb_score = entry, None
                m = joblib.load(path)
                if use_proba and hasattr(m, "predict_proba"):
                    p = float(m.predict_proba(x_row.reshape(1, -1))[0][1])
                else:
                    p = float(m.predict(x_row.reshape(1, -1))[0])
                L = (comb_score / 100.0) if comb_score is not None else 0.5
                w = max(1.0 - L, 1e-3)
                weighted_sum += w * p
                weight_total += w
            return int((weighted_sum / weight_total) >= 0.5)

        # Fallback
        entry = model_info[0]
        path = entry[0] if isinstance(entry, tuple) else entry
        return joblib.load(path).predict(x_row.reshape(1, -1))[0]


    def _stage_train(self, X_train, y_train, X_val, y_val):
        classes = len(self.df[self.df_dict["label"]].unique())
        if classes > 2:
            model_training_list = [
                "LogisticRegression_mult", "SVM_mult", "RandomForest_mult",
                "kNN_mult", "NaiveBayes_mult", "NN_mult", "DecisionTree_mult",
            ]
        elif self.training == "EGRGSR":
            model_training_list = ["EGR", "GSR"]
        else:
            model_training_list = [self.training]

        index = self.df.index.name
        val_id_list = [i for i, _ in X_val.iterrows()]
        run_main = RunTraining(
            X_val, y_val, val_id_list, self.df_dict["sens_attrs"],
            index, self.df_dict["label"], self.df_dict["favored"],
            self.link, self.input_file, self.iteration,
            self.single_model_per_cluster, self.remove, self.df_dict,
        )
        val_df, d, model_list, model_comb = run_main.train(
            model_training_list, X_train, y_train, []
        )
        val_df.to_csv(self.link + "test_predictions.csv", index_label=index)
        val_df = val_df.sort_index()

        return val_df, model_list, model_comb, d


    def _stage_cluster(self, X_val, y_val):
        index = self.df.index.name

        X_val_new = copy.deepcopy(X_val)

        if self.proxy == "reweigh":
            with open(self.link + "reweighing_attributes.txt", "w") as outfile:
                df_new = copy.deepcopy(self.df)
                self.weight_dict = dict()
                cols = list(df_new.columns)
                cols.remove(self.df_dict["label"])
                for sens in self.df_dict["sens_attrs"]:
                    cols.remove(sens)
                for col in cols:
                    x_arr = df_new[col].to_numpy()
                    col_diff = 0
                    for sens in self.df_dict["sens_attrs"]:
                        z_arr = df_new[sens]
                        sens_corr = abs(pearsonr(x_arr, z_arr)[0])
                        if math.isnan(sens_corr):
                            sens_corr = 1
                        col_diff += (1 - sens_corr)
                    col_weight = col_diff / len(self.df_dict["sens_attrs"])
                    self.weight_dict[col] = col_weight
                    df_new[col] *= col_weight
                    X_val_new[col] *= col_weight
                    outfile.write(col + ": " + str(col_weight) + "\n")
            df_new.to_csv(
                "Datasets/reweigh/" + self.df_dict["filename"] + ".csv",
                index_label=index,
            )

        X_val_cluster = copy.deepcopy(X_val_new)
        for sens in self.df_dict["sens_attrs"]:
            X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != sens]

        # Cluster size selection
        if self.ccr[0] == self.ccr[1] and self.ccr[0] != -1:
            clustersize = self.ccr[0]
        else:
            sens_groups = len(X_val_new.groupby(self.df_dict["sens_attrs"]))
            min_cluster = (
                self.ccr[0] if self.ccr[0] != -1
                else min(len(X_val_cluster.columns),
                         int(len(X_val_cluster) / (50 * sens_groups)),
                         2)
            )
            max_cluster = (
                self.ccr[1] if self.ccr[1] != -1
                else min(int(len(X_val_cluster.columns) ** 2 / 2),
                         int(len(X_val_cluster) / (10 * sens_groups)))
            )
            if self.cluster_algorithm == "LOGmeans":
                clustersize = log_means(X_val_cluster, min_cluster, max_cluster)

        with open(self.link + "clustersize.txt", "w") as outfile:
            outfile.write(str(clustersize))

        X_for_clustering = X_val_cluster.values

        if self.proxy in ("no", "reweigh"):
            self.kmeans = KMeans(clustersize, random_state=0).fit(X_for_clustering)
            cluster_results = self.kmeans.predict(X_for_clustering)
            X_val_cluster["cluster"] = cluster_results

        elif self.proxy in ("vanilla", "mcf", "shfd", "shfd_drop"):
            sens_name = self.df_dict["sens_attrs"][0]
            a_val = _binary_group_codes(X_val[sens_name])
            self.kmeans = FairletConstruction(
                mode=self.proxy, random_state=42,
                n_clusters=clustersize, k=self.k, bal=self.bal,
                cluster_strat=self.cluster_strat,
            )
            cluster_results = self.kmeans.fit_predict(X_for_clustering, a_val, y_val)
            X_val_cluster["cluster"] = cluster_results
            with open(self.link + "clustersize_sparse_fallback.txt", "w") as outfile:
                outfile.write(str(self.kmeans.return_clustersize()))


        if self.proxy != "reweigh":
            self.weight_dict = None

        return X_val_cluster


    def _stage_select(self, X_val, val_df, X_val_cluster, model_comb):
        groups = len(self.df.groupby(self.df_dict["sens_attrs"]))
        model_comb_single = copy.deepcopy(model_comb)

        clustered_df = X_val_cluster.groupby("cluster")
        self.model_dict = dict()
        column_list = val_df.columns

        groups_df = val_df[self.df_dict["sens_attrs"]].drop_duplicates(
            self.df_dict["sens_attrs"]
        ).reset_index(drop=True)
        actual_num_of_groups = len(groups_df)
        sens_cols = groups_df.columns
        sensitive_groups = []
        for _, row in groups_df.iterrows():
            sg = [row[col] for col in sens_cols]
            sensitive_groups.append(sg[0] if len(sg) == 1 else tuple(sg))

        classes = len(self.df[self.df_dict["label"]].unique())
        if classes <= 2:
            metricer = Metrics2(self.df_dict["sens_attrs"], self.df_dict["label"])
        else:
            metricer = Metrics_Multi(self.df_dict["sens_attrs"], self.df_dict["label"])

        do_ensemble = self.ensembling_strategy in (
            "majority_vote", "weighted_vote", "weighted_vote_proba"
        )

        for key, _ in clustered_df:
            part_df = clustered_df.get_group(key)
            if len(part_df) == 1:
                continue
            part_df.index.name = "index"
            val_df.index.name = "index"
            part_df2 = val_df.merge(part_df, on="index", how="inner")[column_list]

            groups2 = part_df2[self.df_dict["sens_attrs"]].drop_duplicates(
                self.df_dict["sens_attrs"]
            ).reset_index(drop=True)
            num_of_groups = len(groups2)
            cluster_sensitive_groups = []
            for _, row in groups2.iterrows():
                sg = [row[col] for col in sens_cols]
                cluster_sensitive_groups.append(
                    sg[0] if len(sg) == 1 else tuple(sg)
                )

            # Pull in nearest neighbors from missing groups via kNN
            if num_of_groups != actual_num_of_groups:
                try:
                    cluster_center = self.kmeans.cluster_centers_[key]
                except Exception:
                    continue
                grouped_df_all = X_val.groupby(self.df_dict["sens_attrs"])
                for sens_grp in sensitive_groups:
                    if sens_grp in cluster_sensitive_groups:
                        continue
                    knn_df = grouped_df_all.get_group(sens_grp)
                    for sens_attr in self.df_dict["sens_attrs"]:
                        knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                    if self.proxy == "reweigh":
                        for col in list(knn_df.columns):
                            if col in self.weight_dict:
                                knn_df[col] *= self.weight_dict[col]
                            else:
                                knn_df = knn_df.loc[:, knn_df.columns != col]
                    nbrs = NearestNeighbors(
                        n_neighbors=10, algorithm="kd_tree"
                    ).fit(knn_df.values)
                    indices = nbrs.kneighbors(
                        cluster_center.reshape(1, -1), return_distance=False
                    )
                    real_indices = knn_df.index[indices.flatten()].tolist()
                    nearest_neighbors_df = val_df.loc[real_indices]
                    part_df2 = pd.concat(
                        [part_df2, nearest_neighbors_df], ignore_index=True
                    )

            groups_keys = self.df.groupby(self.df_dict["sens_attrs"]).groups.keys()

            # Single-mode selection (top-1)
            if not do_ensemble:
                if self.single_model_per_cluster == "mix":
                    comb_list_global, _ = metricer.fairness_metric(
                        part_df2, model_comb_single, groups_keys,
                        self.df_dict["favored"], self.metric, self.lam,
                        comb_amount=20, cnr=key,
                    )
                    new_models = [c[0] for c in comb_list_global]
                    model_comb_new = list(itertools.product(new_models, repeat=groups))
                    comb_list_global, _ = metricer.fairness_metric(
                        part_df2, model_comb_new, groups_keys,
                        self.df_dict["favored"], self.metric, self.lam,
                        comb_amount=20, cnr=key,
                    )
                else:
                    comb_list_global = self._run_eval_strategy(
                        metricer, part_df2, model_comb, groups_keys,
                        cnr=key, ensemble_size=1,
                    )[0]
                comb_list_topm = comb_list_global
                all_comb_list_topm = None

            # Ensemble-mode selection (top-m)
            else:
                comb_list_topm, all_comb_list_topm = self._run_eval_strategy(
                    metricer, part_df2, model_comb, groups_keys,
                    cnr=key, ensemble_size=self.ensemble_size,
                )

            # Build the cluster's sub-dictionary
            subdict = dict()
            for i, gt2 in enumerate(list(groups_keys)):
                if isinstance(gt2, (float, int)):
                    dict_key = [float(gt2)]
                else:
                    dict_key = [float(j) for j in tuple(gt2)]
                if do_ensemble:
                    subdict[str(dict_key)] = self._build_subdict_entry(
                        i, comb_list_topm, all_comb_list_topm
                    )
                else:
                    subdict[str(dict_key)] = comb_list_topm[0][i]
            self.model_dict[key] = subdict

        metricer.clusterdf.to_csv(
            f"{self.link}cluster_df_{self.iteration}.csv"
        )

    def _run_eval_strategy(self, metricer, part_df2, model_comb, groups_keys,
                            cnr, ensemble_size):
        favored = self.df_dict["favored"]

        raw_full = metricer.compute_raw_metrics(
            part_df2, model_comb, groups_keys, favored, self.metric, cnr=cnr,
        )

        if self.eval_strategy == "comb":
            best, all_pairs = metricer.rank_by_weight(
                raw_full, weight=self.lam, comb_amount=ensemble_size,
            )
            self._log_clusterdf(metricer, raw_full, self.lam, best, cnr)
            return best, all_pairs

        camount = max(1, int(np.ceil(len(model_comb) * 0.15)))

        params = {
            "acc_first":               (1.0, camount,         0,    0.0),
            "acc_first_thresh":        (1.0, 1,               5,    0.0),
            "bias_first":              (0.0, camount,         0,    1.0),
            "acc_first_bias_second":   (1.0, camount,         0,    0.0),
            "acc_first_hatl_second":   (1.0, camount,         0, self.lam),
        }
        if self.eval_strategy not in params:
            # Default: behave like "comb"
            best, all_pairs = metricer.rank_by_weight(
                raw_full, weight=self.lam, comb_amount=ensemble_size,
            )
            self._log_clusterdf(metricer, raw_full, self.lam, best, cnr)
            return best, all_pairs

        w1, n1, thr1, w2 = params[self.eval_strategy]

        shortlist, _ = metricer.rank_by_weight(
            raw_full, weight=w1, comb_amount=n1, threshold=thr1,
        )

        shortlist_set = {tuple(c) for c in shortlist}
        raw_short = raw_full[
            raw_full["comb"].apply(lambda c: tuple(c) in shortlist_set)
        ].reset_index(drop=True)

        best, all_pairs = metricer.rank_by_weight(
            raw_short, weight=w2, comb_amount=ensemble_size,
        )
        self._log_clusterdf(metricer, raw_short, w2, best, cnr)
        return best, all_pairs

    @staticmethod
    def _log_clusterdf(metricer, raw_df, weight, best_comb_list, cnr):
        scored = raw_df.copy()
        scored["score"] = (
            weight * scored["error"] + (1.0 - weight) * scored["bias"]
        )
        for _, row in scored.iterrows():
            metricer.clusterdf.loc[len(metricer.clusterdf)] = {
                "cnr": cnr,
                "error": row["error"],
                "bias": row["bias"],
                "score": row["score"],
                "comb": row["comb"],
                "opt": "",
            }
        if best_comb_list:
            best_match = scored.loc[
                scored["comb"].apply(lambda c: c == best_comb_list[0])
            ]
            best_score = (
                float(best_match["score"].iloc[0])
                if len(best_match) > 0 else 0.0
            )
            metricer.clusterdf.loc[len(metricer.clusterdf)] = {
                "cnr": cnr,
                "error": 0,
                "bias": 0,
                "score": best_score,
                "comb": best_comb_list[0],
                "opt": "best",
            }


    def fit(self, X_train, y_train):
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.validationsize, random_state=42)
        # Training stage
        val_df, model_list, model_comb, _d = self._stage_train(
            X_train, y_train, X_val, y_val,
        )
        self.model_list = model_list

        # Clustering stage
        X_val_cluster = self._stage_cluster(X_val, y_val)

        # Per-cluster selection
        self._stage_select(X_val, val_df, X_val_cluster, model_comb)
        return self

    def predict(self, X_test):
        cluster_model = "FairBALC"
        index = self.df.index.name

        sens_attrs = self.df_dict["sens_attrs"]

        X_test_cluster = X_test.drop(columns=sens_attrs)
        if self.proxy == "reweigh":
            cluster_cols = []
            for col in list(X_test_cluster.columns):
                if col in self.weight_dict:
                    X_test_cluster[col] = X_test_cluster[col] * self.weight_dict[col]
                    cluster_cols.append(col)
            X_test_cluster = X_test_cluster[cluster_cols]
        X_test_for_cluster = X_test_cluster.values

        Z_test = X_test.drop(columns=sens_attrs) if self.remove else X_test
        Z_test_values = Z_test.values

        n_test = len(X_test)

        # Batch cluster prediction
        cluster_assignments = self.kmeans.predict(X_test_for_cluster).astype(int)

        # Per-sample (cluster, group_key) lookup. The keys must use Python
        # floats (not numpy) to match how _stage_select wrote them at fit time
        sens_value_arr = X_test[sens_attrs].astype(float).values
        sens_keys = [
            str([float(v) for v in sens_value_arr[i]])
            for i in range(n_test)
        ]

        # Collect all unique model paths
        unique_paths = set()
        needs_proba = (self.ensembling_strategy == "weighted_vote_proba")

        for cluster_dict in self.model_dict.values():
            for entry in cluster_dict.values():
                if isinstance(entry, list):
                    for item in entry:
                        path = item[0] if isinstance(item, tuple) else item
                        unique_paths.add(path)
                else:
                    unique_paths.add(entry)

        # Batched prediction per unique model
        cache_pred = {}
        cache_proba = {}
        for path in unique_paths:
            m = joblib.load(path)
            preds = np.asarray(m.predict(Z_test_values)).astype(int)
            cache_pred[path] = np.where(preds <= 0, 0, 1)
            if needs_proba and hasattr(m, "predict_proba"):
                cache_proba[path] = np.asarray(
                    m.predict_proba(Z_test_values)[:, 1]
                ).astype(float)

        # Assemble the per-row predictions from the cached arrays
        predictions = np.zeros(n_test, dtype=int)
        model_used_log = [None] * n_test

        for i in range(n_test):
            cluster_label = int(cluster_assignments[i])
            sens_key = sens_keys[i]
            model_info = self.model_dict[cluster_label][sens_key]

            is_ensemble = isinstance(model_info, list) and len(model_info) > 1
            if not is_ensemble:
                if isinstance(model_info, list):
                    entry = model_info[0]
                    path = entry[0] if isinstance(entry, tuple) else entry
                else:
                    path = model_info
                predictions[i] = cache_pred[path][i]
                model_used_log[i] = path
                continue

            if self.ensembling_strategy == "majority_vote":
                vote_sum = 0
                vote_count = 0
                for entry in model_info:
                    path = entry[0] if isinstance(entry, tuple) else entry
                    vote_sum += int(cache_pred[path][i])
                    vote_count += 1
                predictions[i] = int(vote_sum >= vote_count / 2.0)

            elif self.ensembling_strategy in ("weighted_vote", "weighted_vote_proba"):
                use_proba = (self.ensembling_strategy == "weighted_vote_proba")
                weighted_sum = 0.0
                weight_total = 0.0
                for entry in model_info:
                    if isinstance(entry, tuple):
                        path, comb_score = entry
                    else:
                        path, comb_score = entry, None
                    if use_proba and path in cache_proba:
                        p = float(cache_proba[path][i])
                    else:
                        p = float(cache_pred[path][i])
                    L = (comb_score / 100.0) if comb_score is not None else 0.5
                    w = max(1.0 - L, 1e-3)
                    weighted_sum += w * p
                    weight_total += w
                predictions[i] = int((weighted_sum / weight_total) >= 0.5)

            else:
                entry = model_info[0]
                path = entry[0] if isinstance(entry, tuple) else entry
                predictions[i] = cache_pred[path][i]

            model_used_log[i] = self.ensembling_strategy

        test_df = pd.DataFrame({
            index: X_test.index.tolist(),
        })
        for attr in sens_attrs:
            test_df[attr] = X_test[attr].values
        test_df[cluster_model] = predictions
        test_df["model_used"] = model_used_log

        config_id = self._config_signature()
        out_path = (
            f"{self.link}predictions_{config_id}_iter{self.iteration}.csv"
        )
        test_df.to_csv(out_path, index=False)

        return list(test_df[cluster_model])