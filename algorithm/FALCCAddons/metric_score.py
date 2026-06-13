import copy
import math
import hashlib
import pandas as pd
import numpy as np
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.neighbors import NearestNeighbors


class Metrics():
    def __init__(self, sens_attrs, label):
        self.sens_attrs = sens_attrs
        self.label = label
        self.clusterdf = pd.DataFrame(columns=["cnr","error","bias","score","comb"])

    def fairness_metric(self, df, model_comb, groups, favored, metric, weight=0.5, threshold=0,
                        comb_amount=1, cnr=-1):
        best_comb_val = math.inf
        best_comb = model_comb[0]
        best_comb_list = []
        all_comb_list = []

        if isinstance(favored, (int, float, str)):
            privileged_classes = [[favored]]
        else:
            privileged_classes = [[fav] for fav in favored]

        dataset_true = StandardDataset(
            df,
            label_name=self.label,
            favorable_classes=[1],
            protected_attribute_names=self.sens_attrs,
            privileged_classes=privileged_classes
        )

        group_keys = list(df.groupby(self.sens_attrs).groups.keys())

        if len(self.sens_attrs) == 1:
            if isinstance(favored, (list, tuple)) and len(favored) == 1:
                favored_key = favored[0]
            else:
                favored_key = favored
        else:
            favored_key = tuple(favored)

        privileged_groups = [{
            self.sens_attrs[i]: (favored_key if len(self.sens_attrs) == 1 else favored_key[i])
            for i in range(len(self.sens_attrs))
        }]

        unprivileged_groups = []
        for g in group_keys:
            g_key = g if len(self.sens_attrs) > 1 else g
            if g_key == favored_key:
                continue
            if len(self.sens_attrs) == 1:
                unprivileged_groups.append({self.sens_attrs[0]: g_key})
            else:
                unprivileged_groups.append({
                    self.sens_attrs[i]: g_key[i] for i in range(len(self.sens_attrs))
                })

        for comb in model_comb:
            preds = []
            for i, row in df.iterrows():
                for j, grp in enumerate(groups):
                    same_group = True
                    if isinstance(grp, (int, float, str)):
                        if grp != row[self.sens_attrs[0]]:
                            same_group = False
                    else:
                        for k in range(len(grp)):
                            if grp[k] != row[self.sens_attrs[k]]:
                                same_group = False
                                break
                    if same_group:
                        preds.append(df.loc[i, str(comb[j])])
                        break  # matched this row's group; go to next row

            dataset_pred = copy.deepcopy(dataset_true)
            dataset_pred.labels = np.array(preds).reshape(-1, 1)

            metric_pred = BinaryLabelDatasetMetric(
                dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )
            class_metric_pred = ClassificationMetric(
                dataset_true, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups
            )

            if "mcc" in metric:
                tp = class_metric_pred.num_true_positives()
                tn = class_metric_pred.num_true_negatives()
                fp = class_metric_pred.num_false_positives()
                fn = class_metric_pred.num_false_negatives()
                numerator = tp * tn - fp * fn
                denominator = (
                    (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                ) ** 0.5

                if denominator == 0:
                    accuracy = 0  # common convention: define MCC = 0 when denominator is 0
                else:
                    accuracy = numerator / denominator * 100.0
            elif "bea" in metric:
                tp = class_metric_pred.num_true_positives()
                tn = class_metric_pred.num_true_negatives()
                fp = class_metric_pred.num_false_positives()
                fn = class_metric_pred.num_false_negatives()
                total = tp + tn + fp + fn
                balance = abs((tp+fp)/total - (tp+fn)/total)

                accuracy = (abs(class_metric_pred.accuracy()) - balance) * 100.0
            else:
                accuracy = abs(class_metric_pred.accuracy()) * 100.0

            if "demographic_parity" in metric:
                bias = abs(class_metric_pred.statistical_parity_difference()) * 100.0
            elif "equalized_odds" in metric:
                if not math.isnan(class_metric_pred.average_abs_odds_difference()):
                    bias = abs(class_metric_pred.average_abs_odds_difference()) * 100.0
                else:
                    bias = 0.0
            elif "equal_opportunity" in metric:
                if not math.isnan(class_metric_pred.equal_opportunity_difference()):
                    bias = abs(class_metric_pred.equal_opportunity_difference()) * 100.0
                else:
                    bias = 0.0
            elif "treatment_equality" in metric:
                # Handle possible division-by-zero safely
                def _safe_ratio(tp):
                    fp = class_metric_pred.num_false_positives(tp)
                    fn = class_metric_pred.num_false_negatives(tp)
                    denom = (fp + fn)
                    return (fp / denom) if denom != 0 else 0.0
                treq1 = _safe_ratio(True)
                treq2 = _safe_ratio(False)
                bias = abs(treq1 - treq2) * 100.0
            elif "consistency" in metric:
                dataset_df, df_dict = dataset_pred.convert_to_dataframe()
                # remove label column
                features_df = dataset_df.loc[:, dataset_df.columns != df_dict["label_names"][0]]
                models_consistency = 0.0
                # fit KNN on the whole (predicted-label) feature space
                nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(features_df.values)
                for ridx, row_outer in features_df.iterrows():
                    indices = nbrs.kneighbors(row_outer.values.reshape(1, -1), return_distance=False)
                    real_indices = features_df.index[indices].tolist()
                    df_local = dataset_df.loc[real_indices[0]]
                    knn_ppv = 0.0
                    knn_count = 0
                    for _, row_local in df_local.iterrows():
                        knn_ppv += row_local[df_dict["label_names"][0]]
                        knn_count += 1
                    knn_pppv = knn_ppv / knn_count if knn_count else 0.0
                    models_consistency += abs(dataset_df.loc[ridx][df_dict["label_names"][0]] - knn_pppv)
                bias = (models_consistency / len(dataset_df)) * 100.0
            else:
                raise ValueError(f"Unknown fairness metric: {metric}")
            
            bias = 0.0 if math.isnan(bias) else bias

            # Blend accuracy (higher better) with fairness gap (lower better)
            new_comb_val = weight * (100.0 - accuracy) + (1.0 - weight) * bias

            self.clusterdf.loc[len(self.clusterdf)] = {
                "cnr": cnr,
                "error": 100.0 - accuracy,
                "bias": bias,
                "score": new_comb_val,
                "comb": comb,
                "opt": ""
            }
            if new_comb_val < best_comb_val:
                best_comb_val = new_comb_val
                best_comb = comb
            if threshold and (new_comb_val <= threshold):
                best_comb_list.append(comb)
            all_comb_list.append((comb, new_comb_val))

        # Return top-k by blended score if no thresholding result
        comb_amount = min(comb_amount, len(all_comb_list))
        if comb_amount != 1 or threshold != 0:
            sorted_list = sorted(all_comb_list, key=lambda x: x[1])
            if comb_amount != 1:
                best_comb_list = [sorted_list[i][0] for i in range(comb_amount)]
            else:
                best_comb_list = []
                for comb, val in sorted_list:
                    if val <= best_comb_val + threshold:
                        best_comb_list.append(comb)
                    else:
                        break
        elif (comb_amount == 1 and threshold == 0) or (best_comb_list == []):
            best_comb_list = [best_comb]

        self.clusterdf.loc[len(self.clusterdf)] = {
                "cnr": cnr,
                "error": 0,
                "bias": 0,
                "score": best_comb_val,
                "comb": best_comb,
                "opt": "best"
            }
        
        return best_comb_list, all_comb_list



class Metrics2():
    def __init__(self, sens_attrs, label):
        self.sens_attrs = sens_attrs
        self.label = label
        self.clusterdf = pd.DataFrame(
            columns=["cnr", "error", "bias", "score", "comb", "opt"]
        )
        self._raw_cache = {}

    @staticmethod
    def _df_fingerprint(df, label, sens_attrs):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(df[label].values).tobytes())
        for s in sens_attrs:
            h.update(np.ascontiguousarray(df[s].values).tobytes())
        pred_cols = [c for c in df.columns if c != label and c not in sens_attrs]
        for c in sorted(pred_cols):
            h.update(c.encode())
            h.update(np.ascontiguousarray(df[c].values).tobytes())
        return h.hexdigest()[:16]

    @staticmethod
    def _model_comb_fingerprint(model_comb):
        h = hashlib.sha256()
        for comb in model_comb:
            h.update(repr(tuple(comb)).encode())
        return h.hexdigest()[:16]

    def _setup_groups(self, df, favored):
        if isinstance(favored, (int, float, str)):
            privileged_classes = [[favored]]
        else:
            privileged_classes = [[fav] for fav in favored]

        group_keys = list(df.groupby(self.sens_attrs).groups.keys())

        if len(self.sens_attrs) == 1:
            if isinstance(favored, (list, tuple)) and len(favored) == 1:
                favored_key = favored[0]
            else:
                favored_key = favored
        else:
            favored_key = tuple(favored)

        privileged_groups = [{
            self.sens_attrs[i]:
                (favored_key if len(self.sens_attrs) == 1 else favored_key[i])
            for i in range(len(self.sens_attrs))
        }]

        unprivileged_groups = []
        for g in group_keys:
            g_key = g if len(self.sens_attrs) > 1 else g
            if g_key == favored_key:
                continue
            if len(self.sens_attrs) == 1:
                unprivileged_groups.append({self.sens_attrs[0]: g_key})
            else:
                unprivileged_groups.append({
                    self.sens_attrs[i]: g_key[i]
                    for i in range(len(self.sens_attrs))
                })
        return (
            privileged_classes, privileged_groups, unprivileged_groups,
            favored_key,
        )

    def _compute_group_index(self, df, groups):
        n = len(df)
        group_idx = np.full(n, -1, dtype=np.int64)

        sens_arrs = [df[s].values for s in self.sens_attrs]

        for j, grp in enumerate(groups):
            if isinstance(grp, (int, float, str)):
                mask = (sens_arrs[0] == grp)
            else:
                mask = np.ones(n, dtype=bool)
                for k in range(len(grp)):
                    mask &= (sens_arrs[k] == grp[k])
            group_idx[mask] = j

        return group_idx

    def _gather_predictions(self, df, comb, group_idx):
        n = len(df)
        preds = np.zeros(n, dtype=np.int64)
        for j, model_name in enumerate(comb):
            mask = (group_idx == j)
            if not mask.any():
                continue
            col_values = df[str(model_name)].values
            preds[mask] = col_values[mask]
        return preds

    def compute_raw_metrics(self, df, model_comb, groups, favored, metric,
                              cnr=-1, use_cache=True):
        cache_key = None
        if use_cache:
            df_fp = self._df_fingerprint(df, self.label, self.sens_attrs)
            mc_fp = self._model_comb_fingerprint(model_comb)
            cache_key = (cnr, metric, df_fp, mc_fp)
            if cache_key in self._raw_cache:
                return self._raw_cache[cache_key].copy()

        (privileged_classes, privileged_groups, unprivileged_groups,
            _favored_key) = self._setup_groups(df, favored)

        dataset_true = StandardDataset(
            df,
            label_name=self.label,
            favorable_classes=[1],
            protected_attribute_names=self.sens_attrs,
            privileged_classes=privileged_classes,
        )

        dataset_pred = copy.deepcopy(dataset_true)
        n = dataset_pred.labels.shape[0]

        group_idx = self._compute_group_index(df, groups)

        records = []
        for comb in model_comb:
            preds = self._gather_predictions(df, comb, group_idx)
            dataset_pred.labels = preds.reshape(-1, 1)

            error, bias = self._error_and_bias(
                dataset_true, dataset_pred,
                privileged_groups, unprivileged_groups,
                metric,
            )
            records.append((comb, error, bias))

        raw_df = pd.DataFrame(records, columns=["comb", "error", "bias"])

        if use_cache and cache_key is not None:
            self._raw_cache[cache_key] = raw_df.copy()
        return raw_df

    def _error_and_bias(self, dataset_true, dataset_pred,
                          privileged_groups, unprivileged_groups, metric):
        class_metric_pred = ClassificationMetric(
            dataset_true, dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        if "mcc" in metric:
            tp = class_metric_pred.num_true_positives()
            tn = class_metric_pred.num_true_negatives()
            fp = class_metric_pred.num_false_positives()
            fn = class_metric_pred.num_false_negatives()
            num = tp * tn - fp * fn
            den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
            accuracy = (num / den * 100.0) if den != 0 else 0.0
        elif "bea" in metric:
            tp = class_metric_pred.num_true_positives()
            tn = class_metric_pred.num_true_negatives()
            fp = class_metric_pred.num_false_positives()
            fn = class_metric_pred.num_false_negatives()
            total = tp + tn + fp + fn
            balance = abs((tp + fp) / total - (tp + fn) / total)
            accuracy = (abs(class_metric_pred.accuracy()) - balance) * 100.0
        else:
            accuracy = abs(class_metric_pred.accuracy()) * 100.0
        error = 100.0 - accuracy

        if "demographic_parity" in metric:
            bias = abs(class_metric_pred.statistical_parity_difference()) * 100.0
        elif "equalized_odds" in metric:
            v = class_metric_pred.average_abs_odds_difference()
            bias = abs(v) * 100.0 if not math.isnan(v) else 0.0
        elif "equal_opportunity" in metric:
            v = class_metric_pred.equal_opportunity_difference()
            bias = abs(v) * 100.0 if not math.isnan(v) else 0.0
        elif "treatment_equality" in metric:
            def _safe_ratio(tp_flag):
                fp = class_metric_pred.num_false_positives(tp_flag)
                fn = class_metric_pred.num_false_negatives(tp_flag)
                denom = (fp + fn)
                return (fp / denom) if denom != 0 else 0.0
            bias = abs(_safe_ratio(True) - _safe_ratio(False)) * 100.0
        elif "consistency" in metric:
            bias = self._consistency_bias(dataset_pred)
        else:
            raise ValueError(f"Unknown fairness metric: {metric}")

        bias = 0.0 if math.isnan(bias) else bias
        return error, bias

    @staticmethod
    def _consistency_bias(dataset_pred):
        dataset_df, df_dict = dataset_pred.convert_to_dataframe()
        label_col = df_dict["label_names"][0]
        features_df = dataset_df.loc[:, dataset_df.columns != label_col]
        nbrs = NearestNeighbors(n_neighbors=10, algorithm="kd_tree").fit(
            features_df.values
        )
        models_consistency = 0.0
        for ridx, row_outer in features_df.iterrows():
            indices = nbrs.kneighbors(
                row_outer.values.reshape(1, -1), return_distance=False,
            )
            real_indices = features_df.index[indices].tolist()
            df_local = dataset_df.loc[real_indices[0]]
            knn_count = len(df_local)
            knn_pppv = (
                df_local[label_col].sum() / knn_count if knn_count else 0.0
            )
            models_consistency += abs(
                dataset_df.loc[ridx][label_col] - knn_pppv
            )
        return (models_consistency / len(dataset_df)) * 100.0

    def rank_by_weight(self, raw_df, weight=0.5, comb_amount=1, threshold=0):
        if len(raw_df) == 0:
            return [], []

        scored = raw_df.copy()
        scored["score"] = (
            weight * scored["error"] + (1.0 - weight) * scored["bias"]
        )

        all_comb_list = list(zip(scored["comb"].tolist(), scored["score"].tolist()))

        sorted_view = scored.sort_values("score", kind="mergesort")
        best_score = sorted_view.iloc[0]["score"]

        if threshold and threshold > 0:
            top = sorted_view[sorted_view["score"] <= best_score + threshold]
            best_comb_list = top["comb"].tolist()
            if not best_comb_list:
                best_comb_list = [sorted_view.iloc[0]["comb"]]
        else:
            n_keep = min(comb_amount, len(sorted_view))
            best_comb_list = sorted_view.head(n_keep)["comb"].tolist()

        return best_comb_list, all_comb_list

    def fairness_metric(self, df, model_comb, groups, favored, metric,
                          weight=0.5, threshold=0, comb_amount=1, cnr=-1):
        raw_df = self.compute_raw_metrics(
            df, model_comb, groups, favored, metric, cnr=cnr,
        )
        best_comb_list, all_comb_list = self.rank_by_weight(
            raw_df, weight=weight, comb_amount=comb_amount, threshold=threshold,
        )

        scored = raw_df.copy()
        scored["score"] = (
            weight * scored["error"] + (1.0 - weight) * scored["bias"]
        )
        for _, row in scored.iterrows():
            self.clusterdf.loc[len(self.clusterdf)] = {
                "cnr": cnr,
                "error": row["error"],
                "bias": row["bias"],
                "score": row["score"],
                "comb": row["comb"],
                "opt": "",
            }
        if len(best_comb_list) > 0:
            best_score = scored.loc[
                scored["comb"].apply(lambda c: c == best_comb_list[0])
            ]["score"]
            best_score_val = (
                float(best_score.iloc[0]) if len(best_score) > 0 else 0.0
            )
            self.clusterdf.loc[len(self.clusterdf)] = {
                "cnr": cnr,
                "error": 0,
                "bias": 0,
                "score": best_score_val,
                "comb": best_comb_list[0],
                "opt": "best",
            }
        return best_comb_list, all_comb_list

    def clear_raw_cache(self):
        self._raw_cache = {}