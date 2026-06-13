"""
This code evaluates the results of the experiments based on several metrics.
"""
import warnings
import argparse
import ast
import copy
import shelve
import pandas as pd
import time
import math
import joblib
import re
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

warnings.simplefilter(action='ignore', category=FutureWarning)


def compute_local_fairness_knn(X_pred, y_pred, y_true, models, sens_attrs, k=5):
    """
    Evaluate local fairness metrics via kNN neighborhoods across all points in X_test.
    This generalizes to multiple protected attributes and groups.
    """
    protected_attrs = X_pred[sens_attrs].apply(tuple, axis=1)
    unique_groups = protected_attrs.unique()

    # Determine how many neighbors per group
    neighbors_per_group = max(1, k // len(unique_groups))

    X_features = X_pred.drop(columns=sens_attrs).to_numpy()

    group_indices = {
        group: protected_attrs[protected_attrs == group].index.to_numpy() for group in unique_groups
    }
    # Convert index to positions (relative to X_test):
    position_map = {idx: pos for pos, idx in enumerate(X_pred.index)}
    group_indices = {
        group: np.array([position_map[idx] for idx in indices]) for group, indices in group_indices.items()
    }

    local_metrics = dict()
    for model in models:
        local_metrics[model] = {"dp": [], "eod": [], "eop": [], "di": [], "te": [], "geo": []}
        
    for idx, (row, row_group) in enumerate(zip(X_features, protected_attrs)):
        neighbors_indices = []
        for group in unique_groups:
            group_features = X_features[group_indices[group]]
            nn = NearestNeighbors(n_neighbors=k).fit(group_features)
            distances, indices = nn.kneighbors([row])
            neighbors_in_group = group_indices[group][indices.flatten()]
            neighbors_indices.extend(neighbors_in_group)

        for model in models:
            preds = y_pred[model].iloc[neighbors_indices]
            true = y_true.iloc[neighbors_indices]
            neighbor_groups = protected_attrs.iloc[neighbors_indices]

            group_preds = {}
            group_true = {}
            for group in unique_groups:
                mask = neighbor_groups == group
                group_preds[group] = preds[mask.values]
                group_true[group] = true[mask.values]

            # Demographic Parity (DP)
            rates = {g: (p.mean() if len(p) > 0 else 0) for g, p in group_preds.items()}
            dp = abs(max(rates.values()) - min(rates.values()))
            local_metrics[model]["dp"].append(dp)

            # Equalized Odds (EOD) & Equal Opportunity (EOP)
            tprs = {}
            fprs = {}
            for g in unique_groups:
                y_g_true = group_true[g]
                y_g_pred = group_preds[g]
                if len(y_g_true) == 0:
                    tprs[g], fprs[g] = 0, 0
                    continue
                tp = np.sum((y_g_true == 1) & (y_g_pred == 1))
                fn = np.sum((y_g_true == 1) & (y_g_pred == 0))
                fp = np.sum((y_g_true == 0) & (y_g_pred == 1))
                tn = np.sum((y_g_true == 0) & (y_g_pred == 0))
                tprs[g] = tp / (tp + fn) if (tp + fn) > 0 else 0
                fprs[g] = fp / (fp + tn) if (fp + tn) > 0 else 0

            eod = abs(max(tprs.values()) - min(tprs.values())) + abs(max(fprs.values()) - min(fprs.values()))
            eop = abs(max(tprs.values()) - min(tprs.values()))
            local_metrics[model]["eod"].append(eod)
            local_metrics[model]["eop"].append(eop)

            # Disparate Impact (DI)
            positive_rates = [r for r in rates.values() if r > 0]
            di = min(positive_rates) / max(positive_rates) if len(positive_rates) > 1 else 1
            local_metrics[model]["di"].append(abs(1 - di))  # distance to 1 for consistency with others (smaller is fairer)

            # Treatment Equality (TE)
            fps = [np.sum((group_true[g] == 0) & (group_preds[g] == 1)) for g in unique_groups]
            fns = [np.sum((group_true[g] == 1) & (group_preds[g] == 0)) for g in unique_groups]
            te_values = [fp / fn if fn > 0 else 0 for fp, fn in zip(fps, fns) if fn > 0]
            te = max(te_values) - min(te_values) if len(te_values) > 1 else 0
            local_metrics[model]["te"].append(te)

            # Generalized Entropy (GEO)
            preds_all = np.concatenate(list(group_preds.values()))
            if len(preds_all) > 0:
                mu = preds_all.mean()
                geo = np.mean(((preds_all / mu) ** 2 - 1) / 2) if mu != 0 else 0
            else:
                geo = 0
            local_metrics[model]["geo"].append(geo)

    for model in models:
        for metric in local_metrics[model]:
            local_metrics[model][metric] = np.mean(local_metrics[model][metric])
    #avg_local_metrics = {key: np.mean(values) for key, values in local_metrics.items()}

    return local_metrics


def local_lipschitzness(model, X, sens_attrs, epsilon=1e-4):
    """
    Measures local Lipschitzness by perturbing non-protected features and checking the change in predictions.
    
    :param model: Trained model.
    :param X: Input feature matrix.
    :param protected_index: Index of the protected attribute in X.
    :param epsilon: Small perturbation applied to non-protected features.
    :return: Average local Lipschitzness score.
    """
    #X_non_protected = np.delete(X, protected_index, axis=1)  # Remove the protected attribute
    #n_samples, n_features = X_non_protected.shape
    lipschitz_values = []


    X_pert = copy.deepcopy(X)
    for col in list(X.columns):
        if col not in sens_attrs:
            X_pert[col] = X_pert[col] + random.uniform(-epsilon, epsilon)

    # Get the original prediction (ensure it's a scalar)
    original_pred = model.predict(X)  # Probability of class 1
    
    # Perturb the input slightly by adding epsilon to all non-protected features
    perturbed_pred = model.predict(X_pert)  # Probability of class 1

    X_nosens = copy.deepcopy(X)
    X_pert_nosens = copy.deepcopy(X_pert)
    for sens in sens_attrs:
        X_nosens = X_nosens.loc[:, X_nosens.columns != sens]
        X_pert_nosens = X_pert_nosens.loc[:, X_pert_nosens.columns != sens]
    
    # For each individual
    for i, pred in enumerate(original_pred):
        """
        try:
            # Get the original prediction (ensure it's a scalar)
            original_input = X_non_protected[i, :].reshape(1, -1)
            original_pred = model.predict(original_input)[0]  # Probability of class 1
            
            # Perturb the input slightly by adding epsilon to all non-protected features
            perturbed_input = original_input + epsilon
            perturbed_pred = model.predict(perturbed_input)[0]  # Probability of class 1
        except:
            # Get the original prediction (ensure it's a scalar)
            original_input = X[i, :].reshape(1, -1)
            original_pred = model.predict(original_input)[0]  # Probability of class 1
            
            # Perturb the input slightly by adding epsilon to all non-protected features
            perturbed_input = original_input + epsilon
            perturbed_pred = model.predict(perturbed_input)[0]  # Probability of class 1
        """
         
        # Compute the change in predictions and input
        delta_pred = np.abs(perturbed_pred[i] - pred)
        delta_input = np.linalg.norm(X_pert_nosens.to_numpy() - X_nosens.to_numpy())  # Euclidean distance
        
        # Calculate the Lipschitz constant for this sample
        if delta_input != 0:
            lipschitz_value = delta_pred / delta_input
            lipschitz_values.append(lipschitz_value)
    
    # Return the average Lipschitz constant over all samples
    return np.mean(lipschitz_values) if lipschitz_values else 0


def individual_consistency(X, y_pred, protected_index, k=5):
    """
    Measures individual consistency by checking if similar individuals (non-protected features) receive similar predictions.
    
    :param X: Input feature matrix.
    :param y_pred: Predicted labels.
    :param protected_index: Index of the protected attribute in X.
    :param k: Number of nearest neighbors to consider.
    :return: Consistency score (average similarity in predictions).
    """
    # Remove the protected attribute from feature set
    X_non_protected = np.delete(X, protected_index, axis=1)

    # Find k nearest neighbors based on non-protected attributes
    knn = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
    knn.fit(X_non_protected)
    neighbors = knn.kneighbors(X_non_protected, return_distance=False)

    consistency_scores = []
    for i, neighbor_indices in enumerate(neighbors):
        # Check if the predictions for the individual and their neighbors are similar
        consistency = 0
        for n in neighbor_indices:
            consistency += abs(y_pred[n] - y_pred[i])
        consistency_scores.append(round(consistency/k, 3))
    
    # The higher the score, the more fair the model is in terms of individual consistency
    return np.mean(consistency_scores)


def full_evaluation(link, input_file, index, sens_attrs, label, favored, model_list, metric, proxy, lam, lrd):
    #Read the original dataset
    if proxy == "no":
        original_data = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
    elif proxy == "reweigh":
        original_data = pd.read_csv("Datasets/reweigh/" + input_file + ".csv",  index_col=index)
    elif proxy == "remove":
        original_data = pd.read_csv("Datasets/removed/" + input_file + ".csv",  index_col=index)
    dataset = copy.deepcopy(original_data)
    original_data = original_data.drop(columns=[label])
    col_order = list(original_data.columns)
    for sens in sens_attrs:
        original_data = original_data.loc[:, original_data.columns != sens]

    for i, model in enumerate(model_list):
        try:
            original_data_short = pd.read_csv(link + model + "_prediction.csv", index_col=index)
            modelnr = i
            break
        except:
            pass

    original_data_short = pd.merge(original_data_short, original_data, left_index=True, right_index=True)
    original_data_short = original_data_short.loc[:, original_data_short.columns != model_list[modelnr]]
    orig_datas = copy.deepcopy(original_data_short)
    original_data_short = original_data_short.loc[:, original_data_short.columns != label]
    valid_data = dataset.loc[orig_datas.index, :]

    for sens in sens_attrs:
        original_data_short = original_data_short.loc[:, original_data_short.columns != sens]
    dataset2 = copy.deepcopy(original_data_short)
    total_size = len(original_data_short)
    print("Total size: ", total_size)

    groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
    actual_num_of_groups = len(groups)
    sensitive_groups = []
    sens_cols = groups.columns
    for i, row in groups.iterrows():
        sens_grp = []
        for col in sens_cols:
            sens_grp.append(row[col])
        sensitive_groups.append(tuple(sens_grp))

    if lrd:
        filename = link + "cluster_0.out"
        my_shelf = shelve.open(filename)
        kmeans = my_shelf["kmeans"]
        my_shelf.close()

        """
        weight_dict = {}

        with open(link + "reweighing_attributes.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or ":" not in line:
                    continue  # skip empty or malformed lines
                key, value = line.split(":", 1)
                weight_dict[key.strip()] = float(value.strip())

        X_val_cluster = copy.deepcopy(dataset2)
        for col in list(X_val_cluster.columns):
            if col in weight_dict:
                X_val_cluster[col] *= weight_dict[col]
            else:
                X_val_cluster = X_val_cluster.loc[:, X_val_cluster.columns != col]
        """

        X_val_cluster = copy.deepcopy(dataset2)
        cluster_results = kmeans.predict(X_val_cluster)
        X_val_cluster["cluster"] = cluster_results
        clustered_df = X_val_cluster.groupby("cluster")

        cluster_count = 0
        total_size = 0
        cluster_ids = []
        for key, item in clustered_df:
            cluster_count += 1
            part_df = clustered_df.get_group(key)
            index_list = []
            for i, row in part_df.iterrows():
                index_list.append(i)
            df_local = orig_datas.loc[index_list]
            groups2 = df_local[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
            num_of_groups = len(groups2)
            cluster_sensitive_groups = []
            for i, row in groups2.iterrows():
                sens_grp = []
                for col in sens_cols:
                    sens_grp.append(row[col])
                cluster_sensitive_groups.append(tuple(sens_grp))

            #If a cluster does not contain samples of all groups, it will take the k nearest neighbors
            #(default value = 15) to test the model combinations
            if num_of_groups != actual_num_of_groups:
                cluster_center = kmeans.cluster_centers_[key]
                for sens_grp in sensitive_groups:
                    if sens_grp not in cluster_sensitive_groups:
                        if len(sens_attrs) == 1:
                            sens_grp = sens_grp[0]
                        grouped_df = valid_data.groupby(sens_attrs)
                        for key_inner, item_inner in grouped_df:
                            if len(sens_attrs) == 1:
                                key_inner = key_inner[0]
                            if key_inner == sens_grp:
                                knn_df = grouped_df.get_group(key_inner)
                                for sens_attr in sens_attrs:
                                    knn_df = knn_df.loc[:, knn_df.columns != sens_attr]
                                knn_df = knn_df.loc[:, knn_df.columns != "index"]
                                knn_df = knn_df.loc[:, knn_df.columns != label]
                                nbrs = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(knn_df.values)
                                indices = nbrs.kneighbors(cluster_center.reshape(1, -1), return_distance=False)
                                #real_indices = valid_data.index[indices].tolist()
                                real_indices = valid_data.index.take(indices.ravel()).tolist()
                                for ind in real_indices:
                                    index_list.append(ind)
            cluster_ids.append(index_list)

    df_count = 0
    result_df = pd.DataFrame()
    cluster_res = pd.DataFrame()
    ccount = 0

    for model in model_list:
        result_df.at[df_count, "model"] = model
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col=index)
            if np.isnan(df.values).any():
                continue
        except:
            continue

        df = pd.merge(df, original_data_short, left_index=True, right_index=True)
        y_true = df[label]

        if actual_num_of_groups == 2:
            if isinstance(favored, int) or isinstance(favored, float):
                ds = StandardDataset(df, 
                    label_name=label, 
                    favorable_classes=[1], 
                    protected_attribute_names=sens_attrs, 
                    privileged_classes=[[favored]])
            else:
                ds = StandardDataset(df, 
                    label_name=label, 
                    favorable_classes=[1], 
                    protected_attribute_names=sens_attrs, 
                    privileged_classes=[favored])

            dataset_pred = ds.copy()
            
            dataset_pred.labels = df[model]

            attr = dataset_pred.protected_attribute_names[0]
            idx = dataset_pred.protected_attribute_names.index(attr)
            privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
            unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]
            """
            priv_dict = dict()
            unpriv_dict = dict()
            if isinstance(favored, tuple):
                for i, fav_val in enumerate(favored):
                    priv_dict[sens_attrs[i]] = fav_val
                    all_val = list(df.groupby(sens_attrs[i]).groups.keys())
                    for poss_val in all_val:
                        if poss_val != fav_val:
                            unpriv_dict[sens_attrs[i]] = poss_val
            else:
                if favored == 0:
                    priv_dict[sens_attrs[0]] = 0
                    unpriv_dict[sens_attrs[0]] = 1
                elif favored == 1:
                    priv_dict[sens_attrs[0]] = 1
                    unpriv_dict[sens_attrs[0]] = 0

            privileged_groups = [priv_dict]
            unprivileged_groups = [unpriv_dict]

            print(df[sens_attrs])
            """

            metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            class_metric_pred = ClassificationMetric(ds, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

            result_df.at[df_count, "error_rate"] = class_metric_pred.error_rate() * 100
            result_df.at[df_count, "demographic_parity"] = abs(metric_pred.statistical_parity_difference()) * 100
            result_df.at[df_count, "equalized_odds"] = abs(class_metric_pred.average_abs_odds_difference()) * 100
            result_df.at[df_count, "equal_opportunity"] = abs(class_metric_pred.equal_opportunity_difference()) * 100
            treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
            treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
            result_df.at[df_count, "treatment_equality"] = abs(treq1 - treq2) * 100
            fp = class_metric_pred.num_false_positives()
            fn = class_metric_pred.num_false_negatives()
            tp = class_metric_pred.num_true_positives()
            tn = class_metric_pred.num_true_negatives()
            total_n = fp + fn + tp + tn
            result_df.at[df_count, "bea"] = ((tp+tn)/total_n - abs((tp+fp)/total_n - (tp+fn)/total_n)) * 100
            result_df.at[df_count, "tp"] = tp
            result_df.at[df_count, "tn"] = tn
            result_df.at[df_count, "fp"] = fp
            result_df.at[df_count, "fn"] = fn

            result_df.at[df_count, "generalized_entropy_index"] = abs(class_metric_pred.generalized_entropy_index()) * 100
            result_df.at[df_count, "smoothed_edf"] = abs(metric_pred.smoothed_empirical_differential_fairness()) * 100
            
            result_df.at[df_count, "num_positives"] = abs(metric_pred.num_positives())
            result_df.at[df_count, "num_negatives"] = abs(metric_pred.num_negatives())
            
            result_df.at[df_count, "false_discovery_rate"] = abs(class_metric_pred.false_discovery_rate()) * 100
            result_df.at[df_count, "false_omission_rate"] = abs(class_metric_pred.false_omission_rate()) * 100
            result_df.at[df_count, "false_discovery_rate_difference"] = abs(class_metric_pred.false_discovery_rate_difference()) * 100
            result_df.at[df_count, "false_omission_rate_difference"] = abs(class_metric_pred.false_omission_rate_difference()) * 100

            #result_df.at[df_count, "average_predictive_value_difference"] = abs(class_metric_pred.average_predictive_value_difference()) * 100
            result_df.at[df_count, "between_all_groups_coefficient_of_variation"] = abs(class_metric_pred.between_all_groups_coefficient_of_variation()) * 100
            result_df.at[df_count, "between_all_groups_generalized_entropy_index"] = abs(class_metric_pred.between_all_groups_generalized_entropy_index()) * 100
            result_df.at[df_count, "between_all_groups_theil_index"] = abs(class_metric_pred.between_all_groups_theil_index()) * 100
            result_df.at[df_count, "between_group_coefficient_of_variation"] = abs(class_metric_pred.between_group_coefficient_of_variation()) * 100
            result_df.at[df_count, "between_group_generalized_entropy_index"] = abs(class_metric_pred.between_group_generalized_entropy_index()) * 100
            result_df.at[df_count, "differential_fairness_bias_amplification"] = abs(class_metric_pred.differential_fairness_bias_amplification()) * 100
            result_df.at[df_count, "false_positive_rate_difference"] = class_metric_pred.false_positive_rate_difference() * 100
            result_df.at[df_count, "false_negative_rate_difference"] = class_metric_pred.false_negative_rate_difference() * 100
            result_df.at[df_count, "FPFN_balance_loss"] = abs(0.5 - fp/(fp+fn)) * 100

            if lrd:
                lrd_dp = 0
                lrd_eod = 0
                lrd_eop = 0
                lrd_te = 0
                lrd_di = 0
                cc = 0
                cc2 = 0
                for i, clusters in enumerate(cluster_ids):
                    cluster_df = df.loc[clusters]
                    cluster_size = len(cluster_df)
                    cc += cluster_size
                    cc2 += cluster_size

                    if isinstance(favored, int) or isinstance(favored, float):
                        ds = StandardDataset(cluster_df, 
                            label_name=label, 
                            favorable_classes=[1], 
                            protected_attribute_names=sens_attrs, 
                            privileged_classes=[[favored]])
                    else:
                        ds = StandardDataset(cluster_df, 
                            label_name=label, 
                            favorable_classes=[1], 
                            protected_attribute_names=sens_attrs, 
                            privileged_classes=[favored])

                    dataset_pred = ds.copy()
                    dataset_pred.labels = cluster_df[model]
                    #dataset_pred.label = df[model]

                    #attr = dataset_pred.protected_attribute_names[0]
                    #idx = dataset_pred.protected_attribute_names.index(attr)
                    #privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
                    #unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]
                    priv_dict = dict()
                    unpriv_dict = dict()
                    if isinstance(favored, tuple):
                        for i, fav_val in enumerate(favored):
                            priv_dict[sens_attrs[i]] = fav_val
                            all_val = list(df.groupby(sens_attrs[i]).groups.keys())
                            for poss_val in all_val:
                                if poss_val != fav_val:
                                    unpriv_dict[sens_attrs[i]] = poss_val
                    else:
                        if favored == 0:
                            priv_dict[sens_attrs[0]] = 0
                            unpriv_dict[sens_attrs[0]] = 1
                        elif favored == 1:
                            priv_dict[sens_attrs[0]] = 1
                            unpriv_dict[sens_attrs[0]] = 0

                    privileged_groups = [priv_dict]
                    unprivileged_groups = [unpriv_dict]

                    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
                    class_metric_pred = ClassificationMetric(ds, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

                    lrd_dp += abs(metric_pred.statistical_parity_difference()) * 100 * cluster_size
                    if not math.isnan(class_metric_pred.average_abs_odds_difference()):
                        lrd_eod += abs(class_metric_pred.average_abs_odds_difference()) * 100 * cluster_size
                        #lrd_eop += abs(class_metric_pred.equal_opportunity_difference()) * 100 * cluster_size
                        treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
                        treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
                        if np.isnan(treq1) and np.isnan(treq2):
                            lrd_te += 0
                        elif np.isnan(treq1):
                            lrd_te += abs(0.5 - treq2) * 100 * cluster_size
                        elif np.isnan(treq2):
                            lrd_te += abs(treq1 - 0.5) * 100 * cluster_size
                        else:
                            lrd_te += abs(treq1 - treq2) * 100 * cluster_size
                    else:
                        cc2 -= cluster_size
                    disparate_impact = abs(metric_pred.disparate_impact())
                    if disparate_impact > 1:
                        disparate_impact = disparate_impact ** -1
                    lrd_di += disparate_impact * 100 * cluster_size
                    if not math.isnan(class_metric_pred.equal_opportunity_difference()):
                        lrd_eop += abs(class_metric_pred.equal_opportunity_difference()) * 100 * cluster_size


                    cluster_res.at[ccount, "c"] = i
                    cluster_res.at[ccount, "model"] = model
                
                    bias = 0
                    if metric == "demographic_parity":
                        bias = abs(metric_pred.statistical_parity_difference()) * 100
                    elif metric == "equalized_odds":
                        if not math.isnan(class_metric_pred.average_abs_odds_difference()):
                            bias = abs(class_metric_pred.average_abs_odds_difference()) * 100
                    elif metric == "equal_opportunity":
                        if not math.isnan(class_metric_pred.equal_opportunity_difference()):
                            bias = abs(class_metric_pred.equal_opportunity_difference()) * 100
                    elif metric == "treatment_equality":
                        if np.isnan(treq1) and np.isnan(treq2):
                            bias = 0
                        elif np.isnan(treq1):
                            bias = abs(0.5 - treq2) * 100
                        elif np.isnan(treq2):
                            bias = abs(treq1 - 0.5) * 100
                        else:
                            bias = abs(treq1 - treq2) * 100

                    try:
                        error = class_metric_pred.error_rate() * 100
                        cluster_res.at[ccount, "error_rate"] = round(error,4)
                        cluster_res.at[ccount, "bias"] = round(bias,4)
                        cluster_res.at[ccount, "score"] = round(lam*error + (1-lam)*bias,4)
                        cdf = cluster_df.groupby(sens_attrs[0])
                        try:
                            cluster_res.at[ccount, "g0"] = int(len(cdf.get_group(0)))
                        except:
                            cluster_res.at[ccount, "g0"] = 0
                        try:
                            cluster_res.at[ccount, "g1"] = int(len(cdf.get_group(1)))
                        except:
                            cluster_res.at[ccount, "g1"] = 0
                        ccount += 1
                    except:
                        pass

                result_df.at[df_count, "cluster_dp"] = lrd_dp / cc
                result_df.at[df_count, "cluster_eod"] = lrd_eod / cc
                result_df.at[df_count, "cluster_eop"] = lrd_eop / cc
                result_df.at[df_count, "cluster_te"] = lrd_te / cc
                result_df.at[df_count, "cluster_di"] = lrd_di / cc

                # --- Extended local fairness (kernel/diffusion/density/soft) ---
                try:
                    from evaluation_extended import (
                        compute_local_fairness_kernel,
                        compute_local_fairness_diffusion,
                        compute_local_fairness_density_adaptive,
                        compute_local_fairness_soft_clustering
                    )
                    # Build inputs: features X from original_data_short, protected S, predictions and labels
                    X_ext = original_data_short.loc[df.index, :].values
                    y_pred_ext = df[model].values
                    y_true_ext = df[label].values
                    if len(sens_attrs) > 1:
                        S_ext = pd.Categorical(df[sens_attrs].apply(tuple, axis=1)).codes
                    else:
                        S_ext = df[sens_attrs[0]].values

                    '''
                    # Kernel
                    rk_dp   = compute_local_fairness_kernel(X_ext, y_pred_ext, S_ext, metric="dp",   y_true=y_true_ext)
                    rk_eod  = compute_local_fairness_kernel(X_ext, y_pred_ext, S_ext, metric="eod",  y_true=y_true_ext)
                    #rk_eop  = compute_local_fairness_kernel(X_ext, y_pred_ext, S_ext, metric="eop",  y_true=y_true_ext)
                    rk_treq = compute_local_fairness_kernel(X_ext, y_pred_ext, S_ext, metric="treq", y_true=y_true_ext)
                    result_df.at[df_count, "kernel_dp"]   = rk_dp["mean"]
                    result_df.at[df_count, "kernel_eod"]  = rk_eod["mean"]
                    #result_df.at[df_count, "kernel_eop"]  = rk_eop["mean"]
                    result_df.at[df_count, "kernel_treq"] = rk_treq["mean"]
                    
                    # Diffusion
                    rd_dp   = compute_local_fairness_diffusion(X_ext, y_pred_ext, S_ext, metric="dp",   y_true=y_true_ext)
                    rd_eod  = compute_local_fairness_diffusion(X_ext, y_pred_ext, S_ext, metric="eod",  y_true=y_true_ext)
                    rd_eop  = compute_local_fairness_diffusion(X_ext, y_pred_ext, S_ext, metric="eop",  y_true=y_true_ext)
                    rd_treq = compute_local_fairness_diffusion(X_ext, y_pred_ext, S_ext, metric="treq", y_true=y_true_ext)
                    result_df.at[df_count, "diffusion_dp"]   = rd_dp["mean"]
                    result_df.at[df_count, "diffusion_eod"]  = rd_eod["mean"]
                    result_df.at[df_count, "diffusion_eop"]  = rd_eop["mean"]
                    result_df.at[df_count, "diffusion_treq"] = rd_treq["mean"]
                    '''

                    # Density-adaptive
                    #rda_dp   = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="dp",   y_true=y_true_ext)
                    #rda_eod  = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="eod",  y_true=y_true_ext)
                    ##rda_eop  = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="eop",  y_true=y_true_ext)
                    #rda_treq = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="treq", y_true=y_true_ext)
                    #result_df.at[df_count, "density_dp"]   = rda_dp["mean"]
                    #result_df.at[df_count, "density_eod"]  = rda_eod["mean"]
                    ##result_df.at[df_count, "density_eop"]  = rda_eop["mean"]
                    #result_df.at[df_count, "density_te"] = rda_treq["mean"]

                    '''
                    # Soft clustering
                    rs_dp   = compute_local_fairness_soft_clustering(X_ext, y_pred_ext, S_ext, metric="dp",   y_true=y_true_ext)
                    rs_eod  = compute_local_fairness_soft_clustering(X_ext, y_pred_ext, S_ext, metric="eod",  y_true=y_true_ext)
                    rs_eop  = compute_local_fairness_soft_clustering(X_ext, y_pred_ext, S_ext, metric="eop",  y_true=y_true_ext)
                    rs_treq = compute_local_fairness_soft_clustering(X_ext, y_pred_ext, S_ext, metric="treq", y_true=y_true_ext)
                    result_df.at[df_count, "soft_dp"]   = rs_dp["mean"]
                    result_df.at[df_count, "soft_eod"]  = rs_eod["mean"]
                    result_df.at[df_count, "soft_eop"]  = rs_eop["mean"]
                    result_df.at[df_count, "soft_treq"] = rs_treq["mean"]
                    '''
                    # Density-adaptive
                    rda_dp   = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="dp",   y_true=y_true_ext)
                    rda_eod  = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="eod",  y_true=y_true_ext)
                    ##rda_eop  = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="eop",  y_true=y_true_ext)
                    rda_treq = compute_local_fairness_density_adaptive(X_ext, y_pred_ext, S_ext, metric="treq", y_true=y_true_ext)
                    result_df.at[df_count, "density_dp"]   = rda_dp["mean"]
                    result_df.at[df_count, "density_eod"]  = rda_eod["mean"]
                    ##result_df.at[df_count, "density_eop"]  = rda_eop["mean"]
                    result_df.at[df_count, "density_te"] = rda_treq["mean"]
                except Exception as e:
                    warnings.warn(f"Extended local fairness computation failed: {e}")


        else:
            """
            TODO LRD einbauen von FALCC Paper.
            """
            result_df.at[df_count, "model"] = model
            if sens_attrs[0] not in df.columns:
                for sens in sens_attrs:
                    df[sens] = orig_datas[sens]
            grouped_df = df.groupby(sens_attrs)
            total_ppv = 0
            total_size = 0
            total_ppv_y0 = 0
            total_size_y0 = 0
            total_ppv_y1 = 0
            total_size_y1 = 0
            discr_ppv_y0 = 0
            discr_size_y0 = 0
            discr_ppv_y1 = 0
            discr_size_y1 = 0
            wrong_predicted = 0
            wrong_predicted_y0 = 0
            wrong_predicted_y1 = 0
            total_fp = 0
            total_fn = 0
            num_pos = 0
            num_neg = 0
            #counterfactual = 0
            group_predsize = []
            #Get the favored group to test against and also the averages over the whole dataset
            di = dict()
            for key, item in grouped_df:
                predsize = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    predsize += 1
                    total_ppv = total_ppv + row[model]
                    total_size = total_size + 1
                    wrong_predicted = wrong_predicted + abs(row[model] - row[label])
                    if row[label] == 0:
                        total_ppv_y0 = total_ppv_y0 + row[model]
                        total_size_y0 = total_size_y0 + 1
                        wrong_predicted_y0 = wrong_predicted_y0 + abs(row[model] - row[label])
                        if row[model] == 1:
                            total_fp += 1
                            num_pos += 1
                        else:
                            num_neg += 1
                    elif row[label] == 1:
                        total_ppv_y1 = total_ppv_y1 + row[model]
                        total_size_y1 = total_size_y1 + 1
                        wrong_predicted_y1 = wrong_predicted_y1 + abs(row[model] - row[label])
                        if row[model] == 0:
                            total_fn += 1
                            num_neg += 1
                        else:
                            num_pos += 1

            result_df.at[df_count, "error_rate"] = wrong_predicted/total_size * 100
            #Iterate again for formula
            count = 0
            dp = 0
            eq_odd = 0
            eq_opp = 0
            tr_eq = 0
            impact = 0
            fp = 0
            fn = 0
            for key, item in grouped_df:
                model_ppv = 0
                model_size = 0
                model_ppv_y0 = 0
                model_size_y0 = 0
                model_ppv_y1 = 0
                model_size_y1 = 0
                part_df = grouped_df.get_group(key)
                for i, row in part_df.iterrows():
                    model_ppv = model_ppv + row[model]
                    model_size = model_size + 1
                    if row[label] == 0:
                        model_ppv_y0 = model_ppv_y0 + row[model]
                        model_size_y0 = model_size_y0 + 1
                        if row[model] == 1:
                            fp += 1
                    elif row[label] == 1:
                        model_ppv_y1 = model_ppv_y1 + row[model]
                        model_size_y1 = model_size_y1 + 1
                        if row[model] == 0:
                            fn += 1

                dp = dp + abs(model_ppv/model_size - total_ppv/total_size)
                eq_odd = (eq_odd + 0.5*abs(model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0)
                    + 0.5*abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1))
                eq_opp = eq_opp + abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1)
                if fp+fn == 0 and total_fp+total_fn == 0:
                    pass
                elif fp+fn == 0:
                    tr_eq = tr_eq + abs(0.5 - total_fp/(total_fp+total_fn))
                elif total_fp+total_fn == 0:
                    tr_eq = tr_eq + abs(fp/(fp+fn) - 0.5)
                else:
                    tr_eq = tr_eq + abs(fp/(fp+fn) - total_fp/(total_fp+total_fn))

            result_df.at[df_count, "demographic_parity"] = dp/(len(grouped_df)-1) * 100
            result_df.at[df_count, "equalized_odds"] = eq_odd/(len(grouped_df)-1) * 100
            result_df.at[df_count, "equal_opportunity"] = eq_opp/(len(grouped_df)-1) * 100
            result_df.at[df_count, "treatment_equality"] = tr_eq/(len(grouped_df)-1) * 100

            if lrd:
                lrd_dp = 0
                lrd_eod = 0
                lrd_eop = 0
                lrd_te = 0
                lrd_di = 0
                cc = 0
                cceod = 0
                cceop = 0
                ccte = 0
                for i, clusters in enumerate(cluster_ids):
                    cluster_df = df.loc[clusters]
                    cluster_size = len(cluster_df)
                    cluster_size_eod = len(cluster_df)
                    cluster_size_eop = len(cluster_df)
                    cluster_size_te = len(cluster_df)
                    cc += cluster_size        

                    grouped_df = cluster_df.groupby(sens_attrs)
                    total_ppv = 0
                    total_size = 0
                    total_ppv_y0 = 0
                    total_size_y0 = 0
                    total_ppv_y1 = 0
                    total_size_y1 = 0
                    discr_ppv_y0 = 0
                    discr_size_y0 = 0
                    discr_ppv_y1 = 0
                    discr_size_y1 = 0
                    wrong_predicted = 0
                    wrong_predicted_y0 = 0
                    wrong_predicted_y1 = 0
                    total_fp = 0
                    total_fn = 0
                    num_pos = 0
                    num_neg = 0
                    #counterfactual = 0
                    group_predsize = []
                    #Get the favored group to test against and also the averages over the whole dataset
                    di = dict()
                    for key, item in grouped_df:
                        predsize = 0
                        part_df = grouped_df.get_group(key)
                        for i, row in part_df.iterrows():
                            predsize += 1
                            total_ppv = total_ppv + row[model]
                            total_size = total_size + 1
                            wrong_predicted = wrong_predicted + abs(row[model] - row[label])
                            if row[label] == 0:
                                total_ppv_y0 = total_ppv_y0 + row[model]
                                total_size_y0 = total_size_y0 + 1
                                wrong_predicted_y0 = wrong_predicted_y0 + abs(row[model] - row[label])
                                if row[model] == 1:
                                    total_fp += 1
                                    num_pos += 1
                                else:
                                    num_neg += 1
                            elif row[label] == 1:
                                total_ppv_y1 = total_ppv_y1 + row[model]
                                total_size_y1 = total_size_y1 + 1
                                wrong_predicted_y1 = wrong_predicted_y1 + abs(row[model] - row[label])
                                if row[model] == 0:
                                    total_fn += 1
                                    num_neg += 1
                                else:
                                    num_pos += 1

                    #Iterate again for formula
                    count = 0
                    dp = 0
                    eq_odd = 0
                    eq_opp = 0
                    tr_eq = 0
                    impact = 0
                    fp = 0
                    fn = 0
                    for key, item in grouped_df:
                        model_ppv = 0
                        model_size = 0
                        model_ppv_y0 = 0
                        model_size_y0 = 0
                        model_ppv_y1 = 0
                        model_size_y1 = 0
                        part_df = grouped_df.get_group(key)
                        for i, row in part_df.iterrows():
                            model_ppv = model_ppv + row[model]
                            model_size = model_size + 1
                            if row[label] == 0:
                                model_ppv_y0 = model_ppv_y0 + row[model]
                                model_size_y0 = model_size_y0 + 1
                                if row[model] == 1:
                                    fp += 1
                            elif row[label] == 1:
                                model_ppv_y1 = model_ppv_y1 + row[model]
                                model_size_y1 = model_size_y1 + 1
                                if row[model] == 0:
                                    fn += 1

                        dp = dp + abs(model_ppv/model_size - total_ppv/total_size)

                        if total_size_y1 == 0 or model_size_y1 == 0 or total_size_y0 == 0 or model_size_y0 == 0:
                            cluster_size_eod -= model_size
                        else:
                            eq_odd = (eq_odd + 0.5*abs(model_ppv_y0/model_size_y0 - total_ppv_y0/total_size_y0)
                                + 0.5*abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1))
                        if total_size_y1 == 0 or model_size_y1 == 0:
                            cluster_size_eop -= model_size
                        else:
                            eq_opp = eq_opp + abs(model_ppv_y1/model_size_y1 - total_ppv_y1/total_size_y1)
                        if fp+fn == 0 and total_fp+total_fn == 0:
                            cluster_size_te -= model_size
                        elif fp+fn == 0:
                            tr_eq = tr_eq + abs(0.5 - total_fp/(total_fp+total_fn))
                        elif total_fp+total_fn == 0:
                            tr_eq = tr_eq + abs(fp/(fp+fn) - 0.5)
                        else:
                            tr_eq = tr_eq + abs(fp/(fp+fn) - total_fp/(total_fp+total_fn))

                    lrd_dp += dp/(len(grouped_df)-1) * 100 * cluster_size
                    lrd_eod += eq_odd/(len(grouped_df)-1) * 100 * cluster_size
                    lrd_eop += eq_opp/(len(grouped_df)-1) * 100 * cluster_size
                    lrd_te += tr_eq/(len(grouped_df)-1) * 100 * cluster_size

                    cceod += cluster_size_eod
                    cceop += cluster_size_eop
                    ccte += cluster_size_te

                result_df.at[df_count, "cluster_dp"] = lrd_dp / cc
                result_df.at[df_count, "cluster_eod"] = lrd_eod / cc
                result_df.at[df_count, "cluster_eop"] = lrd_eop / cc
                result_df.at[df_count, "cluster_te"] = lrd_te / cc

        df_count += 1

    start = time.time()
    #elif metric == "consistency":
    model_list2 = []
    for model in model_list:
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col="index")
            model_list2.append(model)
            if np.isnan(df.values).any():
                continue
        except:
            continue

    df = pd.read_csv(link + model_list2[0] + "_prediction.csv", index_col="index")
    for i, model in enumerate(model_list2):
        if i == 0:
            continue
        df2 = pd.read_csv(link + model + "_prediction.csv", index_col="index")
        df = pd.merge(df, df2[[model]], how="inner", left_index=True, right_index=True)

    #Now evaluate each model according to the metrics implemented.
    model_count = 0
    models_consistency = [0 for model in model_list2]
    #CONSISTENCY TEST, COMPARE PREDICTION TO PREDICTIONS OF NEIGHBORS
    consistency = 0
    for i, row_outer in df.iterrows():
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(dataset2.values)
        indices = nbrs.kneighbors(dataset2.loc[i].values.reshape(1, -1),\
            return_distance=False)
        real_indices = df.index[indices[0]].tolist()
        df_local = df.loc[real_indices]
        model_count = 0
        for model in model_list2:
            knn_ppv = 0
            knn_count = 0
            inacc = 0
            for j, row in df_local.iterrows():
                inacc += abs(row[model] - row[label])
                knn_ppv = knn_ppv + row[model]
                knn_count = knn_count + 1
            knn_pppv = knn_ppv/knn_count
            models_consistency[model_count] = models_consistency[model_count] + abs(df.loc[i][model] - knn_pppv)
            model_count = model_count + 1

    model_count = 0
    for model in model_list2:
        result_df.at[model_count, "consistency"] = models_consistency[model_count]/len(df) * 100
        model_count = model_count + 1
    #print(time.time() - start)

    for sens in sens_attrs:
        dataset2[sens] = df[sens]

    """
    local_knn_results = compute_local_fairness_knn(dataset2, df, y_true, model_list2, sens_attrs, k=10)
    model_count = 0
    local_metrics = ["dp", "eod", "te"]
    for model in model_list2:
        for metric in local_metrics:
            result_df.at[model_count, "knn_" + metric] = local_knn_results[model][metric] * 100
        model_count += 1
    """
    
    """
    start = time.time()
    model_list2 = []
    for model in model_list:
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col="index")
            if np.isnan(df.values).any():
                continue
            model_list2.append(model)
        except:
            continue

    df = pd.read_csv(link + model_list2[0] + "_prediction.csv", index_col="index")
    for i, model in enumerate(model_list2):
        if i == 0:
            continue
        df2 = pd.read_csv(link + model + "_prediction.csv", index_col="index")
        df = pd.merge(df, df2[[model]], how="inner", left_index=True, right_index=True)
    
    X = orig_datas.loc[:, orig_datas.columns != label]
    X = X[col_order]
    X_cop = copy.deepcopy(X)
    idx = list(X.columns).index(sens_attrs[0])
    X = X.to_numpy()
    
    model_count = 0
    for model in model_list2:
        consistency_score = individual_consistency(X, df[model].to_list(), idx, 10)
        used_model = joblib.load(link + model + "_model.pkl")
        lipschitz_score = local_lipschitzness(used_model, X_cop, sens_attrs, epsilon=1e-2)
        result_df.at[model_count, "consistencyV2"] = consistency_score * 100
        result_df.at[model_count, "lipschitz"] = lipschitz_score * 100
        model_count += 1
    print(time.time() - start)
    """


    return result_df, cluster_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="Name of the input .csv file.")
    parser.add_argument("--folder", type=str, help="Directory of the generated output files.")
    parser.add_argument("--index", default="index", type=str, help="Column name containing the index\
        of each entry. Default given column name: index.")
    parser.add_argument("--sensitive", type=str, help="List of column names of the sensitive attributes.")
    parser.add_argument("--label", type=str, help="Column name of the target value.")
    parser.add_argument("--favored", default=None, type=str, help="Tuple of favored group.\
        Otherwise some metrics can't be used. Default: None.")
    parser.add_argument("--models", default=None, type=str, help="List of models.")
    parser.add_argument("--metric", type=str, help="Chosen fairness metric.")
    parser.add_argument("--name", default="EVALUATION", type=str, help="Chosen evaluation file name.")
    parser.add_argument("--proxy", default="no", type=str, help="Choose the proxy strategy for FALCC.")
    parser.add_argument("--lam", default=0.5, type=float, help="Choose the lambda weight value.")
    parser.add_argument("--lrd", default="False", type=str, help="Choose if local cluster evaluation or not.")
    args = parser.parse_args()

    input_file = args.ds
    link = args.folder
    index = args.index
    sens_attrs = ast.literal_eval(args.sensitive)
    label = args.label
    favored = ast.literal_eval(args.favored)
    model_list = ast.literal_eval(args.models)
    metric = args.metric
    name = args.name
    proxy = args.proxy
    lam = float(args.lam)

    lrd = args.lrd == "True"
    result_df, cluster_res = full_evaluation(link, input_file, index, sens_attrs, label,
                                             favored, model_list, metric, proxy, lam, lrd=lrd)

    result_df.to_csv(link + name + "_" + str(input_file) + ".csv")
    if lrd:
        cluster_res.sort_values(by="c", inplace=True, kind="mergesort")
        cluster_res.to_csv(link + name + "_" + str(input_file) + "_CLUSTER.csv")