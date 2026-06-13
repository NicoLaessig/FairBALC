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


def full_evaluation(df, link, input_file, index, sens_attrs, label, favored, model_list, metric, proxy):
    #Read the original dataset
    if proxy == "reweigh":
        original_data = pd.read_csv("Datasets/reweigh/" + input_file + ".csv",  index_col=index)
    elif proxy == "remove":
        original_data = pd.read_csv("Datasets/removed/" + input_file + ".csv",  index_col=index)
    else:
        original_data = pd.read_csv("Datasets/" + input_file + ".csv", index_col=index)
    dataset = copy.deepcopy(original_data)
    original_data = original_data.drop(columns=[label])
    col_order = list(original_data.columns)
    for sens in sens_attrs:
        original_data = original_data.loc[:, original_data.columns != sens]

    """
    for i, model in enumerate(model_list):
        try:
            original_data_short = pd.read_csv(link + model + "_prediction.csv", index_col=index)
            modelnr = i
            break
        except:
            pass
    """

    """
    original_data_short = pd.merge(original_data_short, original_data, left_index=True, right_index=True)
    original_data_short = original_data_short.loc[:, original_data_short.columns != model_list[modelnr]]
    orig_datas = copy.deepcopy(original_data_short)
    original_data_short = original_data_short.loc[:, original_data_short.columns != label]
    valid_data = dataset.loc[orig_datas.index, :]

    for sens in sens_attrs:
        original_data_short = original_data_short.loc[:, original_data_short.columns != sens]
    dataset2 = copy.deepcopy(original_data_short)
    total_size = len(original_data_short)
    """

    groups = dataset[sens_attrs].drop_duplicates(sens_attrs).reset_index(drop=True)
    actual_num_of_groups = len(groups)
    sensitive_groups = []
    sens_cols = groups.columns
    for i, row in groups.iterrows():
        sens_grp = []
        for col in sens_cols:
            sens_grp.append(row[col])
        sensitive_groups.append(tuple(sens_grp))

    df_count = 0
    result_df = pd.DataFrame()
    cluster_res = pd.DataFrame()
    ccount = 0

    for model in model_list:
        """
        result_df.at[df_count, "model"] = model
        try:
            df = pd.read_csv(link + model + "_prediction.csv", index_col=index)
            if np.isnan(df.values).any():
                continue
        except:
            continue

        df = pd.merge(df, original_data_short, left_index=True, right_index=True)
        """

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
            
            dataset_pred.labels = df[str(model)]

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

            result_df.at[df_count, "model"] = str(model)
            result_df.at[df_count, "error_rate"] = class_metric_pred.error_rate() * 100
            result_df.at[df_count, "demographic_parity"] = abs(metric_pred.statistical_parity_difference()) * 100
            result_df.at[df_count, "equalized_odds"] = abs(class_metric_pred.average_abs_odds_difference()) * 100
            result_df.at[df_count, "equal_opportunity"] = abs(class_metric_pred.equal_opportunity_difference()) * 100
            treq1 = class_metric_pred.num_false_positives(True)/(class_metric_pred.num_false_positives(True) + class_metric_pred.num_false_negatives(True))
            treq2 = class_metric_pred.num_false_positives(False)/(class_metric_pred.num_false_positives(False) + class_metric_pred.num_false_negatives(False))
            result_df.at[df_count, "treatment_equality"] = abs(treq1 - treq2) * 100
            fp = class_metric_pred.num_false_positives()
            fn = class_metric_pred.num_false_negatives()
            result_df.at[df_count, "tp"] = class_metric_pred.num_true_positives()
            result_df.at[df_count, "tn"] = class_metric_pred.num_true_negatives()
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

        else:
            """
            TODO LRD einbauen von FALCC Paper.
            """
            result_df.at[df_count, "model"] = str(model)
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

        df_count += 1

    """
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
        real_indices = df.index[indices].tolist()
        df_local = df.loc[real_indices[0]]
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
    print(time.time() - start)


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
        #used_model = joblib.load(link + model + "_model.pkl")
        #lipschitz_score = local_lipschitzness(used_model, X_cop, sens_attrs, epsilon=1e-2)
        result_df.at[model_count, "consistencyV2"] = consistency_score * 100
        #result_df.at[model_count, "lipschitz"] = lipschitz_score * 100
        model_count += 1
    print(time.time() - start)
    """

    return result_df
