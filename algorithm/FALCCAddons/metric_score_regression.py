"""
In this python file, the best classifier(s) or combination(s) of classifiers in terms of
accuracy & fairness are determined and returned.
"""
import copy
import math
import statistics
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.neighbors import NearestNeighbors
from aif360.datasets import StandardDataset
from aif360.metrics.metric import Metric
from aif360.metrics.dataset_metric import DatasetMetric
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm, wasserstein_distance

def demographic_parity_score(y_true, y_pred, sensitive_attribute):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sensitive': sensitive_attribute})
    
    # Unique classes and sensitive attribute values
    sensitive_values = df['sensitive'].unique()
    
    cls_probs = []
    for val in sensitive_values:
        subset = df[df['sensitive'] == val]
        prob = sum(subset['pred']) / len(subset)
        cls_probs.append(prob)
    dp_score = 0
    for cp in cls_probs:
        dp_score += abs(cp - statistics.mean(cls_probs))
    dp_score = dp_score / (len(sensitive_values) - 1)
    
    return dp_score


class Metrics_Reg():
    """This class is used to check the accuracy and fairness of given data via one of the
    given metrics.

    Parameters
    ----------
    sens_attrs: list of strings
        List of the column names of the sensitive attributes in the dataset.

    label: string
        String name of the target column.
    """
    def __init__(self, sens_attrs, label):
        self.sens_attrs = sens_attrs
        self.label = label
        self.clusterdf = pd.DataFrame(columns=["cnr","error","bias","score","comb"])


    def fairness_metric(self, df, model_comb, groups, favored, metric="wasserstein", weight=0.5, threshold=0,
        comb_amount=1, cnr=-1):
        """This function returns the best combination(s) of models for the tested dataset region.

        Parameters
        ----------
        df: DataFrame, shape (n_samples, m_features)
            Each entry contains index, model name, values of the sensitive attributes, number
            and probability of positive predicted values, number and probability of wrong
            predicted label, number of entries in the group. An entry is made for each model
            + sensitive group combination.

        models: list of strings OR list of tuples of strings
            IF list of strings: List of all model names which are considered. Each possible
            combinations of each models are created.
            IF list of tuples: List of fixed combinations which are considered.

        favored: tuple of float
            Tuple of the values of the favored group.

        metric: string
            Name of the metric which should be used to get the best result.

        weight: float (0-1)
            Value to balance the accuracy and fairness parts of the metrics.
            Under 0.5: Give fairness higher importance.
            Over 0.5: Give accuracy higher importance.

        threshold: float
            Each combination which metric value score stays under that threshold is added to the
            output list. If no combination has its value under the threshold, the best combination
            is returned.

        comb_amount: integer
            Number of combinations which are returned.


        Returns
        ----------
        best_comb_list: list of list of strings
            List of best model combinations.

        group_list: list of float
            List of the values of the sensitive attributes/keys for each sensitive group.
        """
        best_comb_val = math.inf

        best_comb = model_comb[0]
        best_comb_list = []
        all_comb_list = []
        if isinstance(favored, int) or isinstance(favored, float):
            dataset_true = StandardDataset(df, 
                label_name=self.label, 
                favorable_classes=[1], 
                protected_attribute_names=self.sens_attrs, 
                privileged_classes=[[favored]])
        else:
            dataset_true = StandardDataset(df, 
                label_name=self.label, 
                favorable_classes=[1], 
                protected_attribute_names=self.sens_attrs, 
                privileged_classes=[favored])

        for comb in model_comb:
            gdf = df.groupby(self.sens_attrs)

            preds = []
            for i, row in df.iterrows():
                for j, grp in enumerate(groups):
                    same_group = True
                    if isinstance(grp, int) or isinstance(grp, float):
                        if grp != row[self.sens_attrs[0]]:
                            same_group = False

                    else:
                        for k in range(len(grp)):
                            if grp[k] != row[self.sens_attrs[k]]:
                                same_group = False
                                break

                    if same_group:
                        preds.append(df.loc[i, comb[j]])

            dataset_pred = copy.deepcopy(dataset_true)
            dataset_pred.labels = np.array(preds).reshape(-1, 1)

            #attr = dataset_pred.protected_attribute_names[0]
            #idx = dataset_pred.protected_attribute_names.index(attr)
            #privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
            #unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]
            priv_dict = dict()
            unpriv_dict = dict()
            if isinstance(favored, tuple):
                for i, fav_val in enumerate(favored):
                    priv_dict[self.sens_attrs[i]] = fav_val
                    all_val = list(df.groupby(self.sens_attrs[i]).groups.keys())
                    for poss_val in all_val:
                        if poss_val != fav_val:
                            unpriv_dict[self.sens_attrs[i]] = poss_val
            else:
                if favored == 0:
                    priv_dict[self.sens_attrs[0]] = 0
                    unpriv_dict[self.sens_attrs[0]] = 1
                elif favored == 1:
                    priv_dict[self.sens_attrs[0]] = 1
                    unpriv_dict[self.sens_attrs[0]] = 0

            privileged_groups = [priv_dict]
            unprivileged_groups = [unpriv_dict]

            metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
            class_metric_pred = ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

            # Compute RMSE disparity
            test_sens = dataset_pred.convert_to_dataframe()[0][self.sens_attrs[0]]
            preds = dataset_pred.labels.reshape(-1,)
            truth = dataset_true.labels.reshape(-1,)
            privileged_pred = []
            privileged_true = []
            unprivileged_pred = []
            unprivileged_true = []
            c = 0
            for i, val in test_sens.items():
                if val == 0:
                    privileged_pred.append(preds[c])
                    privileged_true.append(truth[c])
                else:
                    unprivileged_pred.append(preds[c])
                    unprivileged_true.append(truth[c])
                c += 1


            new_df = dataset_true.convert_to_dataframe()[0]
            new_df["pred"] = preds
            rmse, mae, r2, wasserstein = 0, 0, 0, 0

            avg_rmse = mean_squared_error(new_df[self.label], new_df["pred"], squared=False)
            avg_mae = mean_absolute_error(new_df[self.label], new_df["pred"])
            avg_r2 = r2_score(new_df[self.label], new_df["pred"])
            total_pred = new_df["pred"]

            gdf = new_df.groupby(self.sens_attrs)
            for key, grp_df in gdf:
                #grp_df = gdf.get_group(key)
                grp_rmse = mean_squared_error(grp_df[self.label], grp_df["pred"], squared=False)
                grp_mae = mean_absolute_error(grp_df[self.label], grp_df["pred"])
                grp_r2 = r2_score(grp_df[self.label], grp_df["pred"])
                grp_wasserstein = wasserstein_distance(grp_df["pred"], total_pred)

                rmse += abs(avg_rmse - grp_rmse)
                mae += abs(avg_mae - grp_mae)
                r2 += abs(avg_r2 - grp_r2)
                wasserstein += grp_wasserstein

            rmse = rmse/(len(gdf)-1)
            mae = mae/(len(gdf)-1)
            r2 = r2/(len(gdf)-1)
            wasserstein = wasserstein/(len(gdf)-1)

            #dp = demographic_parity_score(new_df[self.label], new_df["pred"], new_df[self.sens_attrs[0]])
            ####TODOOOOOOOOO
            if metric == "wasserstein":
                new_comb_val = weight*(avg_rmse) + (1-weight)*wasserstein
                bias = wasserstein
            elif metric == "rmse_disparity":
                new_comb_val = weight*(avg_rmse) + (1-weight)*rmse
                bias = rmse

            self.clusterdf.loc[len(self.clusterdf)] = {
                "cnr": cnr,
                "error": avg_rmse,
                "bias": bias,
                "score": new_comb_val,
                "comb": comb,
                "opt": ""
            }

            if new_comb_val < best_comb_val:
                best_comb_val = new_comb_val
                best_comb = comb
            if new_comb_val <= threshold:
                best_comb_list.append(comb)
            all_comb_list.append((comb, new_comb_val))

        #If comb_amount is set to a value x > 1. Return the x best model combinations.
        #If neither the comb_amount or threshold are manually set to a different value
        #OR if no combination is under the given threshold, return only the best model
        #combination as a list (so it can be handled like if multiple combinations are returned).
        #Else (threshold manually set) return a list of all combinations of models which metric
        #score are under the given threshold.
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

        return best_comb_list
