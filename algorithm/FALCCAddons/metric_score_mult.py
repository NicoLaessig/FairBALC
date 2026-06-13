"""
In this python file, the best classifier(s) or combination(s) of classifiers in terms of
accuracy & fairness are determined and returned.
"""
import copy
import math
import statistics
import pandas as pd
import numpy as np
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.neighbors import NearestNeighbors


def demographic_parity_score(y_true, y_pred, sensitive_attribute):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sensitive': sensitive_attribute})
    
    # Unique classes and sensitive attribute values
    classes = df['true'].unique()
    sensitive_values = df['sensitive'].unique()
    
    dp_scores = {}
    for cls in classes:
        cls_probs = []
        for val in sensitive_values:
            subset = df[df['sensitive'] == val]
            if len(subset) >= 1:
                prob = len(subset[subset['pred'] == cls]) / len(subset)
                cls_probs.append(prob)
        dp_score = 0
        for cp in cls_probs:
            dp_score += abs(cp - statistics.mean(cls_probs))
        dp_scores[cls] = dp_score
    
    return dp_scores


def equalized_odds_score(y_true, y_pred, sensitive_attribute):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sensitive': sensitive_attribute})
    
    # Unique classes and sensitive attribute values
    classes = df['true'].unique()
    sensitive_values = df['sensitive'].unique()
    
    eo_scores = {}
    for cls in classes:
        cls_probs = []
        for val in sensitive_values:
            subset = df[(df['sensitive'] == val) & (df['true'] == cls)]
            if len(subset) >= 1:
                prob = len(subset[subset['pred'] == cls]) / len(subset)
                cls_probs.append(prob)
        eo_score = 0
        for cp in cls_probs:
            eo_score += abs(cp - statistics.mean(cls_probs))

        #now iterate for FP
        cls_probs = []
        for val in sensitive_values:
            subset = df[(df['sensitive'] == val) & (df['true'] != cls)]
            if len(subset) >= 1:
                prob = len(subset[subset['pred'] == cls]) / len(subset)
                cls_probs.append(prob)
        eo_score2 = 0
        for cp in cls_probs:
            eo_score2 += abs(cp - statistics.mean(cls_probs))

        eo_scores[cls] = eo_score * 0.5 + eo_score2 * 0.5
    
    return eo_scores


def equal_opportunity_score(y_true, y_pred, sensitive_attribute):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sensitive': sensitive_attribute})
    
    # Unique classes and sensitive attribute values
    classes = df['true'].unique()
    sensitive_values = df['sensitive'].unique()
    
    eo_scores = {}
    for cls in classes:
        cls_probs = []
        for val in sensitive_values:
            subset = df[(df['sensitive'] == val) & (df['true'] == cls)]
            if len(subset) >= 1:
                prob = len(subset[subset['pred'] == cls]) / len(subset)
                cls_probs.append(prob)
        eo_score = 0
        for cp in cls_probs:
            eo_score += abs(cp - statistics.mean(cls_probs))
        eo_scores[cls] = eo_score
    
    return eo_scores


def treatment_equality_score(y_true, y_pred, sensitive_attribute):
    df = pd.DataFrame({'true': y_true, 'pred': y_pred, 'sensitive': sensitive_attribute})
    
    # Unique classes and sensitive attribute values
    classes = df['true'].unique()
    sensitive_values = df['sensitive'].unique()
    
    te_scores = {}
    for cls in classes:
        cls_probs2 = []
        cls_probs3 = []
        for val in sensitive_values:
            subset = df[(df['sensitive'] == val) & (df['true'] == cls)]
            subset2 = df[(df['sensitive'] == val) & (df['true'] != cls)]
            if len(subset) >= 1 and len(subset2) >= 1:
                prob = len(subset[subset['pred'] != cls]) / len(subset)
                cls_probs2.append(prob)
                
                prob = len(subset2[subset2['pred'] == cls]) / len(subset2)
                cls_probs3.append(prob)
        
        if len(cls_probs2) >= 1:
            avg_bal = 0
            for i, cp in enumerate(cls_probs2):
                avg_bal += cp/(cp + cls_probs3[i])

            avg_bal = avg_bal / len(cls_probs2)
            te_score = 0
            for i, cp in enumerate(cls_probs2):
                te_score += abs(cp/(cp + cls_probs3[i]) - avg_bal)*2

            te_scores[cls] = te_score
    
    return te_scores


class Metrics_Multi():
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


    def fairness_metric(self, df, model_comb, groups, favored, metric, weight=0.5, threshold=0,
        comb_amount=1):
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
                        for k in range(grp):
                            if grp[k] != row[self.sens_attrs[k]]:
                                same_group = False
                                break

                    if same_group:
                        preds.append(df.loc[i, comb[j]])

            y_true = df[self.label]

            error = sum(1 for x, y in zip(y_true, preds) if x != y) / len(preds)

            if metric == "demographic_parity":
                dp_scores = demographic_parity_score(y_true, preds, df[self.sens_attrs[0]])
                dp = 0
                for clf, score in dp_scores.items():
                    dp += score
                bias = dp/len(dp_scores)
            elif metric == "equalized_odds":
                eo_scores = equalized_odds_score(y_true, preds, df[self.sens_attrs[0]])
                eo = 0
                for clf, score in eo_scores.items():
                    eo += score
                bias = eo/len(eo_scores)
            elif metric == "equal_opportunity":
                eo_scores = equal_opportunity_score(y_true, preds, df[self.sens_attrs[0]])
                eo = 0
                for clf, score in eo_scores.items():
                    eo += score
                bias = eo/len(eo_scores)
            elif metric == "treatment_equality":
                te_scores = treatment_equality_score(y_true, preds, df[self.sens_attrs[0]])
                te = 0
                for clf, score in te_scores.items():
                    te += score
                bias = te/len(te_scores)
                
            new_comb_val = weight*(error) + (1-weight)*bias

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
        comb_amount = min(comb_amount, len(all_comb_list))
        if comb_amount != 1:
            sorted_list = sorted(all_comb_list, key=lambda x: x[1])
            best_comb_list = []
            for i in range(comb_amount):
                best_comb_list.append(sorted_list[i][0])
        elif (comb_amount == 1 and threshold == 0) or (best_comb_list == []):
            best_comb_list = list([best_comb])

        return best_comb_list
