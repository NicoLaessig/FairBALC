import copy
import pandas as pd
import numpy as np
from aif360.algorithms.postprocessing import EqOddsPostprocessing
from aif360.datasets import BinaryLabelDataset

class AIF_EqOddsPostprocessing():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 remove=False,
                 training=False):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.remove = remove
        self.training = training


    def fit(self, X_train=None, y_train=None, dataset_orig_valid=None, dataset_orig_valid_pred=None):
        """
        Information

        Args:

        Returns:
        """
        if self.training:
            train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)

            dataset_train = BinaryLabelDataset(
                df=train_df,
                label_names=[self.df_dict["label"]],
                protected_attribute_names=self.df_dict["sens_attrs"]
            )

            dataset_orig_train, dataset_orig_valid = dataset_train.split([0.4], shuffle=True)

            X_train = dataset_orig_train.features
            y_train = dataset_orig_train.labels.ravel().astype(float)

            X_valid = dataset_orig_valid.features

            if self.remove:
                X_train = dataset_orig_train.convert_to_dataframe()[0]
                X_train = X_train.loc[:, X_train.columns != self.df_dict["label"]]

                X_valid = dataset_orig_valid.convert_to_dataframe()[0]
                X_valid = X_valid.loc[:, X_valid.columns != self.df_dict["label"]]

                for sens in self.df_dict["sens_attrs"]:
                    X_train = X_train.drop(sens, axis=1)
                    X_valid = X_valid.drop(sens, axis=1)

            dataset_orig_valid_pred = copy.deepcopy(dataset_orig_valid)

            self.classifier.fit(X_train, y_train)

            # Get probability for favorable label 1.0 robustly
            proba = self.classifier.predict_proba(X_valid)

            if hasattr(self.classifier, "classes_"):
                positive_idx = list(self.classifier.classes_).index(1.0)
            else:
                positive_idx = 1

            prediction_scores = proba[:, positive_idx].astype(float)

            # AIF360 expects scores and labels as numeric 2D arrays
            dataset_orig_valid_pred.scores = prediction_scores.reshape(-1, 1)
            dataset_orig_valid_pred.labels = (prediction_scores >= 0.5).astype(float).reshape(-1, 1)

            # Also force original validation labels/protected attrs to numeric
            dataset_orig_valid.labels = dataset_orig_valid.labels.astype(float)
            dataset_orig_valid_pred.labels = dataset_orig_valid_pred.labels.astype(float)

            dataset_orig_valid.protected_attributes = (
                dataset_orig_valid.protected_attributes.astype(float)
            )
            dataset_orig_valid_pred.protected_attributes = (
                dataset_orig_valid_pred.protected_attributes.astype(float)
            )

        self.model = EqOddsPostprocessing(
            unprivileged_groups=self.df_dict["unprivileged_groups"],
            privileged_groups=self.df_dict["privileged_groups"]
        )

        self.model.fit(dataset_orig_valid, dataset_orig_valid_pred)

        return self


    #Is y_test required? What about thresholds?
    def predict(self, X_test=None, dataset_test=None):
        """
        Information

        Args:

        Returns:
        """
        if self.training:
            X_test = X_test.copy()

            # Dummy label required by BinaryLabelDataset
            X_test[self.df_dict["label"]] = 0.0

            dataset_test = BinaryLabelDataset(
                df=X_test,
                label_names=[self.df_dict["label"]],
                protected_attribute_names=self.df_dict["sens_attrs"]
            )

            X_test = dataset_test.features

            if self.remove:
                X_test = dataset_test.convert_to_dataframe()[0]

                # Remove label column
                X_test = X_test.loc[:, X_test.columns != self.df_dict["label"]]

                # Remove sensitive attributes
                for sens in self.df_dict["sens_attrs"]:
                    X_test = X_test.drop(sens, axis=1)

            proba = self.classifier.predict_proba(X_test)

            if hasattr(self.classifier, "classes_"):
                positive_idx = list(self.classifier.classes_).index(1.0)
            else:
                positive_idx = 1

            prediction_scores = proba[:, positive_idx].astype(float)

            dataset_test.scores = prediction_scores.reshape(-1, 1)
            dataset_test.labels = (prediction_scores >= 0.5).astype(float).reshape(-1, 1)

            dataset_test.protected_attributes = dataset_test.protected_attributes.astype(float)

        pred = list(self.model.predict(dataset_test).labels.ravel())

        return pred
