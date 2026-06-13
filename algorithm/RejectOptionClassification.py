import copy
import pandas as pd
from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.datasets import BinaryLabelDataset

class AIF_RejectOptionClassification():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 metric,
                 eps,
                 remove=False,
                 training=False):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.metric = metric
        self.eps = eps
        self.remove = remove
        self.training = training


    def fit(self, X_train=None, y_train=None, dataset_orig_valid=None, dataset_orig_valid_pred=None):
        """
        Information

        Args:

        Returns:
        """
        if self.metric == "demographic_parity":
            metric = "Statistical parity difference"
        elif self.metric == "equalized_odds":
            metric = "Average odds difference"
        elif self.metric == "equal_opportunity":
            metric = "Equal opportunity difference"
        else:
            metric = "Statistical parity difference"

        if self.training:
            train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
            dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
            dataset_orig_train, dataset_orig_valid = dataset_train.split([0.4], shuffle=True)
            X_train = dataset_orig_train.features
            y_train = dataset_orig_train.labels.ravel()
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
            prediction = self.classifier.predict_proba(X_valid)[:,1]
            dataset_orig_valid_pred.scores = prediction.reshape(-1, 1)
            dataset_orig_valid_pred.labels = self.classifier.predict(X_valid).reshape(-1, 1)

        self.model = RejectOptionClassification(self.df_dict["unprivileged_groups"], self.df_dict["privileged_groups"], metric_name=metric, metric_ub=self.eps, metric_lb=-self.eps)
        self.model.fit(dataset_orig_valid, dataset_orig_valid_pred)

        return self


    def predict(self, X_test=None, dataset_test=None):
        """
        Information

        Args:

        Returns:
        """
        if self.training:
            #test_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
            X_test[self.df_dict["label"]] = 0
            dataset_test = BinaryLabelDataset(df=X_test, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])

            X_test = dataset_test.features
            if self.remove:
                X_test = dataset_test.convert_to_dataframe()[0]
                for sens in self.df_dict["sens_attrs"]:
                    X_test = X_test.drop(sens, axis=1)

            prediction = self.classifier.predict_proba(X_test)[:,1]
            dataset_test.scores = prediction.reshape(-1, 1)
            dataset_test.labels = self.classifier.predict(X_test).reshape(-1, 1)

        pred = list(self.model.predict(dataset_test).labels.ravel())

        return pred
