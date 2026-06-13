import pandas as pd
from aif360.algorithms.preprocessing import LFR
from aif360.datasets import BinaryLabelDataset


class AIF_LFR():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 k,
                 Ax,
                 Ay,
                 Az,
                 transform,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.k = k
        self.Ax = Ax
        self.Ay = Ay
        self.Az = Az
        self.transform = transform
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])
        self.model = LFR(self.df_dict["unprivileged_groups"], self.df_dict["privileged_groups"], k=self.k, Ax=self.Ax, Ay=self.Ay, Az=self.Az)
        self.model = self.model.fit(dataset_train)
        dataset_transf_train = self.model.transform(dataset_train)
        dataset_transf_train = dataset_transf_train.convert_to_dataframe()[0]
        X_train = dataset_transf_train.loc[:, dataset_transf_train.columns != self.df_dict["label"]]
        y_train = dataset_transf_train[self.df_dict["label"]]
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                X_train = X_train.drop(sens, axis=1)

        #In case that every label is set to the same value
        label_vals = y_train.unique()
        if len(label_vals) == 1:
            self.same_label = True
            self.label_val = label_vals[0]
        else:
            self.same_label = False
            self.classifier.fit(X_train, y_train)

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if not self.transform:
            if self.same_label:
                pred = [self.label_val for i in range(len(X_test))]
            else:
                if self.remove:
                    for sens in self.df_dict["sens_attrs"]:
                        X_test = X_test.drop(sens, axis=1)

                pred = self.classifier.predict(X_test)
        else:
            dummy_y = pd.Series(0, index=X_test.index, name=self.df_dict["label"])
            test_df = pd.merge(X_test, dummy_y, left_index=True, right_index=True)

            dataset_test = BinaryLabelDataset(
                df=test_df,
                label_names=[self.df_dict["label"]],
                protected_attribute_names=self.df_dict["sens_attrs"]
            )

            dataset_transf_test = self.model.transform(dataset_test)
            dataset_transf_test = dataset_transf_test.convert_to_dataframe()[0]

            X_test_transf = dataset_transf_test.drop(columns=[self.df_dict["label"]])

            if self.remove:
                for sens in self.df_dict["sens_attrs"]:
                    X_test_transf = X_test_transf.drop(sens, axis=1)

            pred = self.classifier.predict(X_test_transf)

        return pred
