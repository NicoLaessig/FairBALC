import pandas as pd
from aif360.algorithms.preprocessing import Reweighing
from aif360.datasets import BinaryLabelDataset

class AIF_Reweighing():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        train_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
        dataset_train = BinaryLabelDataset(df=train_df, label_names=[self.df_dict["label"]], protected_attribute_names=self.df_dict["sens_attrs"])

        self.model = Reweighing(self.df_dict["unprivileged_groups"], self.df_dict["privileged_groups"])
        dataset_cleaned = self.model.fit_transform(dataset_train)

        dataset_cleaned, attr = dataset_cleaned.convert_to_dataframe()
        self.X_train = dataset_cleaned.loc[:, dataset_cleaned.columns != self.df_dict["label"]]
        self.y_train = dataset_cleaned[self.df_dict["label"]]

        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                self.X_train = self.X_train.drop(sens, axis=1)

        self.classifier.fit(self.X_train, self.y_train, attr["instance_weights"])

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                X_test = X_test.drop(sens, axis=1)

        pred = self.classifier.predict(X_test)

        return pred
        