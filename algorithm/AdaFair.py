from .AdaFair_files.AdaFair import AdaFair
from .AdaFair_files.AdaFairEQOP import AdaFairEQOP
from .AdaFair_files.AdaFairSP import AdaFairSP
import numpy as np

class AdaFairClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 classifier,
                 metric,
                 estimators,
                 c,
                 CSB):
        """
        Args:
        """
        self.df_dict = df_dict
        self.classifier = classifier
        self.metric = metric
        self.estimators = estimators
        self.c = c
        self.CSB = CSB

    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        sens_idx = X_train.columns.get_loc(self.df_dict["sens_attrs"][0])
        label = self.df_dict["label"]
        if self.metric == "demographic_parity":
            self.model = AdaFairSP(estimator=self.classifier, n_estimators=self.estimators, CSB=self.CSB, trade_off_c=self.c, saIndex=sens_idx, saValue=self.df_dict["favored"])
        elif self.metric in ("equalized_odds", "equal_opportunity"):
            self.model = AdaFairEQOP(estimator=self.classifier, n_estimators=self.estimators, CSB=self.CSB, trade_off_c=self.c, saIndex=sens_idx, saValue=self.df_dict["favored"])
        else:
            self.model = AdaFair(estimator=self.classifier, n_estimators=self.estimators, CSB=self.CSB, trade_off_c=self.c, saIndex=sens_idx, saValue=self.df_dict["favored"])

        y_train_np = np.where(y_train.to_numpy() == 0, -1, 1).astype(int)
        self.model.fit(X_train.to_numpy(), y_train_np.reshape((y_train.to_numpy().shape[0],)))

        return self


    def predict(self, X_test):
        """
        Information

        Args:

        Returns:
        """
        pred = self.model.predict(X_test.to_numpy())
        pred = np.where(pred == -1, 0, 1)

        return pred
