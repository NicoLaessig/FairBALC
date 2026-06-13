from sklearn.model_selection import train_test_split
from .FAGTB_files.functions import FAGTB
import copy

class FAGTBClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self,
                 df_dict,
                 estimators,
                 learning_rate,
                 lam,
                 remove):
        """
        Args:
        """
        self.df_dict = df_dict
        self.estimators = estimators
        self.lr = learning_rate
        self.lam = lam
        self.remove = remove


    def fit(self, X_train, y_train):
        """
        Information

        Args:

        Returns:
        """
        if self.remove:
            for sens in self.df_dict["sens_attrs"]:
                X_train = X_train.drop(sens, axis=1)

        #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.4, random_state=42)
        X_val = copy.deepcopy(X_train)
        y_val = copy.deepcopy(y_train)

        self.model = FAGTB(n_estimators=self.estimators, learning_rate=self.lr, max_features=None,
            min_samples_split=2, min_impurity=None, max_depth=9, regression=None)

        self.model.fit(X_train.to_numpy(), y_train.to_numpy().reshape((y_train.to_numpy().shape[0],)), sensitive=X_train[self.df_dict["sens_attrs"][0]].values, LAMBDA=self.lam,
            Xtest=X_val.to_numpy(), yt=y_val.to_numpy().reshape((y_val.to_numpy().shape[0],)), sensitivet=X_val[self.df_dict["sens_attrs"][0]].values)

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

        pred = self.model.predict(X_test.to_numpy())

        for i, p in enumerate(pred):
            if p >= 0.5:
                pred[i] = 1
            else:
                pred[i] = 0

        return pred
        