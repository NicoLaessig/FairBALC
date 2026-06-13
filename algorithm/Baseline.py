from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

class LogisticRegressionClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = LogisticRegression(solver='lbfgs')
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred



class DecisionTreeClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = DecisionTreeClassifier()
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred



class SVMClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = SVC()
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred



class MLPClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred



class kNNClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = KNeighborsClassifier()
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred



class RandomForestClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred



class AdaBoostClass():
    """
    information [...]

    References:
        ...
    """
    def __init__(self, df_dict, remove):
        """
        Args:
        """
        self.df_dict = df_dict
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
        self.model = AdaBoostClassifier()
        self.model.fit(X_train, y_train)

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
        pred = self.model.predict(X_test)

        return pred
