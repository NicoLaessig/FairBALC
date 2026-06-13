from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_validate
from algorithm.FALCCClassic_files import DiversityMeasures


class CrossVal:
    """Cross validation implementations of random forest and adaboost"""

    def rf_cv(self, n_estimators, criterion, max_depth, max_features, X, y, score, cv):
        """Perform cross-validation for a random forest classifier and return the mean score.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        criterion : str
            The function to measure the quality of a split.
        max_depth : int
            The maximum depth of the tree.
        max_features : int, float, str or None
            The number of features to consider when looking for the best split.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        score : str
            The score to return ('Q_statistic', 'dis_measure', 'entropy').
        cv : int
            Number of cross-validation folds.

        Returns
        -------
        float
            The mean score of the selected metric.
        """
        estimator = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            random_state=42,
            n_jobs=-1,
        )
        scoring = {
            "accuracy": "accuracy",
            # "Q_statistic": DiversityMeasures().QS_score,
            # "dis_measure": DiversityMeasures().dis_score,
            "entropy": DiversityMeasures().entropy_score,
        }
        cval = cross_validate(estimator, X, y, scoring=scoring, cv=cv)

        if score == "Q_statistic":
            return -cval["test_Q_statistic"].mean()
        if score == "dis_measure":
            return cval["test_dis_measure"].mean()
        elif score == "entropy":
            return cval["test_entropy"].mean()

    def ada_cv(
        self,
        algorithm,
        n_estimators,
        base_estimator__criterion,
        base_estimator__splitter,
        base_estimator__max_depth,
        base_estimator__min_samples_split,
        base_estimator__min_samples_leaf,
        base_estimator__max_features,
        X,
        y,
        score,
    ):
        """Perform cross-validation for an AdaBoost classifier and return the mean score.

        Parameters
        ----------
        algorithm : str
            The algorithm to use in the AdaBoost classifier.
        n_estimators : int
            The maximum number of estimators at which boosting is terminated.
        base_estimator__criterion : str
            The function to measure the quality of a split.
        base_estimator__splitter : str
            The strategy used to choose the split at each node.
        base_estimator__max_depth : int
            The maximum depth of the tree.
        base_estimator__min_samples_split : int
            The minimum number of samples required to split an internal node.
        base_estimator__min_samples_leaf : int
            The minimum number of samples required to be at a leaf node.
        base_estimator__max_features : int, float, str or None
            The number of features to consider when looking for the best split.
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        score : str
            The score to return ('accuracy', 'Q_statistic', 'dis_measure', 'entropy').

        Returns
        -------
        float
            The mean score of the selected metric.
        """
        estimator = AdaBoostClassifier(
            algorithm=algorithm,
            n_estimators=n_estimators,
            base_estimator=DecisionTreeClassifier(
                criterion=base_estimator__criterion,
                splitter=base_estimator__splitter,
                max_depth=base_estimator__max_depth,
                min_samples_split=base_estimator__min_samples_split,
                min_samples_leaf=base_estimator__min_samples_leaf,
                max_features=base_estimator__max_features,
                random_state=42,
            ),
        )

        scoring = {
            "accuracy": "accuracy",
            # "Q_statistic": DiversityMeasures().QS_score,
            # "dis_measure": DiversityMeasures().dis_score,
            "entropy": entropy_score,
        }
        cval = cross_validate(estimator, X, y, scoring=scoring, cv=cv)
        if score == "Q_statistic":
            return -cval["test_Q_statistic"].mean()
        if score == "dis_measure":
            return cval["test_dis_measure"].mean()
        elif score == "entropy":
            return cval["test_entropy"].mean()
