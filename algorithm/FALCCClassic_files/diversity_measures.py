import math
import numpy as np


class DiversityMeasures:
    """A class to calculate diversity measures for an ensemble of estimators.

    This class provides methods to compute Q-statistic, disagreement measure, and entropy
    for a given ensemble of estimators.

    Attributes
    ----------
    is_zero_vec : numpy.vectorize
        A vectorized version of the is_zero function.
    """

    def __init__(self):
        self.is_zero_vec = np.vectorize(self.is_zero)

    def is_zero(self, x):
        """Auxilaly function. Check if the input value is zero.

        Parameters
        ----------
        x : float or int
            The input value to be checked.

        Returns
        -------
        int
            1 if the input value is zero, 0 otherwise.
        """
        if x == 0:
            return 1
        else:
            return 0

    def correct_predictions(self, X, y, estimator):
        """Check which entries are correctly classified by an estimator and save it in an array.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Label vector relative to the test data X_train.
        estimator : estimator object
            This is assumed to implement the scikit-learn estimator interface.

        Returns
        -------
        numpy.ndarray
            An array with each entry being 0 for wrong corresponding classification and 1 for correct classification.
        """
        estimator_difference = estimator.predict(X).reshape(-1, 1) - y.values.reshape(-1, 1)
        return self.is_zero_vec(estimator_difference.reshape(1, -1)[0])

    def calculate_pairwise(self, X, y, estimators):
        """
        Calculate pairwise diversity metrics (Q-statistics and disagreement measure) for an ensemble of estimators.
        This function has quadratic runtime complexity in the number of estimators given! Thus it is not
        suitable for a high amount (>20) of estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Vector of labels relative to the prediction data X.
        estimators : list of estimator objects
            The base estimators of a trained ensemble.

        Returns
        -------
        tuple of numpy.ndarray
            The pairwise diversities for an ensemble of estimators.
        """
        n_estimators = len(estimators)
        Q = np.zeros((n_estimators, n_estimators))
        dis = np.zeros((n_estimators, n_estimators))
        maxDiv = 1
        for i in range(n_estimators):
            for k in range(i + 1, n_estimators):
                correct_predictions_added = self.correct_predictions(
                    X, y, estimators[i]
                ) + self.correct_predictions(X, y, estimators[k])
                N_00 = np.count_nonzero(correct_predictions_added == 0)
                N_11 = np.count_nonzero(correct_predictions_added == 2)
                correct_predictions_subtracted = self.correct_predictions(
                    X, y, estimators[i]
                ) - self.correct_predictions(X, y, estimators[k])
                N_10 = np.count_nonzero(correct_predictions_subtracted == 1)
                N_01 = np.count_nonzero(correct_predictions_subtracted == -1)

                Q[i, k] = (N_00 * N_11 - N_10 * N_01) / (
                    (N_00 * N_11 + N_10 * N_01) + 1
                )  # +1 so we never divide by 0
                if Q[i, k] < maxDiv:
                    maxDiv = Q[i, k]
                dis[i, k] = (N_01 + N_10) / (len(y))
        return (Q, dis, maxDiv)

    def calculate_entropy_measure(self, X, y, estimators):
        """Calculate the entropy for an ensemble of estimators.
        This function has linear runtime complexity in the number of estimators given!

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data vector, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like of shape (n_samples,)
            Label vector relative to the prediction data X.
        estimators : list of estimator objects
            The base estimators of a trained ensemble.

        Returns
        -------
        float
            The entropy for an ensemble of estimators.
        """
        total_samples = len(X)
        n_estimators = len(estimators)

        # to return min(l(z_j),L-l(z_j)) efficiently, check if l(z_j) is higher than L/2 and then
        # return the L-l(z_j) if true.
        def minimum_in_E_formular(correct_preds):
            if correct_preds >= math.ceil(n_estimators / 2):
                return n_estimators - correct_preds
            else:
                return correct_preds

        minimum_in_E_formular_vec = np.vectorize(minimum_in_E_formular)

        sum_of_correct_pred = np.zeros(total_samples)

        for estimator in estimators:
            sum_of_correct_pred = sum_of_correct_pred + self.correct_predictions(
                X, y, estimator
            )
        entropy = (
            1 / (total_samples * (n_estimators - math.ceil(n_estimators / 2)))
        ) * sum(minimum_in_E_formular_vec(sum_of_correct_pred))
        return entropy

    def get_ensemble_diversity(self, estimators, diversity_metric, X, y):
        if diversity_metric == "Q_statistic":
            # calculate the average Q-statistic for the ensemble
            Q, dis, maxDiv = self.calculate_pairwise(X, y, estimators)
            overall_Q = 2 / (len(estimators) * (len(estimators) - 1)) * Q.sum()
            return overall_Q, maxDiv
        elif diversity_metric == "dis_measure":
            # calculate the average disagreement measure for the ensemble
            dis = self.calculate_pairwise(X, y, estimators)[1]
            overall_dis = 2 / (len(estimators) * (len(estimators) - 1)) * dis.sum()
            return overall_dis
        elif diversity_metric == "entropy":
            entropy = self.calculate_entropy_measure(X, y, estimators)
            return entropy

    def QS_score(self, estimator, X, y, maxScore=False):
        Q, maxDiv = self.get_ensemble_diversity(
            estimator.estimators_, diversity_metric="Q_statistic", X=X, y=y
            )
        if maxScore:
            return maxDiv
        else:
            return Q

    def dis_score(self, estimator, X, y):
        return self.get_ensemble_diversity(
            estimator.estimators_, diversity_metric="dis_measure", X=X, y=y
        )

    def entropy_score(self, estimator, X, y):
        return self.get_ensemble_diversity(
            estimator.estimators_, diversity_metric="entropy", X=X, y=y
        )
