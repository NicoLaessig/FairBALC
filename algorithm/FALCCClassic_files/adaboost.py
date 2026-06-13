"""
This file implements the adapted AdaBoostClassifier which trains multiple classifiers in each
iteration and keeps the best one.
"""
import math
import copy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from algorithm.FALCCClassic_files import DiversityMeasures


class AdaBoostClassifierMult:
    """Multiple different model learners are part of this class.

    Parameters
    ----------
    base_estimator_list: list of models
        List of classifiers that will be trained in each iteration.

    iterations: integer
        Amount of (maximum) iterations == Amount of classifiers in the output.
    self.estimators_: list of models
        List that contains the classifiers of each iteration.
    """
    def __init__(self, base_estimator_list, iterations=6):
        self.base_estimator_list = base_estimator_list
        self.iterations = iterations
        self.estimators_ = []


    def fit(self, X, y):
        """Takes as input the model that will be trained and will return the trained model
        name and will save the model as .pkl & also save some informations in the dictionary.

        Parameter
        -------
        X: {array-like, sparse matrix}, shape (n_samples, m_features)
            Prediction data vector, where n_samples is the number of samples and
            m_features is the number of features.

        y: array-like, shape (n_samples)
            Label vector relative to the prediction data X_pred.


        Returns
        -------
        self.estimators_: list of models
            List that contains the best classifier of each iteration.
        """
        dataset_length = len(X)

        sample_weight = [1/dataset_length for i in range(dataset_length)]
        sample_weight2 = [1 for i in range(dataset_length)]


        for i in range(self.iterations):
            estimator = self.base_estimator_list[i%3]
            if isinstance(estimator, Pipeline):
                current_estimator = copy.deepcopy(estimator.fit(X, y,
                    **{'clf__sample_weight': sample_weight2}))
            elif isinstance(estimator, DecisionTreeClassifier):
                current_estimator = copy.deepcopy(estimator.fit(X, y, sample_weight))
            else:
                current_estimator = copy.deepcopy(estimator.fit(X, y, sample_weight2))

            estimator_predictions = current_estimator.predict(X)
            error = 0
            wrong_classified = []
            correct_classified = []
            for j, pred in enumerate(estimator_predictions):
                if pred != y.iloc[j].values[0]:
                    error += sample_weight[j]
                    wrong_classified.append(j)
                else:
                    correct_classified.append(j)

            self.estimators_.append(current_estimator)
            alpha = 1/2 * np.log((1-error)/(error + 1e-24))

            for j in wrong_classified:
                sample_weight[j] = sample_weight[j] * math.exp(alpha)
                sample_weight2[j] += 1

            for j in correct_classified:
                sample_weight[j] = sample_weight[j] * math.exp(-alpha)

            weight_total = sum(sample_weight)
            weight_total2 = sum(sample_weight2)

            for j in range(len(sample_weight)):
                sample_weight[j] = sample_weight[j] / weight_total
                sample_weight2[j] = sample_weight2[j] / weight_total2 * dataset_length

        diversity = DiversityMeasures().entropy_score(self, X, y)
        
        print("Entropy: "+ str(diversity))

        return self.estimators_
