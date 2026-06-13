import math
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)

# Optional XGBoost
try:
    import xgboost as xgb
except ImportError:
    xgb = None

from algorithm.FALCCClassic_files import DiversityMeasures


# -------------------------------------------------------------------
#  Single-tree weak-learner wrapper for XGBoost
# -------------------------------------------------------------------
class XGBTreeEstimator:
    """
    A lightweight sklearn-like estimator representing a single boosted
    XGBoost tree (boosting round) with a .predict(X) method.

    Works for **binary classification** only.

    Uses:
        booster.predict(DMatrix, iteration_range=(t, t+1), output_margin=True)

    Prediction rule:
        predict = (margin >= 0).astype(int)
    """

    def __init__(self, booster, tree_index):
        self.booster = booster
        self.tree_index = tree_index
        # feature names used at training time (can be None if training used numpy)
        self.feature_names = getattr(booster, "feature_names", None)

    def predict(self, X):
        # Align feature names to what the booster saw during training
        X_use = X

        if isinstance(X, pd.DataFrame) and self.feature_names is not None:
            # Select exactly the columns used in training, in the same order
            X_use = X.loc[:, self.feature_names]

        # Build DMatrix with aligned features
        dmat = xgb.DMatrix(X_use)

        # Output margin from only tree_index (t)
        margin = self.booster.predict(
            dmat,
            output_margin=True,
            iteration_range=(self.tree_index, self.tree_index + 1),
        )

        # Convert margin sign into binary prediction
        return (margin >= 0.0).astype(int)


# -------------------------------------------------------------------
#  HyperOptimizedLearner
# -------------------------------------------------------------------
class HyperOptimizedLearner:
    """
    Hyperparameter optimization for ensemble-based classifiers.

    Supports learners:
        - 'RandomForest'
        - 'AdaBoost'
        - 'ExtraTrees'
        - 'XGBoost'

    Uses entropy-based scoring:
        scoring = {'accuracy': 'accuracy',
                   'entropy': DiversityMeasures().entropy_score}
    refit='entropy'
    """

    def __init__(
        self,
        learner,
        search_method,
        input_file,
        sbt=False,
        n_iter=25,
        cv=5,
        ensemble_entropy=1,
        sens_attrs=None,
    ):
        self.learner = learner
        self.search_method = search_method
        self.input_file = input_file
        self.sbt = sbt
        self.estimators_ = None          # stored weak learners
        self.n_iter = n_iter
        self.cv = cv
        self.ensemble_entropy = ensemble_entropy
        self.sens_attrs = sens_attrs or []

        # keep your scoring exactly
        self.scoring = {
            "accuracy": "accuracy",
            "entropy": DiversityMeasures().entropy_score,
        }

    # -------------------------------------------------------------------
    #                              FIT
    # -------------------------------------------------------------------
    def fit(self, X, y):

        # ================================
        # 1) Parameter Distributions
        # ================================
        if len(self.sens_attrs) > 1:
            rf_param_grid = {
                "n_estimators": range(4, 6),
                "criterion": ["gini", "entropy"],
                "max_depth": range(1, 7),
            }
            ab_param_grid = {
                "n_estimators": range(4, 6),
                "estimator__criterion": ["gini", "entropy"],
                "estimator__max_depth": range(1, 7),
            }
        else:
            rf_param_grid = {
                "n_estimators": range(14, 20),
                "criterion": ["gini", "entropy"],
                "max_depth": range(1, 7),
            }
            ab_param_grid = {
                #"n_estimators": range(14, 20),
                #"estimator__criterion": ["gini", "entropy"],
                #"estimator__max_depth": range(1, 7),
                "n_estimators": [20, 30],
                "estimator__criterion": ["gini"],
                "estimator__max_depth": range(1, 3),
                "estimator__max_features": ["sqrt", 0.5],
                "estimator__min_samples_leaf": [5, 10],
                "learning_rate": [0.5, 1.0]
            }

        et_param_grid = rf_param_grid.copy()

        if xgb is not None:
            xgb_param_grid = {
                "n_estimators": [20],
                "max_depth": [2, 3, 4, 5],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "subsample": [0.6, 0.9],
                "colsample_bytree": [0.6, 0.9],
            }
        else:
            xgb_param_grid = None

        # ================================
        # 2) Build Search Object
        # ================================
        search = None

        # -------- RandomForest --------
        if self.search_method == "random" and self.learner == "RandomForest":
            search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                param_distributions=rf_param_grid,
                scoring=self.scoring,
                refit="entropy",
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=4,
            )

        elif self.search_method == "full" and self.learner == "RandomForest":
            search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid=rf_param_grid,
                scoring=self.scoring,
                refit="entropy",
                cv=self.cv,
                n_jobs=4,
            )

        # -------- AdaBoost --------
        elif self.search_method == "random" and self.learner == "AdaBoost":
            search = RandomizedSearchCV(
                AdaBoostClassifier(estimator=DecisionTreeClassifier()),
                param_distributions=ab_param_grid,
                scoring=self.scoring,
                refit="entropy",
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=4,
            )

        elif self.search_method == "full" and self.learner == "AdaBoost":
            search = GridSearchCV(
                AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(random_state=42)
                ),
                param_grid=ab_param_grid,
                scoring=self.scoring,
                refit="entropy",
                cv=self.cv,
                n_jobs=4,
            )

        # -------- ExtraTrees --------
        elif self.search_method == "random" and self.learner == "ExtraTrees":
            search = RandomizedSearchCV(
                ExtraTreesClassifier(random_state=42),
                param_distributions=et_param_grid,
                scoring=self.scoring,
                refit="entropy",
                n_iter=self.n_iter,
                cv=self.cv,
                n_jobs=4,
            )

        elif self.search_method == "full" and self.learner == "ExtraTrees":
            search = GridSearchCV(
                ExtraTreesClassifier(random_state=42),
                param_grid=et_param_grid,
                scoring=self.scoring,
                refit="entropy",
                cv=self.cv,
                n_jobs=4,
            )

        # -------- XGBoost --------
        elif self.learner == "XGBoost" and self.search_method in ("random", "full"):
            if xgb is None:
                raise ImportError("XGBoost not installed.")

            estimator = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=4,
                random_state=42,
            )

            if self.search_method == "random":
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions=xgb_param_grid,
                    scoring=self.scoring,
                    refit="entropy",
                    n_iter=self.n_iter,
                    cv=self.cv,
                    n_jobs=4,
                )
            else:
                search = GridSearchCV(
                    estimator,
                    param_grid=xgb_param_grid,
                    scoring=self.scoring,
                    refit="entropy",
                    cv=self.cv,
                    n_jobs=4,
                )

        else:
            raise ValueError(
                f"Unsupported combination learner={self.learner}, search={self.search_method}"
            )

        # ================================
        # 3) Run the Search
        # ================================
        search.fit(X, y)

        results = pd.DataFrame(search.cv_results_).sort_values(
            "rank_test_entropy", axis=0, ascending=True
        ).reset_index(drop=True)

        results.to_csv(
            f"Results/{self.learner}_{self.search_method}_{self.input_file}_sbt{self.sbt}_results.csv",
            index=False,
        )

        # ================================
        # 4) Select Best-Entropy Hyperparams
        # ================================
        entropies = search.cv_results_["mean_test_entropy"]
        params = search.cv_results_["params"]

        if self.ensemble_entropy == "mean":
            target = entropies.mean()
        elif self.ensemble_entropy == "middle":
            target = (entropies.max() + entropies.min()) / 2
        elif self.ensemble_entropy == "quarter":
            target = entropies.min() + (entropies.max() - entropies.min()) / 4
        elif self.ensemble_entropy == "three_quarters":
            target = entropies.min() + 3 * (entropies.max() - entropies.min()) / 4
        else:
            target = self.ensemble_entropy

        idx = np.abs(entropies - target).argmin()
        chosen_params = params[idx].copy()

        # ================================
        # 5) Train Final Ensemble
        # ================================
        if self.learner == "RandomForest":
            final = RandomForestClassifier(**chosen_params).fit(X, y)

        elif self.learner == "ExtraTrees":
            final = ExtraTreesClassifier(**chosen_params).fit(X, y)

        elif self.learner == "AdaBoost":
            inner_params = {}
            for k in list(chosen_params.keys()):
                if k.startswith("estimator__"):
                    inner_params[k.replace("estimator__", "")] = chosen_params.pop(k)
            final = AdaBoostClassifier(
                **chosen_params,
                estimator=DecisionTreeClassifier(**inner_params)
            ).fit(X, y)

        elif self.learner == "XGBoost":
            final = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                use_label_encoder=False,
                n_jobs=4,
                random_state=42,
                **chosen_params,
            ).fit(X, y)

        else:
            raise ValueError("Unsupported learner after search.")

        # ================================
        # 6) Extract Weak Learners
        # ================================
        if self.learner in ("RandomForest", "ExtraTrees", "AdaBoost"):
            self.estimators_ = final.estimators_

        elif self.learner == "XGBoost":
            booster = final.get_booster()
            n_trees = final.n_estimators
            self.estimators_ = [XGBTreeEstimator(booster, t) for t in range(n_trees)]

        else:
            self.estimators_ = None

        return
