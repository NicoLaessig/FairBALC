"""
In this python file multiple classification models are trained.
"""
import copy
import re
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor, Ridge, Lars, HuberRegressor, BayesianRidge, ElasticNet, Lasso
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.utils.validation import check_X_y, check_array
from xgboost import XGBClassifier
import lightgbm as lgb
from algorithm.FALCCClassic_files import AdaBoostClassifierMult
from algorithm.FALCCClassic_files import HyperOptimizedLearner
from algorithm.FairBoost import FairBoost
from sklearn.ensemble import ExtraTreesClassifier
from algorithm import (
    AdaFairClass,
    FairBoost,
)

class RandomParamTreeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=0):
        self.random_state = random_state

    def fit(self, X, y):
        #X, y = check_X_y(X, y)
        rng = np.random.RandomState(self.random_state)

        # stärkere Diversität, weil nur 30 Mitglieder
        max_depth = rng.choice([None, 3, 4, 5, 6, 8, 10, 12])
        min_samples_leaf = rng.choice([1, 2, 5, 10, 20])
        min_samples_split = rng.choice([2, 5, 10, 20, 50])
        mf_kind = rng.choice(["float", "str"])
        if mf_kind == "float":
            max_features = float(rng.choice([0.3, 0.5, 0.7, 1.0]))
        else:
            max_features = rng.choice(["sqrt", "log2"])
        
        self.estimator_ = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=1.0,
            random_state=rng.randint(0, 2**31 - 1)
        )

        self.estimator_.fit(X, y)

        self.drawn_params_ = self.estimator_.get_params()
        return self

    def predict(self, X):
        X = check_array(X)
        return self.estimator_.predict(X)

try:
    
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False

class _PredictorEntry:
    """
    Wraps a fitted classifier with the metadata needed to apply it to new data.

    Some classifiers in the pool were fit on a feature subset (subgroup-specific,
    random subspace) or on transformed inputs. This wrapper records the
    feature columns the classifier expects, so .predict() at inference time
    can slice the input correctly.
    """

    def __init__(self, classifier, feature_cols=None, source=None, meta=None):
        self.classifier = classifier
        self.feature_cols = feature_cols  # None means "use all input features"
        self.source = source
        self.meta = meta or {}

    def predict(self, X):
        if self.feature_cols is not None:
            if hasattr(X, "loc"):
                X_for_pred = X[self.feature_cols]
            else:
                # numpy array path; assume feature_cols is a positional index list
                X_for_pred = X[:, self.feature_cols]
        else:
            X_for_pred = X
        return self.classifier.predict(X_for_pred)

class PoolModel:
    """
    Container for a heterogeneous predictor pool. Exposes .predict() that
    delegates to the first predictor in the pool (used only when downstream
    code calls model.predict() directly; FairBALC's per-cluster selection
    bypasses this and uses the pool directly).
    """

    def __init__(self, entries):
        self.entries = entries

    def predict(self, X):
        # Default: predict with the first entry. Most callers should use
        # the entry list directly instead of calling .predict() here.
        if not self.entries:
            raise ValueError("PoolModel is empty")
        return self.entries[0].predict(X)

    def __len__(self):
        return len(self.entries)


def _make_base_factories(seed=42):
    """Return a dict of name -> factory functions producing fresh classifier instances."""
    factories = {
        "LR": lambda: LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear", max_iter=200, random_state=seed,
        ),
        "RF": lambda: RandomForestClassifier(
            n_estimators=100, max_features="sqrt", n_jobs=-1, random_state=seed,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed,
        ),
        "GBM": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed,
        ),
        "GBM_shallow": lambda: GradientBoostingClassifier(
            n_estimators=200,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            random_state=seed,
        ),

        "GBM_deep": lambda: GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            random_state=seed,
        )
    }
    if _HAVE_XGB:
        factories["XGB"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", verbosity=0,
            n_jobs=-1, random_state=seed,
        )

        factories["XGB_shallow"] = lambda: XGBClassifier(
            n_estimators=400,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=seed,
        )

        factories["XGB_regularized"] = lambda: XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=3.0,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0,
            n_jobs=-1,
            random_state=seed,
        )

    return factories


def _source_adafair(X_train, y_train, sensitive_train, base_factories,
                df_dict, metric):
    """
    AdaFair sweep across (base_classifier, constraint, eps).

    Uses the existing AIF_ExponentiatedGradientReduction wrapper for
    consistency with the rest of the framework.
    """
    entries = []
    #for base_name, factory in base_factories.items():
    #try:
    classifier = DecisionTreeClassifier(max_depth=1)
    model = AdaFairClass(df_dict, classifier, metric, 200, 1.0, "CSB1")

    model.fit(X_train.copy(), y_train.copy())

    # Extract internal predictors (triple-nested .model)
    predictors = model.model.estimators_
    for t, h in enumerate(predictors):
        entries.append(_PredictorEntry(
            classifier=h, feature_cols=None,
            source="AdaFair"
        ))

    #except Exception as e:
        #   warnings.warn(
        #      f"AdaFair failed for base={base_name}"
        # )
    return entries


def _source_egr(X_train, y_train, sensitive_train, base_factories,
                eps_grid, constraints, df_dict):
    """
    EGR sweep across (base_classifier, constraint, eps).

    Uses the existing AIF_ExponentiatedGradientReduction wrapper for
    consistency with the rest of the framework.
    """
    entries = []
    for base_name, factory in base_factories.items():
        for cname, metric_str in constraints:
            for eps in eps_grid:
                try:
                    model = AIF_ExponentiatedGradientReduction(
                        df_dict, factory(), metric_str,
                        eps=eps, learning_rate=2.0, nu=None, remove=False, reg=False,
                    )
                    model.fit(X_train.copy(), y_train.copy())
                    # Extract internal predictors (triple-nested .model)
                    predictors = model.model.model.model_.predictors_
                    for t, h in enumerate(predictors):
                        entries.append(_PredictorEntry(
                            classifier=h, feature_cols=None,
                            source="EGR",
                            meta={"base": base_name, "constraint": cname,
                                "eps": eps, "iterate": t},
                        ))
                except Exception as e:
                    warnings.warn(
                        f"EGR failed for base={base_name}, "
                        f"constraint={cname}, eps={eps}: {e}"
                    )
    return entries


def _source_gsr(X_train, y_train, sensitive_train, base_factories,
                constraints, df_dict, grid_size=15):
    """GSR sweep across (base_classifier, constraint)."""
    entries = []
    for base_name, factory in base_factories.items():
        for cname, metric_str in constraints:
            try:
                model = AIF_GridSearchReduction(
                    df_dict, factory(), metric_str,
                    grid_size=10, lam=0.5, remove=False, reg=False,
                )
                model.fit(X_train.copy(), y_train.copy())
                # Extract internal predictors (triple-nested .model)
                predictors = model.model.model.model_.predictors_
                for t, h in enumerate(predictors):
                    entries.append(_PredictorEntry(
                        classifier=h, feature_cols=None,
                        source="GSR",
                        meta={"base": base_name, "constraint": cname,
                            "iterate": t},
                    ))
            except Exception as e:
                warnings.warn(
                    f"GSR failed for base={base_name}, "
                    f"constraint={cname}"
                )

    return entries


def _source_rf_trees(X_train, y_train, n_estimators=100,
                    seed=42):
    """Extract individual trees from a trained Random Forest."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_features="sqrt",
        n_jobs=-1, random_state=seed,
    )
    rf.fit(X_train, y_train)
    entries = []
    for i, tree in enumerate(rf.estimators_):
        entries.append(_PredictorEntry(
            classifier=tree, feature_cols=None,
            source="RF_tree",
            meta={"tree_idx": i},
        ))
    return entries


def _source_adaboost(X_train, y_train, n_estimators=20,
                    seed=42):
    """Extract AdaBoost iterates as individual predictors."""
    ab = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(
            criterion="gini", max_depth=1, splitter="random",
            max_features="sqrt", min_samples_leaf=5, random_state=seed,
        ),
        n_estimators=n_estimators,
        random_state=seed,
    )
    ab.fit(X_train, y_train)
    entries = []
    for i, est in enumerate(ab.estimators_):
        entries.append(_PredictorEntry(
            classifier=est, feature_cols=None,
            source="AdaBoost",
            meta={"iter": i},
        ))
    return entries


def _source_cost_sensitive(X_train, y_train, sensitive_train,
                        base_factories, favorable_label=1):
    """
    Train base classifiers with different sample-weight schemes that emphasise
    different (group, label) combinations. Each scheme produces a classifier
    sitting at a different point on the fairness frontier.
    """
    n = len(y_train)
    pos_rate = max(np.mean(y_train), 1e-3)
    neg_rate = max(1 - pos_rate, 1e-3)
    priv_rate = max(np.mean(sensitive_train), 1e-3)
    unpriv_rate = max(1 - priv_rate, 1e-3)

    y_train = y_train.squeeze()

    values, counts = np.unique(sensitive_train, return_counts=True)
    majority_group = values[np.argmax(counts)]
    minority_group = values[np.argmin(counts)]

    rates = {}
    for g in np.unique(sensitive_train):
        rates[g] = np.mean(y_train[sensitive_train == g])

    privileged_group = max(rates, key=rates.get)
    unprivileged_group = min(rates, key=rates.get)

    pos_rate = max(np.mean(y_train == favorable_label), 1e-3)
    neg_rate = max(1.0 - pos_rate, 1e-3)

    majority_rate = max(np.mean(sensitive_train == majority_group), 1e-3)
    minority_rate = max(np.mean(sensitive_train == minority_group), 1e-3)

    weight_schemes = [
        ("uniform", np.ones(n)),
        ("minority_class",
            np.where(y_train == favorable_label,
                     1.0 / pos_rate,
                     1.0 / neg_rate)),
        ("minority_group",
            np.where(sensitive_train == minority_group,
                     1.0 / minority_rate,
                     1.0 / majority_rate)),
        ("unpriv_favorable",
            np.where((sensitive_train == unprivileged_group)
                     & (y_train == favorable_label),
                     4.0,
                     1.0)),
        ("priv_unfavorable",
            np.where((sensitive_train == privileged_group)
                     & (y_train != favorable_label),
                     4.0,
                     1.0)),
    ]

    entries = []
    for base_name, factory in base_factories.items():
        for scheme_name, weights in weight_schemes:
            try:
                clf = factory()
                clf.fit(X_train, y_train, sample_weight=weights)
                entries.append(_PredictorEntry(
                    classifier=clf, feature_cols=None,
                    source="cost_sensitive",
                    meta={
                        "base": base_name,
                        "scheme": scheme_name,
                        "privileged_group": privileged_group,
                        "unprivileged_group": unprivileged_group,
                        "majority_group": majority_group,
                        "minority_group": minority_group,
                    },
                ))
            except (TypeError, ValueError):
                # Some classifiers don't accept sample_weight
                pass

    return entries


def _source_subgroup(X_train, y_train, sensitive_train,
                    base_factories, min_group_size=30):
    """
    Train one classifier per sensitive group on that group's data only.
    Useful as a candidate for the per-group slot in FairBALC's combinations.
    """
    entries = []
    feature_cols = list(X_train.columns) if hasattr(X_train, "columns") else None
    for base_name, factory in base_factories.items():
        for group in np.unique(sensitive_train):
            mask = sensitive_train == group
            if mask.sum() < min_group_size:
                continue
            try:
                X_sub = X_train.loc[mask] if hasattr(X_train, "loc") else X_train[mask]
                y_sub = y_train.loc[mask] if hasattr(y_train, "loc") else y_train[mask]
                clf = factory()
                clf.fit(X_sub, y_sub)
                entries.append(_PredictorEntry(
                    classifier=clf, feature_cols=None,
                    source="subgroup",
                    meta={"base": base_name, "group": int(group)},
                ))
            except Exception as e:
                warnings.warn(f"subgroup failed for {base_name}/{group}: {e}")
    return entries

def _compute_predictions(entries, X_val):
    """Compute predictions for each entry on the validation set; drop entries that error out."""
    kept = []
    preds_list = []
    for entry in entries:
        try:
            preds = np.asarray(entry.predict(X_val)).astype(int)
            preds = np.where(preds <= 0, 0, 1)
            kept.append(entry)
            preds_list.append(preds)
        except Exception as e:
            warnings.warn(f"Prediction failed for {entry.source}: {e}")
    return kept, np.array(preds_list) if preds_list else np.empty((0, len(X_val)))


def _dedup_by_predictions(entries, preds_matrix, threshold=0.02):
    """
    Remove entries whose validation predictions agree with an already-kept
    entry's predictions on more than (1 - threshold) of the samples.
    """
    if len(entries) == 0:
        return entries, preds_matrix

    n_val = preds_matrix.shape[1]
    kept_idx = []
    for i in range(len(entries)):
        is_dup = False
        for j in kept_idx:
            disagreement = np.mean(preds_matrix[i] != preds_matrix[j])
            if disagreement < threshold:
                is_dup = True
                break
        if not is_dup:
            kept_idx.append(i)

    kept_entries = [entries[i] for i in kept_idx]
    kept_preds = preds_matrix[kept_idx]
    return kept_entries, kept_preds


def _quality_filter(entries, preds_matrix, y_val, sensitive_val, max_er=0.5, max_dp=0.5):
    """
    Drop predictors with extreme error rate or DP violation. Returns kept
    entries, their predictions, and a per-predictor (er, dp) score array
    used by stratified_trim.
    """
    y_val = np.asarray(y_val).astype(int)
    sensitive_val = np.asarray(sensitive_val).astype(int)

    kept_idx = []
    scores = []
    for i, _ in enumerate(entries):
        preds = preds_matrix[i]
        er = float(np.mean(preds != y_val))
        if er >= max_er:
            continue
        marginal = float(np.mean(preds == 1))
        groups = np.unique(sensitive_val)
        if len(groups) < 2:
            dp = 0.0
        else:
            dp = float(np.mean([
                abs(np.mean(preds[sensitive_val == g] == 1) - marginal)
                for g in groups
            ]))
        if dp >= max_dp:
            continue
        kept_idx.append(i)
        scores.append((er, dp))

    kept_entries = [entries[i] for i in kept_idx]
    kept_preds = preds_matrix[kept_idx] if kept_idx else preds_matrix[:0]
    score_arr = np.array(scores) if scores else np.empty((0, 2))
    return kept_entries, kept_preds, score_arr


def _stratified_trim(entries, preds_matrix, scores, target, accuracy_weight=0.4,
                    fairness_weight=0.4, diversity_weight=0.2):
    """
    Reduce pool to `target` size using a composite score that balances
    accuracy, fairness, and diversity. Selects greedily, picking the entry
    with the best composite at each step. Diversity is measured as average
    disagreement against already-selected entries.
    """
    if len(entries) <= target:
        return entries, preds_matrix

    n = len(entries)
    er = scores[:, 0]
    dp = scores[:, 1]

    # Normalise er and dp to [0, 1] within the pool (lower is better)
    er_norm = (er - er.min()) / (er.max() - er.min() + 1e-12)
    dp_norm = (dp - dp.min()) / (dp.max() - dp.min() + 1e-12)

    selected_idx = []
    selected_set = set()

    while len(selected_idx) < target:
        # Compute diversity score: avg disagreement against already-selected
        if selected_idx:
            sel_preds = preds_matrix[selected_idx]
            # disagreement[i] = avg fraction of samples where i differs from
            # the closest already-selected entry
            diversity = np.zeros(n)
            for i in range(n):
                if i in selected_set:
                    continue
                # min disagreement against any already-selected entry
                # (we want to keep entries that are different from all selected)
                d_to_sel = np.mean(
                    preds_matrix[i][None, :] != sel_preds, axis=1
                )
                diversity[i] = d_to_sel.min()
        else:
            diversity = np.ones(n)  # first pick gets max diversity

        # composite: lower er, lower dp, higher diversity all good
        # we're minimising, so flip diversity
        composite = (
            accuracy_weight * er_norm
            + fairness_weight * dp_norm
            - diversity_weight * diversity
        )
        # Mask already-selected
        for i in selected_set:
            composite[i] = np.inf
        next_idx = int(np.argmin(composite))
        selected_idx.append(next_idx)
        selected_set.add(next_idx)

    kept_entries = [entries[i] for i in selected_idx]
    kept_preds = preds_matrix[selected_idx]
    return kept_entries, kept_preds


def _stratified_trim_with_anchors(entries, preds_matrix, scores, target,
                                    k_acc=3, k_fair=3, k_balanced=5,
                                    accuracy_weight=0.3,
                                    fairness_weight=0.3,
                                    diversity_weight=0.4):
    """
    Reduce pool to `target` size with three anchor types:
      - top-k_acc by error rate
      - top-k_fair by DP gap
      - top-k_balanced by 50/50 ER+DP composite
    Then diversity-weighted greedy fill for remaining slots.
    """
    if len(entries) <= target:
        return entries, preds_matrix

    er = scores[:, 0]
    dp = scores[:, 1]
    n = len(entries)

    # Normalise once for the anchor composite
    er_norm = (er - er.min()) / (er.max() - er.min() + 1e-12)
    dp_norm = (dp - dp.min()) / (dp.max() - dp.min() + 1e-12)
    balanced_score = 0.5 * er_norm + 0.5 * dp_norm

    # Identify three sets of anchor indices
    anchor_idx = set()
    anchor_idx.update(np.argsort(er)[:k_acc].tolist())            # top-accuracy
    anchor_idx.update(np.argsort(dp)[:k_fair].tolist())           # top-fairness
    anchor_idx.update(np.argsort(balanced_score)[:k_balanced].tolist())  # top-balanced

    # Cap anchors at target size if over-allocated
    if len(anchor_idx) >= target:
        anchor_list = list(anchor_idx)
        # Among anchors, keep the ones with best composite under user weights
        comp_user = (
            accuracy_weight * er_norm[anchor_list]
            + fairness_weight * dp_norm[anchor_list]
        )
        keep_order = np.argsort(comp_user)[:target]
        keep = [anchor_list[i] for i in keep_order]
        return [entries[i] for i in keep], preds_matrix[keep]

    # Greedy diversity-weighted fill for remaining slots
    selected = list(anchor_idx)

    while len(selected) < target:
        sel_preds = preds_matrix[selected]
        diversity = np.zeros(n)
        for i in range(n):
            if i in selected:
                continue
            d = np.mean(preds_matrix[i][None, :] != sel_preds, axis=1)
            diversity[i] = d.min()

        composite = (
            accuracy_weight * er_norm
            + fairness_weight * dp_norm
            - diversity_weight * diversity
        )
        for i in selected:
            composite[i] = np.inf

        next_idx = int(np.argmin(composite))
        selected.append(next_idx)

    return [entries[i] for i in selected], preds_matrix[selected]

    

class Models():
    """Multiple different model learners are part of this class.

    Parameters
    ----------
    X_train: {array-like, sparse matrix}, shape (n_samples, m_features)
        Training data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_train: array-like, shape (n_samples)
        Label vector relative to the training data X_train.

    X_test: {array-like, sparse matrix}, shape (n_samples, m_features)
        Test data vector, where n_samples is the number of samples and
        m_features is the number of features.

    y_test: array-like, shape (n_samples)
        Label vector relative to the test data X_test.

    sens_attrs: list of strings
        List of the column names of sensitive attributes in the dataset.

    favored: tuple of float
        Tuple of the values of the favored group.

    ignore_sens: boolean
        Proxy is set to TRUE if the sensitive attribute should be ignored.
    """
    def __init__(self, X_train, X_test, y_train, y_test, sens_attrs, favored, ignore_sens=False, df_dict=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.sens_attrs = sens_attrs
        self.favored = favored
        self.ignore_sens = ignore_sens
        self.df_dict = df_dict
        self.metric = re.sub("bea_", "", self.df_dict["metric"])


    def decision_tree(self, sample_weight=None):
        """Fit the decision tree model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        dt_pred: list of predicted label for our testdata X_test

        "dectree": string of the used model name
        """
        clf = DecisionTreeClassifier()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            dt_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            dt_pred = clf.predict(self.X_test)

        return clf, dt_pred, "dectree"


    def linear_svm(self, sample_weight=None):
        """Fit the linear support vector machine model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        svm_pred: list of predicted label for our testdata X_test

        "linsvm": string of the used model name
        """
        clf = Pipeline([("scaler", StandardScaler()), ("linear_svc", LinearSVC(C=1, loss="hinge"))])
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train,
                    **{'linear_svc__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            svm_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, **{'linear_svc__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train, self.y_train)
            svm_pred = clf.predict(self.X_test)

        return clf, svm_pred, "linsvm"


    def nonlinear_svm(self, sample_weight=None):
        """Fit the nonlinear support vector machine model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        svm_pred: list of predicted label for our testdata X_test

        "nonlinsvm": string of the used model name
        """
        clf = Pipeline([("poly_features", PolynomialFeatures(degree=2)), \
            ("scaler", StandardScaler()), ("svm_clf", LinearSVC(C=10, loss="hinge"))])
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train,
                    **{'svm_clf__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            svm_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, **{'svm_clf__sample_weight': sample_weight})
            else:
                clf.fit(self.X_train, self.y_train)
            svm_pred = clf.predict(self.X_test)

        return clf, svm_pred, "nonlinsvm"


    def log_regr(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = LogisticRegression(solver='lbfgs')
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            reg_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            reg_pred = clf.predict(self.X_test)

        return clf, reg_pred, "logregr"


    def softmax_regr(self, sample_weight=None):
        """Fit the Softmax regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "softmaxregr": string of the used model name
        """
        clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            reg_pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            reg_pred = clf.predict(self.X_test)

        return clf, reg_pred, "softmaxregr"


    def adapted_adaboost(self):
        """Fit the AdaBoost models according to the given training data via the adapted
        AdaBoostClassifier.

        Parameters
        ----------
        modelsize: integer
            Amount of models that should be trained.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "adaboost": string of the used model name
        """
        abc = AdaBoostClassifierMult([LogisticRegression(),
            Pipeline([("scaler", StandardScaler()), ("clf", LinearSVC(C=1, loss="hinge"))]),
            DecisionTreeClassifier()], iterations=6)
        if self.ignore_sens:
            classifier_list = abc.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            classifier_list = abc.fit(self.X_train, self.y_train)
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "adapted_adaboost"

    def opt_learner(self, ensemble_strat, input_file, sbt):
        if ensemble_strat=="RandomForest":
            estimator = HyperOptimizedLearner(learner="RandomForest", search_method="full", input_file=input_file, sbt=sbt, cv=3, ensemble_entropy=1, sens_attrs=self.sens_attrs)
        elif ensemble_strat=="AdaBoost":
            estimator = HyperOptimizedLearner(learner="AdaBoost", search_method="full", input_file=input_file, sbt=sbt, cv=3, ensemble_entropy=1, sens_attrs=self.sens_attrs)
        elif ensemble_strat=="ExtraTrees":
            estimator = HyperOptimizedLearner(learner="ExtraTrees", search_method="full", input_file=input_file, sbt=sbt, cv=3, ensemble_entropy=1, sens_attrs=self.sens_attrs)
        elif ensemble_strat=="XGBoost":
            estimator = HyperOptimizedLearner(learner="XGBoost", search_method="full", input_file=input_file, sbt=sbt, cv=3, ensemble_entropy=1, sens_attrs=self.sens_attrs)
        
        if self.ignore_sens:
            estimator.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            estimator.fit(self.X_train, self.y_train)
        classifier_list = estimator.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, ensemble_strat


    def adaboost_classic(self):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """
        ab = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                criterion="gini", max_depth=1, splitter="random", max_features="sqrt", min_samples_leaf=5
                ),
            n_estimators=20)
        if self.ignore_sens:
            ab.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            ab.fit(self.X_train, self.y_train)
        classifier_list = ab.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "AdaBoostClassic", ab
    

    def extra_trees(self, large=False):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """
        if large:
            et = ExtraTreesClassifier(
                n_estimators=100,       # your ~20 weak classifiers
                criterion="gini",
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=3,    # slightly smaller leaf since data is low-dim
                max_features=0.5,      # ≈ 5 out of 10 features per split
                bootstrap=False,
                n_jobs=-1,
                random_state=0
            )
        else:
            et = ExtraTreesClassifier(
                n_estimators=20,       # your ~20 weak classifiers
                criterion="gini",
                max_depth=5,
                min_samples_split=2,
                min_samples_leaf=3,    # slightly smaller leaf since data is low-dim
                max_features=0.5,      # ≈ 5 out of 10 features per split
                bootstrap=True,
                n_jobs=-1,
                random_state=0
            )
        if self.ignore_sens:
            et.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            et.fit(self.X_train, self.y_train)
        classifier_list = et.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "ExtraTrees", et
    
    def ada2(self, large=False):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """
        if large:
            et = AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(
                        max_depth=2,            # key choice for AdaBoost
                        min_samples_leaf=3,
                        random_state=0
                    ),
                    n_estimators=100,
                    learning_rate=0.5,          # 0.1–1.0 typical
                    algorithm="SAMME.R",
                    random_state=0
                )
        if self.ignore_sens:
            et.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            et.fit(self.X_train, self.y_train)
        classifier_list = et.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "AdaBoost", et
    

    def boost_reg(self):
        """
        """
        br = BaggingRegressor(
            estimator=RandomParamTreeRegressor(random_state=42),
            n_estimators=30,
            max_samples=0.7,
            max_features=1.0,
            bootstrap=True,
            n_jobs=-1,
            random_state=42
        )

        br.fit(self.X_train, self.y_train)

        estimator_list = br.estimators_  # Liste von 30 fitted Wrapper-Estimators
        # Vorhersagen je Estimator: shape = (n_estimators, n_test_samples)
        estimator_predictions = np.vstack([est.predict(self.X_test) for est in estimator_list])

        return estimator_list, estimator_predictions, "BoostReg", br
    
    
    def rf2(self, large=False):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """
        if large:
            et = RandomForestClassifier(
                n_estimators=100,
                criterion="gini",
                max_depth=3,
                min_samples_split=2,
                min_samples_leaf=10,
                max_features=0.8,
                bootstrap=True,        # usually True for RF
                n_jobs=-1,
                random_state=0
            )
        if self.ignore_sens:
            et.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            et.fit(self.X_train, self.y_train)
        classifier_list = et.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "RandomForest", et
    
    def xgb2(self, large=False):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """
        if large:
            et = XGBClassifier(
                n_estimators=100,
                max_depth=4,               # 3–5 is comparable to your depth-5 ET
                learning_rate=0.1,         # lower = more stable
                subsample=0.8,             # row sampling for diversity/regularization
                colsample_bytree=0.5,      # similar spirit to max_features=0.5
                reg_lambda=1.0,            # L2
                reg_alpha=0.0,             # L1 (set >0 if you want more sparsity)
                min_child_weight=3,        # analogous-ish to min_samples_leaf
                gamma=0.0,                 # split penalty (try 0.1 if overfitting)
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
                random_state=0
            )
        if self.ignore_sens:
            et.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            et.fit(self.X_train, self.y_train)
        classifier_list = et.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "XGBoost", et


    def rf_classic(self, n_estimators=17, max_depth=2, criterion="gini"):
        """Fit the AdaBoost models according to the given training data via given parameters.

        Parameters
        ----------
        n_estimators: integer
            The number of trees in the forest.

        criterion: “gini”, “entropy”, “log_loss”}
            The split criterion of that the base models will use for training.

        max_depth: integer
            The maximum depth of the trees used as base models. 

        max_features: {“sqrt”, “log2”, None}
            The number of random features to consider when looking for the best split.

        splitter: {"best", "random"}
            The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "AdaBoostOpt": string of the used model name
        """


        rf = RandomForestClassifier(criterion=criterion, max_depth=max_depth, n_estimators=n_estimators)
        if self.ignore_sens:
            rf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
        else:
            rf.fit(self.X_train, self.y_train)
        classifier_list = rf.estimators_
        estimator_predictions = []
        for classifier in classifier_list:
            if self.ignore_sens:
                estimator_predictions.append(classifier.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))]))
            else:
                estimator_predictions.append(classifier.predict(self.X_test))
        return classifier_list, estimator_predictions, "RandomForestClassic", rf


    def lfr360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_LFR(self.df_dict, classifier, k=5, Ax=0.01, Ay=1, Az=2, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "LFR"

    def rew360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_Reweighing(self.df_dict, classifier, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "Reweighing"

    def dir360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_DisparateImpactRemover(self.df_dict, classifier, repair=1, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "DisparateImpactRemover"
    
    def gsr360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_GridSearchReduction(self.df_dict, classifier, self.metric, lam=0.5, remove=True, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "GridSearchReduction"

    def gfc360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_GerryFairClassifier(self.df_dict, gamma=0.01, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "GerryFairClassifier"

    def ceop360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_CalibratedEqOddsPostprocessing(self.df_dict, classifier, remove=True, training=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "CalibratedEqOddsPostprocessing"

    def eop360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_EqOddsPostprocessing(self.df_dict, classifier, remove=False, training=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "EqOddsPostprocessing"
    
    def roc360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_RejectOptionClassification(self.df_dict, classifier, self.metric, eps=0.05, remove=False, training=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "RejectOptionClassification"
    
    def ltdd360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = LTDD(self.df_dict, classifier, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "LTDD"
    
    def fagtb360(self):
        model = FAGTBClass(self.df_dict, estimators=300, learning_rate=0.01, lam=0.15, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FAGTB"
    
    def smote360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = FairSMOTE(self.df_dict, classifier, cr=0.8, f=0.8, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "Fair-SMOTE"
    
    def fax360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = FaX(self.df_dict)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FaX"
    
    def ssllx360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = FairSSL(self.df_dict, classifier, ssl_type="LabelPropagation", balancing=True, cr=0.8, f=0.8, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairSSL-Lx"
    
    def sslxt360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = FairSSL(self.df_dict, classifier, ssl_type="CoTraining", balancing=True, cr=0.8, f=0.8, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairSSL-xT"
    
    def fairrr360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = FairRR(self.df_dict, classifier, self.metric, level=0.1, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairRR"
    
    def jn360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = JiangNachum(self.df_dict, classifier, self.metric, estimators=100, learning_rate=1, remove=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "JiangNachum"
    
    def adafair360(self):
        classifier = DecisionTreeClassifier()
        model = AdaFairClass(self.df_dict, classifier, self.metric, estimators=50, learning_rate=1)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "AdaFair360"
    
    def fcm360(self):
        model = FairnessConstraintModelClass(self.df_dict, c=1e-3, tau=0.5, mu=1.2, eps=1e-4)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairnessConstraintModel"
    
    def hsic360(self):
        model = HSICLinearRegressionClass(self.df_dict, lam=0.1, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "HSICLinearRegression"
    
    def gferm360(self):
        model = GeneralFairERMClass(self.df_dict, eps=0, k=10, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "GeneralFairERM"
    
    def fairboost360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = FairBoost(self.df_dict, classifier, estimators=100, eps=0.1, k=25)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairBoost360"
    
    def fairret360(self):
        model = fairret(self.df_dict, self.metric, lam=1, lr=0.001, h_layer=32)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "fairret"
    
    def fesf360(self):
        model = FESFClass(self.df_dict, K=200)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FESF"
    
    def gc360(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = GradualCompatibility(self.df_dict, reg=0, reg_val=1, weights_init="None", lam=0.0001)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "GradualCompatibility"


    def gsr(self, gridsize=10):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_GridSearchReduction(self.df_dict, "LogisticRegression", self.metric, grid_size=gridsize, lam=0.5, remove=False, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())

        classifier_list = model.model.model.model_.predictors_
        estimator_predictions = []
        for classifier in classifier_list:
            estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "GSR", model
    
    
    def egr(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_ExponentiatedGradientReduction(self.df_dict, "LogisticRegression", self.metric,  eps=0.01, learning_rate=2, nu="None", remove=False, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())

        classifier_list = model.model.model.model_.predictors_
        estimator_predictions = []
        for classifier in classifier_list:
            estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "EGR", model
    

    def egr2(self):
        classifier = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=100)
        model = AIF_ExponentiatedGradientReduction(self.df_dict, classifier, self.metric,  eps=0.01, learning_rate=2, remove=True, reg=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())

        classifier_list = model.model.model.model_.predictors_
        estimator_predictions = []
        for classifier in classifier_list:
            estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "EGR2", model
    

    def egr_reg(self):
        model = Agarwal(self.df_dict, c=0.5, penalty="BGL", sensitive_predictor=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())

        classifier_list = model.model._model.predictors_
        estimator_predictions = []
        for classifier in classifier_list:
            estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "EGR_reg", model
    
    
    def reduction_reg(self):
        model = Agarwal(self.df_dict, c=0.5, penalty="BGL", sensitive_predictor=False)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "ReductionApproach"
    
    def unaware_reg(self):
        model = UnawareFairReg(self.df_dict, base="SGD3", L="auto", eps=8)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "UnawareFairReg"
    
    def heckman_reg(self):
        model = FairSampling(self.df_dict, num_epochs=100, lr=0.1)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairHeckman"
    
    def adv_deb_reg(self):
        model = AdversarialDebiasing_Reg(self.df_dict, batch=64, learning_rate=0.001, mu=0.7, epochs=200, model_type="deep_model", reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "AdversarialDebiasing_reg"
    
    def hgr_reg(self):
        model = HGR(self.df_dict, batch=256, learning_rate=0.001, mu=0.7, epochs=50, model_type="deep_model", reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "HGR_reg"
    
    def fd_reg(self):
        model = FairDummies(self.df_dict, batch=10000, learning_rate=0.01, mu=0.8, second_scale=1, epochs=50, model_type="deep_model", reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairDummies_reg"

    def fglm_reg(self):
        model = FairGeneralizedLinearModelClass(self.df_dict, lam=0.01, discretization="equal_count", reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "FairGeneralizedLinearModel_reg"

    def gferm_reg(self):
        model = GeneralFairERMClass(self.df_dict, eps=50, k=10, reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "GeneralFairERM_reg"
    
    def hsic_reg(self):
        model = HSICLinearRegressionClass(self.df_dict, lam=0.1, reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "HSICLinearRegression_reg"
    
    def cfm_reg(self):
        model = ConvexFrameworkModelClass(self.df_dict, lam=1e-3, penalty="group", reg=True)
        model.fit(self.X_train.copy(), self.y_train.copy())
        prediction = model.predict(self.X_test.copy())
        return model, prediction, "ConvexFrameworkModel_reg"


    def linear_regression(self, sample_weight=None):
        """
        """
        clf = LinearRegression()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        pred_final = []
        for i in pred:
            pred_final.append(i[0])

        return clf, pred_final, "LinearRegression"


    def sgd_regressor(self, sample_weight=None):
        """
        """
        clf = SGDRegressor(max_iter=1000, tol=1e-3)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "SGDRegressor"


    def ridge(self, sample_weight=None):
        """
        """
        clf = Ridge(alpha=1.0)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        pred_final = []
        for i in pred:
            pred_final.append(i[0])

        return clf, pred_final, "Ridge"


    def elasticnet(self, sample_weight=None):
        """
        """
        clf = ElasticNet(alpha=1.0, l1_ratio=0.5)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "ElasticNet"


    def lasso(self, sample_weight=None):
        """
        """
        clf = Lasso(alpha=0.1)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "Lasso"


    def lars(self, sample_weight=None):
        """
        """
        clf = Lars()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "Lars"


    def bayesian_ridge(self, sample_weight=None):
        """
        """
        clf = BayesianRidge()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "BayesianRidge"


    def huber_regressor(self, sample_weight=None):
        """
        """
        clf = HuberRegressor()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf = clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "HuberRegressor"


    def adafair(self):
        """Fit the AdaBoost models according to the given training data via the adapted
        AdaBoostClassifier.

        Parameters
        ----------
        modelsize: integer
            Amount of models that should be trained.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "adaboost": string of the used model name
        """
        sens_idx = self.X_train.columns.get_loc(self.sens_attrs[0])

        if len(self.sens_attrs) > 1:
            af = AdaFairSP(base_estimator=LogisticRegression(), n_estimators=6, learning_rate=1., saIndex=sens_idx, saValue=self.favored)
        else:
            af = AdaFairSP(base_estimator=LogisticRegression(), n_estimators=10, learning_rate=1., saIndex=sens_idx, saValue=self.favored)
        af.fit(self.X_train, self.y_train)
        classifier_list = af.trained_clfs
        estimator_predictions = []
        for classifier in classifier_list:
            estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "AdaFair", af


    def fairboost(self):
        """Fit the AdaBoost models according to the given training data via the adapted
        AdaBoostClassifier.

        Parameters
        ----------
        modelsize: integer
            Amount of models that should be trained.


        Returns
        -------
        classifier_list: Trained AdaBoost classifier list

        estimator_predictions: list of list of predicted label for each estimator testdata X_test

        "adaboost": string of the used model name
        """
        df_dict = dict()
        df_dict["sens_attrs"] = self.sens_attrs
        df_dict["favored"] = self.favored
        if len(self.sens_attrs) > 1:
            fb = FairBoost(df_dict, LogisticRegression(), 6, 0.001, 35)
        else:
            fb = FairBoost(df_dict, LogisticRegression(), 15, 0.001, 25)
        fb.fit(self.X_train, self.y_train)
        classifier_list = fb.trained_estimator
        estimator_predictions = []
        for classifier in classifier_list:
            estimator_predictions.append(classifier.predict(self.X_test))

        return classifier_list, estimator_predictions, "FairBoost", fb


    def log_regr_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.
        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "LogisticRegression_mult"


    def svm_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = SVC()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "SVM_mult"


    def random_forest_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = RandomForestClassifier()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "RandomForest_mult"


    def dectree_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = DecisionTreeClassifier()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "DecisionTree_mult"


    def nn_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = MLPClassifier(hidden_layer_sizes=(100,))
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "NN_mult"


    def naive_bayes_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = GaussianNB()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "NaiveBayes_mult"


    def kNN_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = KNeighborsClassifier(n_neighbors=3)
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "kNN_mult"


    def xgboost_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = XGBClassifier()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "XGBoost_mult"


    def lightgbm_mult(self, sample_weight=None):
        """Fit the logistic regression model according to the given training data.

        Parameters
        ----------
        sample_weight: array of float
            Weight of each sample of the training dataset


        Returns
        -------
        clf: Trained classifier

        reg_pred: list of predicted label for our testdata X_test

        "logregr": string of the used model name
        """
        clf = lgb.LGBMClassifier()
        if self.ignore_sens:
            if sample_weight is not None:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train, sample_weight)
            else:
                clf.fit(self.X_train[list(set(self.X_train.columns)-set(self.sens_attrs))], self.y_train)
            pred = clf.predict(self.X_test[list(set(self.X_test.columns)-set(self.sens_attrs))])
        else:
            if sample_weight is not None:
                clf.fit(self.X_train, self.y_train, sample_weight)
            else:
                clf.fit(self.X_train, self.y_train)
            pred = clf.predict(self.X_test)

        return clf, pred, "LightGBM_mult"



    def diverse_ensemble(self,
                        target_pool_size=150,
                        dedup_threshold=0.02,
                        accuracy_weight=0.4,
                        fairness_weight=0.4,
                        diversity_weight=0.2,
                        include_egr=False,
                        include_gsr=False,
                        include_adafair=False,
                        include_rf_trees=True,
                        include_adaboost=True,
                        include_cost_sensitive=True,
                        include_subgroup=False,
                        ):
        """
        Train a diverse, fairness-aware ensemble for FairBALC's candidate pool.
        Currently GSR, EGR, AdaFair are not used within the paper, as we only train
        non-fairness-related classifiers!

        Drop-in replacement for self.egr() and self.adaboost_classic() with
        matching return shape: (classifier_list, estimator_predictions, name, model).

        Args:
            target_pool_size: max number of predictors retained after refinement.
            dedup_threshold: predictors with validation-prediction disagreement
                below this fraction are treated as duplicates.
            accuracy_weight, fairness_weight, diversity_weight: weights in the
                composite score used by stratified_trim. Should sum to 1.0.
            include_*: enable/disable individual pool sources for ablations.

        Returns:
            classifier_list: list of fitted predictor objects (compatible with
                existing code that iterates over self.estimators_)
            estimator_predictions: list of 1-D arrays of test predictions, one
                per predictor in classifier_list
            name: string identifier ("DiverseEnsemble")
            model: PoolModel container exposing .predict() for compatibility
        """
        # Pull from self (assumes the parent class has these attributes,
        # matching the egr/adaboost_classic method signatures)
        X_tr = copy.deepcopy(self.X_train)
        y_tr = copy.deepcopy(self.y_train)
        X_val = copy.deepcopy(self.X_test)
        y_val = copy.deepcopy(self.y_test)
        df_dict = self.df_dict
        sens_attr = df_dict["sens_attrs"][0]
        sensitive_train = X_tr[sens_attr].values
        sensitive_test = X_val[sens_attr].values

        # Build a held-out validation set from training data for refinement.
        # If you already have a separate val set, pass it through self instead.
        
        #X_tr, X_val, y_tr, y_val = train_test_split(
         #   X_train, y_train, test_size=0.25, random_state=42, stratify=y_train,
        #)
        sens_val = X_val[sens_attr].values
        sens_tr = X_tr[sens_attr].values

        # Base classifier factories
        base_factories = _make_base_factories(seed=42)

        # Constraint set for EGR/GSR
        constraints = [
            ("DP", "demographic_parity"),
            ("EOD", "equalized_odds"),
        ]

        # Build raw pool from each enabled source
        entries: List[_PredictorEntry] = []

        if include_egr:
            entries.extend(_source_egr(
                X_tr, y_tr, sens_tr, base_factories,
                eps_grid=[0.01, 0.05, 0.1], constraints=constraints,
                df_dict=df_dict,
            ))

        if include_gsr:
            entries.extend(_source_gsr(
                X_tr, y_tr, sens_tr, base_factories,
                constraints=constraints, df_dict=df_dict, grid_size=15,
            ))

        if include_adafair:
            entries.extend(_source_adafair(
                X_tr, y_tr, sens_tr, base_factories,
                df_dict=df_dict, metric=self.metric
            ))

        if include_rf_trees:
            entries.extend(_source_rf_trees(X_tr, y_tr, n_estimators=100, seed=42))

        if include_adaboost:
            entries.extend(_source_adaboost(X_tr, y_tr, n_estimators=20, seed=42))

        if include_cost_sensitive:
            entries.extend(_source_cost_sensitive(
                X_tr, y_tr.copy(), sens_tr, base_factories,
            ))

        if include_subgroup:
            entries.extend(_source_subgroup(
                X_tr, y_tr, sens_tr, base_factories, min_group_size=30,
            ))

        print(f"  Raw pool size: {len(entries)}")

        # Compute validation predictions, drop entries that fail
        entries, val_preds = _compute_predictions(entries, X_val)
        print(f"  After predict-success filter: {len(entries)}")

        # Deduplicate
        entries, val_preds = _dedup_by_predictions(
            entries, val_preds, threshold=dedup_threshold,
        )
        print(f"  After dedup (threshold={dedup_threshold}): {len(entries)}")

        # Quality filter
        entries, val_preds, scores = _quality_filter(
            entries, val_preds, y_val, sens_val,
            max_er=0.5, max_dp=0.5,
        )
        print(f"  After quality filter: {len(entries)}")

        # Stratified trim to target size
        if len(entries) > target_pool_size:
            entries, val_preds = _stratified_trim_with_anchors(
                entries, val_preds, scores, target_pool_size,
                accuracy_weight=accuracy_weight,
                fairness_weight=fairness_weight,
                diversity_weight=diversity_weight,
            )
            print(f"  After stratified trim: {len(entries)}")

        # Compute test predictions for return value
        classifier_list = [e.classifier for e in entries]
        estimator_predictions = []
        for entry in entries:
            try:
                estimator_predictions.append(np.asarray(entry.predict(X_val)).astype(int))
            except Exception as e:
                warnings.warn(f"Test prediction failed for {entry.source}: {e}")
                estimator_predictions.append(np.zeros(len(X_val), dtype=int))

        model = PoolModel(entries)
        return classifier_list, estimator_predictions, "DiversePool", model
