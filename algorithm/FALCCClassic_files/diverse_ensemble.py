"""
Diverse ensemble training for FairBALC's candidate pool.

Builds a heterogeneous pool of predictors combining:
  - EGR + GSR with diverse base classifiers and fairness constraints
  - Random Forest tree extraction
  - Cost-sensitive variants emphasising different (group, label) cells
  - Subgroup-specific classifiers
  - AdaBoost iterates from a stump-tree classifier

The pool is then deduplicated by validation predictions, filtered to remove
degenerate predictors, and stratified-trimmed to a target size while
preserving source diversity.

Shape of the return value matches existing model_training functions:
  (classifier_list, estimator_predictions, name, model)
where `model` is a PoolModel container exposing .predict() for downstream
code that expects a single-model interface.
"""

import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from algorithm import (
    AIF_GridSearchReduction,
    AIF_ExponentiatedGradientReduction
)



# Optional: XGBoost is the strongest tabular learner but may not be installed
try:
    
    _HAVE_XGB = True
except ImportError:
    _HAVE_XGB = False


# ---------------------------------------------------------------------------
# A lightweight predictor wrapper that decouples each pool entry from its
# original feature space (useful when subgroup-specific or random-subspace
# classifiers were fit on a feature subset)
# ---------------------------------------------------------------------------

class _PredictorEntry:
    """
    Wraps a fitted classifier with the metadata needed to apply it to new data.

    Some classifiers in the pool were fit on a feature subset (subgroup-specific,
    random subspace) or on transformed inputs. This wrapper records the
    feature columns the classifier expects, so .predict() at inference time
    can slice the input correctly.
    """

    def __init__(self, classifier, feature_cols=None, source: Optional[str] = None,
                 meta: Optional[Dict[str, Any]] = None):
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


# ---------------------------------------------------------------------------
# Container that mimics a single-model interface for compatibility with
# downstream code in fairlet*.py
# ---------------------------------------------------------------------------

class PoolModel:
    """
    Container for a heterogeneous predictor pool. Exposes .predict() that
    delegates to the first predictor in the pool (used only when downstream
    code calls model.predict() directly; FairBALC's per-cluster selection
    bypasses this and uses the pool directly).
    """

    def __init__(self, entries: List[_PredictorEntry]):
        self.entries = entries

    def predict(self, X):
        # Default: predict with the first entry. Most callers should use
        # the entry list directly instead of calling .predict() here.
        if not self.entries:
            raise ValueError("PoolModel is empty")
        return self.entries[0].predict(X)

    def __len__(self):
        return len(self.entries)


# ---------------------------------------------------------------------------
# Base classifier factories
# ---------------------------------------------------------------------------

def _make_base_factories(seed: int = 42) -> Dict[str, Callable]:
    """Return a dict of name -> factory functions producing fresh classifier instances."""
    factories = {
        "LR": lambda: LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear", max_iter=200, random_state=seed,
        ),
        "RF": lambda: RandomForestClassifier(
            n_estimators=100, max_features="sqrt", n_jobs=-1, random_state=seed,
        ),
        "GBM": lambda: GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=seed,
        ),
    }
    if _HAVE_XGB:
        factories["XGB"] = lambda: XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", verbosity=0,
            n_jobs=-1, random_state=seed,
        )
    return factories


# ---------------------------------------------------------------------------
# Pool sources
# ---------------------------------------------------------------------------

def _source_egr(X_train, y_train, sensitive_train, base_factories,
                eps_grid, constraints, df_dict) -> List[_PredictorEntry]:
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
                constraints, df_dict, grid_size=15) -> List[_PredictorEntry]:
    """GSR sweep across (base_classifier, constraint)."""
    entries = []
    for base_name, factory in base_factories.items():
        for cname, metric_str in constraints:
            for eps in eps_grid:
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
                     seed=42) -> List[_PredictorEntry]:
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
                     seed=42) -> List[_PredictorEntry]:
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
                           base_factories) -> List[_PredictorEntry]:
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

    weight_schemes = [
        ("uniform", np.ones(n)),
        ("minority_class",
            np.where(y_train == 1, 1.0 / pos_rate, 1.0 / neg_rate)),
        ("minority_group",
            np.where(sensitive_train == 1,
                     1.0 / priv_rate, 1.0 / unpriv_rate)),
        ("unpriv_positive",
            np.where((sensitive_train == 0) & (y_train == 1), 4.0, 1.0)),
        ("priv_negative",
            np.where((sensitive_train == 1) & (y_train == 0), 4.0, 1.0)),
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
                    meta={"base": base_name, "scheme": scheme_name},
                ))
            except (TypeError, ValueError):
                # Some classifiers don't accept sample_weight
                pass
    return entries


def _source_subgroup(X_train, y_train, sensitive_train,
                     base_factories, min_group_size=30) -> List[_PredictorEntry]:
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


# ---------------------------------------------------------------------------
# Pool refinement: dedup, quality filter, stratified trim
# ---------------------------------------------------------------------------

def _compute_predictions(entries: List[_PredictorEntry],
                         X_val) -> Tuple[List[_PredictorEntry], np.ndarray]:
    """Compute predictions for each entry on the validation set; drop entries that error out."""
    kept = []
    preds_list = []
    for entry in entries:
        try:
            preds = np.asarray(entry.predict(X_val)).astype(int)
            kept.append(entry)
            preds_list.append(preds)
        except Exception as e:
            warnings.warn(f"Prediction failed for {entry.source}: {e}")
    return kept, np.array(preds_list) if preds_list else np.empty((0, len(X_val)))


def _dedup_by_predictions(entries: List[_PredictorEntry], preds_matrix: np.ndarray,
                          threshold: float = 0.02) -> Tuple[List[_PredictorEntry], np.ndarray]:
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


def _quality_filter(entries: List[_PredictorEntry], preds_matrix: np.ndarray,
                    y_val, sensitive_val,
                    max_er: float = 0.5, max_dp: float = 0.5
                    ) -> Tuple[List[_PredictorEntry], np.ndarray, np.ndarray]:
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


def _stratified_trim(entries: List[_PredictorEntry], preds_matrix: np.ndarray,
                     scores: np.ndarray, target: int,
                     accuracy_weight: float = 0.4,
                     fairness_weight: float = 0.4,
                     diversity_weight: float = 0.2,
                     ) -> Tuple[List[_PredictorEntry], np.ndarray]:
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


# ---------------------------------------------------------------------------
# Top-level training function
# ---------------------------------------------------------------------------

def diverse_ensemble(self,
                     target_pool_size: int = 150,
                     dedup_threshold: float = 0.02,
                     accuracy_weight: float = 0.4,
                     fairness_weight: float = 0.4,
                     diversity_weight: float = 0.2,
                     include_egr: bool = True,
                     include_gsr: bool = True,
                     include_rf_trees: bool = True,
                     include_adaboost: bool = True,
                     include_cost_sensitive: bool = True,
                     include_subgroup: bool = True,
                     ) -> Tuple[List[Any], List[np.ndarray], str, PoolModel]:
    """
    Train a diverse, fairness-aware ensemble for FairBALC's candidate pool.

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
    X_train = self.X_train
    y_train = self.y_train
    X_test = self.X_test
    df_dict = self.df_dict
    sens_attr = df_dict["sens_attrs"][0]
    sensitive_train = X_train[sens_attr].values
    sensitive_test = X_test[sens_attr].values

    # Build a held-out validation set from training data for refinement.
    # If you already have a separate val set, pass it through self instead.
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train,
    )
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

    if include_rf_trees:
        entries.extend(_source_rf_trees(X_tr, y_tr, n_estimators=100, seed=42))

    if include_adaboost:
        entries.extend(_source_adaboost(X_tr, y_tr, n_estimators=20, seed=42))

    if include_cost_sensitive:
        entries.extend(_source_cost_sensitive(
            X_tr, y_tr, sens_tr, base_factories,
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
        entries, val_preds = _stratified_trim(
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
            estimator_predictions.append(np.asarray(entry.predict(X_test)).astype(int))
        except Exception as e:
            warnings.warn(f"Test prediction failed for {entry.source}: {e}")
            estimator_predictions.append(np.zeros(len(X_test), dtype=int))

    model = PoolModel(entries)
    return classifier_list, estimator_predictions, "DiversePool", model
