"""Baseline modelling: preprocessing, cross-validation and feature importance.

The framework fits real models rather than only recommending them. A recommendation
that is never tested is an opinion; a cross-validated score is evidence, and it is
what lets the final report state what the data can actually support.

Three choices here are deliberate:

* Every model is fitted inside a Pipeline so that imputation and scaling are
  learned on each training fold only. Fitting them on the whole dataset first
  would leak information from the validation fold and inflate every score.
* Classification is scored on macro F1 and ROC AUC as well as accuracy, because
  accuracy alone is misleading on imbalanced targets.
* A trivial baseline is always included. A model is only worth reporting if it
  beats predicting the majority class or the mean.
"""
from __future__ import annotations

import warnings
from typing import Any

from datetime import datetime

import numpy as np
import sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor,
                              HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              StackingClassifier, StackingRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import get_scorer
from sklearn.model_selection import (KFold, RandomizedSearchCV, StratifiedKFold,
                                     cross_val_predict, cross_validate,
                                     permutation_test_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import CONFIG

# scikit-learn warns about convergence and undefined metrics on small folds;
# these are reported through the results rather than printed as noise.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MAX_CATEGORIES = 30  # above this a categorical column is dropped rather than encoded


def _split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Separate features into numeric, categorical and unusable."""
    numeric, categorical, dropped = [], [], []
    for col in features:
        if col not in df.columns:
            dropped.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        elif df[col].nunique(dropna=True) <= MAX_CATEGORIES:
            categorical.append(col)
        else:
            dropped.append(col)
    return numeric, categorical, dropped


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Impute and encode inside the pipeline so nothing leaks across folds."""
    steps = []
    if numeric:
        steps.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        steps.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", max_categories=MAX_CATEGORIES)),
        ]), categorical))
    return ColumnTransformer(steps, remainder="drop")


def _operating_behaviour(pipe, X, y, splitter, task: str) -> dict[str, Any] | None:
    """How the chosen model behaves on cases it has not seen.

    A score answers "how good is this model" on a scale a reader has no
    intuition for. `macro F1 = 0.7374` does not tell a clinician how many
    patients with the condition this finds, nor how many healthy ones it
    wrongly flags -- and those two numbers are what any decision actually
    turns on. They also trade off against each other, so one summary figure
    cannot express both.

    Predictions come from `cross_val_predict` on the same folds as the scores,
    so every prediction is made by a model that did not see that row. The
    counts are therefore out-of-sample, which is the only kind worth quoting.

    One extra pass over the folds for the winning model only. Cheap beside the
    permutation test, which refits the same model fifty times.
    """
    try:
        predicted = cross_val_predict(pipe, X, y, cv=splitter, n_jobs=1)
    except Exception:
        # Behaviour detail is a bonus; failing to produce it must never fail
        # the modelling stage, which has already produced its scores.
        return None

    if task == "classification":
        classes = sorted(pd.Series(y).unique().tolist(), key=str)
        if len(classes) > 8:
            # A 9x9 breakdown helps nobody read a report.
            return None
        per_class = []
        for label in classes:
            actual = pd.Series(y).reset_index(drop=True) == label
            guessed = pd.Series(predicted).reset_index(drop=True) == label
            support = int(actual.sum())
            found = int((actual & guessed).sum())
            false_alarms = int((~actual & guessed).sum())
            others = int((~actual).sum())
            per_class.append({
                "class": str(label),
                "actual_count": support,
                "correctly_identified": found,
                # Of every 100 cases that really are this class, how many the
                # model finds. Recall, in the words a reader can act on.
                "found_per_100": (round(100 * found / support, 1)
                                  if support else None),
                "false_alarms": false_alarms,
                # Of every 100 cases that are NOT this class, how many get
                # wrongly labelled as it.
                "false_alarms_per_100_others": (
                    round(100 * false_alarms / others, 1) if others else None),
            })
        return {"kind": "classification", "n_predictions": int(len(predicted)),
                "per_class": per_class}

    residuals = np.abs(np.asarray(y, dtype=float)
                       - np.asarray(predicted, dtype=float))
    spread = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
    return {
        "kind": "regression",
        "n_predictions": int(len(predicted)),
        "median_absolute_error": round(float(np.median(residuals)), 4),
        "within_error_per_100": 50,  # the median, stated as a proportion
        "error_90th_percentile": round(float(np.percentile(residuals, 90)), 4),
        # The error means nothing without the range it sits in: being out by 5
        # is excellent for house prices and useless for shoe sizes.
        "central_spread_of_target": round(spread, 4),
        "error_as_share_of_spread": (round(float(np.median(residuals)) / spread, 4)
                                     if spread else None),
    }


def _scores_from_cv(cv: dict, scoring: dict) -> dict[str, Any]:
    """Cross-validation output as reportable scores.

    scikit-learn maximises everything, so error metrics arrive negated:
    `neg_root_mean_squared_error` is -12.3 where the quantity a reader wants is
    12.3. This strips the prefix and restores the sign.

    It exists as a function because it was previously written out twice -- once
    in the candidate loop and once in the tuning block -- and only the first
    copy did the conversion. A tuned regression winner therefore carried
    `neg_rmse: -12.3` into a comparison table whose every other row carried
    `rmse: 12.3`, and the report printed a negative error.
    """
    scores: dict[str, Any] = {}
    for metric in scoring:
        values = cv[f"test_{metric}"]
        key = metric.replace("neg_", "")
        mean = float(np.mean(values))
        scores[key] = round(-mean if metric.startswith("neg_") else mean, 4)
        scores[f"{key}_std"] = round(float(np.std(values)), 4)
    return scores


def _imbalance_ratio(y: Any) -> float:
    """Majority class count divided by minority class count."""
    counts = pd.Series(y).value_counts()
    if len(counts) < 2 or counts.min() == 0:
        return 1.0
    return float(counts.max() / counts.min())


def _candidate_models(task: str, n_rows: int, n_classes: int = 2,
                      imbalance: float = 1.0) -> dict[str, Any]:
    """Baseline models spanning interpretability and flexibility.

    The set is deliberately small and untuned. The purpose is to establish what
    performance the data supports and which model family suits it, not to win a
    benchmark, and untuned baselines are the honest way to report that.

    On a skewed target the set gains class-weighted variants of the families
    that support them. `class_weight="balanced"` was previously searched only
    during tuning, which is optional and is skipped entirely above
    `tuning_max_rows` -- so on a large imbalanced dataset nothing compensated
    for the skew at all. The EDA stage reported the imbalance and this stage
    ignored it.

    They are added rather than substituted, because weighting is a trade and
    not an improvement. Measured on a 10:1 target, balancing a logistic
    regression moved macro F1 from 0.6026 to 0.5808 while moving recall from
    0.5717 to 0.7307: it finds far more of the minority class and is wrong more
    often when it calls one. Which of those a reader wants depends on what a
    miss costs against what a false alarm costs, and that is not in the data.

    So both are fitted and both appear in the comparison, and the framework
    does what it does everywhere else it cannot know the answer -- it measures
    instead of assuming, and shows the reader the choice rather than making it
    for them.
    """
    balanced = (task == "classification"
                and imbalance > CONFIG.imbalance_threshold)

    if task == "classification":
        models = {
            # Two baselines, because a single one is misleading depending on the
            # metric. The majority baseline maximises accuracy on an imbalanced
            # target but scores very poorly on macro F1, since it never predicts
            # the minority class. The stratified baseline predicts classes at
            # their observed rates, which is the correct chance-level reference
            # for macro F1. Comparing against the stronger of the two prevents a
            # model that has learned nothing from appearing to beat chance.
            "Majority baseline": DummyClassifier(strategy="most_frequent"),
            "Random baseline": DummyClassifier(strategy="stratified", random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1),
        }
        if balanced:
            # Only the families where weighting is both supported and cheap.
            # Boosting variants are left out: they are the slowest fits here and
            # doubling them would cost more than the comparison is worth.
            models["Logistic Regression (balanced)"] = LogisticRegression(
                max_iter=1000, random_state=42, class_weight="balanced")
            models["Random Forest (balanced)"] = RandomForestClassifier(
                n_estimators=200, random_state=42, n_jobs=-1,
                class_weight="balanced")

        if n_rows <= 50_000:
            models["Gradient Boosting"] = GradientBoostingClassifier(random_state=42)
            # Histogram-based boosting bins continuous features before splitting,
            # which makes it far faster than the exact variant and often more
            # accurate on datasets with many distinct numeric values.
            models["Hist Gradient Boosting"] = HistGradientBoostingClassifier(
                random_state=42)
            # Extremely randomised trees split at random thresholds rather than
            # optimal ones. Individually weaker, the extra variance between trees
            # makes their average less correlated and sometimes better.
            models["Extra Trees"] = ExtraTreesClassifier(
                n_estimators=300, random_state=42, n_jobs=-1)

        # A stacked ensemble learns how much to trust each family from held-out
        # predictions, rather than picking one outright. It is the only candidate
        # that can exceed every member, and measured on this project's datasets it
        # usually does -- but it fits every base model several times over, so it is
        # offered only where that cost is affordable.
        if n_rows <= CONFIG.stacking_max_rows:
            models["Stacked Ensemble"] = StackingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(
                        n_estimators=200, random_state=42, n_jobs=-1)),
                    ("hgb", HistGradientBoostingClassifier(random_state=42)),
                    ("lr", LogisticRegression(max_iter=1000, random_state=42)),
                ],
                final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                cv=3, n_jobs=1)
        return models

    models = {
        "Mean baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=42),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42, n_jobs=-1),
    }
    if n_rows <= 50_000:
        models["Gradient Boosting"] = GradientBoostingRegressor(random_state=42)
        models["Hist Gradient Boosting"] = HistGradientBoostingRegressor(
            random_state=42)
        models["Extra Trees"] = ExtraTreesRegressor(
            n_estimators=300, random_state=42, n_jobs=-1)

    if n_rows <= CONFIG.stacking_max_rows:
        models["Stacked Ensemble"] = StackingRegressor(
            estimators=[
                ("rf", RandomForestRegressor(
                    n_estimators=200, random_state=42, n_jobs=-1)),
                ("hgb", HistGradientBoostingRegressor(random_state=42)),
                ("ridge", Ridge(random_state=42)),
            ],
            final_estimator=Ridge(random_state=42), cv=3, n_jobs=1)
    return models


def _prepare(df: pd.DataFrame, target: str, features: list[str],
             task: str) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]] | tuple[None, None, dict]:
    """Assemble the modelling matrix and report what had to be excluded."""
    if target not in df.columns:
        return None, None, {"error": f"Target column '{target}' is not in the dataset."}

    features = [f for f in features if f != target]
    if not features:
        return None, None, {"error": "No usable feature columns were supplied."}

    sub = df[features + [target]].copy()

    # Rows without a target cannot train or validate a model.
    n_before = len(sub)
    sub = sub[sub[target].notna()]
    dropped_rows = n_before - len(sub)

    if len(sub) < 30:
        return None, None, {
            "error": f"Only {len(sub)} rows have a target value; too few to model reliably."
        }

    y = sub[target]
    if task == "classification":
        counts = y.value_counts()
        # A class with a single member cannot appear in both a train and a test
        # fold, so stratified cross-validation would fail on it.
        rare = counts[counts < 3]
        if len(rare) > 0:
            sub = sub[~y.isin(rare.index)]
            y = sub[target]
        if y.nunique() < 2:
            return None, None, {"error": "The target has fewer than two classes after cleaning."}

    numeric, categorical, dropped_cols = _split_feature_types(sub, features)
    if not numeric and not categorical:
        return None, None, {"error": "No usable numeric or categorical features remain."}

    meta = {
        "n_rows": int(len(sub)),
        "rows_dropped_missing_target": int(dropped_rows),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_features": dropped_cols,
    }
    return sub[numeric + categorical], y, meta


def _select_best(successful: list[dict[str, Any]], primary: str,
                 task: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the reported model, breaking statistical ties on a second metric.

    Taking the top of the sorted list treats every difference as real. It is
    not: a model leading by less than its own fold-to-fold standard deviation
    has not been shown to be better, and selecting on that difference is
    selecting on noise.

    Observed on this project's wine-quality data, where the leader won macro F1
    by 0.0138 with a standard deviation of 0.0314 -- indistinguishable from four
    other models -- while scoring 5.7 accuracy points *worse* than one of them.
    Reporting it made the framework look weaker than the data warranted, for no
    gain in the metric it was chosen on.

    So models within one standard deviation of the leader are treated as tied,
    and the tie is broken on a secondary criterion: accuracy for classification,
    since it is what a reader expects and the primary metric is unaffected; and
    for regression, the lower variance across folds, since among equally
    accurate models the more stable one generalises more predictably.

    The primary metric still decides who is in contention. This only chooses
    among models the primary metric cannot separate.
    """
    contenders = [r for r in successful if not r["is_baseline"]]
    if not contenders:
        return successful[0], {"rule": "no non-baseline model was available"}

    leader = contenders[0]
    tolerance = float(leader["scores"].get(f"{primary}_std", 0.0) or 0.0)

    tied = [r for r in contenders
            if leader["primary_score"] - r["primary_score"] <= tolerance]
    if len(tied) <= 1:
        return leader, {
            "rule": "highest primary score",
            "primary_metric": primary,
            "tied_within_1_std": [leader["model"]],
        }

    if task == "classification":
        secondary = "accuracy"
        chosen = max(tied, key=lambda r: r["scores"].get(secondary, 0.0))
        criterion = "higher accuracy among models tied on the primary metric"
    else:
        secondary = f"{primary}_std"
        chosen = min(tied, key=lambda r: r["scores"].get(secondary, float("inf")))
        criterion = "lower fold-to-fold variance among models tied on the primary metric"

    return chosen, {
        "rule": criterion,
        "primary_metric": primary,
        "tie_tolerance": round(tolerance, 4),
        "tied_within_1_std": [r["model"] for r in tied],
        "leader_on_primary": leader["model"],
        "selected": chosen["model"],
        "changed_selection": bool(chosen["model"] != leader["model"]),
    }


def evaluate_models(df: pd.DataFrame, target: str, features: list[str],
                    task: str, cv_folds: int = 5,
                    random_state: int = 42) -> dict[str, Any]:
    """Cross-validate baseline models and return comparable scores.

    Every model sees identical folds, so differences between them reflect the
    model rather than the split.
    """
    X, y, meta = _prepare(df, target, features, task)
    if X is None:
        return {"status": "failed", **meta}

    numeric, categorical = meta["numeric_features"], meta["categorical_features"]
    n_rows = len(X)

    # Folds are capped by the rarest class so stratification remains possible.
    if task == "classification":
        min_class = int(y.value_counts().min())
        folds = max(2, min(cv_folds, min_class))
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro",
                   "precision_macro": "precision_macro", "recall_macro": "recall_macro"}
        if y.nunique() == 2:
            scoring["roc_auc"] = "roc_auc"
        primary = "f1_macro"
    else:
        folds = max(2, min(cv_folds, n_rows // 10))
        splitter = KFold(n_splits=folds, shuffle=True, random_state=random_state)
        scoring = {"r2": "r2",
                   "neg_rmse": "neg_root_mean_squared_error",
                   "neg_mae": "neg_mean_absolute_error"}
        primary = "r2"

    pre = _preprocessor(numeric, categorical)
    results = []

    for name, model in _candidate_models(task, n_rows, y.nunique(), _imbalance_ratio(y)).items():
        pipe = Pipeline([("prep", pre), ("model", model)])
        try:
            cv = cross_validate(pipe, X, y, cv=splitter, scoring=scoring,
                                n_jobs=1, error_score="raise")
        except Exception as exc:
            results.append({"model": name, "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}"})
            continue

        scores = _scores_from_cv(cv, scoring)

        results.append({
            "model": name,
            "status": "ok",
            "scores": scores,
            "primary_metric": primary,
            "primary_score": scores[primary],
            "fit_time_s": round(float(np.mean(cv["fit_time"])), 4),
            "is_baseline": name in ("Majority baseline", "Random baseline", "Mean baseline"),
        })

    successful = [r for r in results if r["status"] == "ok"]
    if not successful:
        return {"status": "failed", "error": "Every candidate model failed to fit.",
                "results": results, **meta}

    successful.sort(key=lambda r: -r["primary_score"])
    best, selection = _select_best(successful, primary, task)

    # Compare against the strongest baseline on the primary metric, not the
    # first one found; a weak baseline would flatter every model above it.
    baselines = [r for r in successful if r["is_baseline"]]
    baseline = max(baselines, key=lambda r: r["primary_score"]) if baselines else None

    # Search the winning family's hyperparameters, and adopt the result only if
    # it beat the defaults on rows the search never saw. Tuning is commonly
    # assumed to help; measured on held-out data it sometimes does not,
    # particularly on small datasets where the search overfits its own inner
    # folds. Adopting unconditionally would then make the reported score worse
    # while appearing more sophisticated.
    tuning = None
    if CONFIG.tune_models and not best["is_baseline"]:
        tuning = tune_model(df, target, features, task, best["model"], random_state)
        if tuning and tuning.get("status") == "ok" and tuning["improved"]:
            tuned_model = _candidate_models(task, n_rows, y.nunique(), _imbalance_ratio(y))[best["model"]]
            tuned_model.set_params(**tuning["best_params"])
            tuned_pipe = Pipeline([("prep", _preprocessor(numeric, categorical)),
                                   ("model", tuned_model)])
            try:
                # Re-cross-validated on the same folds as every other candidate,
                # so the reported score stays comparable with the table above
                # rather than being a held-out number among cross-validated
                # ones.
                #
                # This comparability is bought at a price that must be stated.
                # The parameters were selected on 75% of these same rows, which
                # overlaps every outer fold's test portion, so the tuned score
                # is optimistically biased -- while `best["primary_score"]`, the
                # number it is compared against, involved no selection on any of
                # the data. The comparison is therefore asymmetric and tilts
                # towards adopting tuning.
                #
                # Removing the bias properly requires nested cross-validation,
                # re-selecting parameters inside each outer fold, at a cost of
                # `tuning_iterations` x inner folds x outer folds refits. That
                # is beyond what a rate-limited free tier makes practical here.
                # What is done instead is the same thing this framework does
                # everywhere else it cannot measure something cleanly: adopt
                # only when the *unbiased* comparison already agreed --
                # `tune_model` scores tuned against untuned on rows neither was
                # fitted or selected on, and `improved` reflects that -- and
                # record the limitation on the result so the report can state
                # it rather than imply a cleanliness the number does not have.
                cv = cross_validate(tuned_pipe, X, y, cv=splitter, scoring=scoring,
                                    n_jobs=1, error_score="raise")
                tuned_scores = _scores_from_cv(cv, scoring)
                if tuned_scores[primary] > best["primary_score"]:
                    best = {
                        "model": best["model"],
                        "status": "ok",
                        "scores": tuned_scores,
                        "primary_metric": primary,
                        "primary_score": tuned_scores[primary],
                        "fit_time_s": round(float(np.mean(cv["fit_time"])), 4),
                        "is_baseline": False,
                        "tuned": True,
                        "params": tuning["best_params"],
                        # The clean number: tuned against untuned on rows used
                        # for neither fitting nor selection.
                        "held_out_gain": tuning.get("gain"),
                        "score_caveat": (
                            "Hyperparameters were selected on a 75% subset of "
                            "these rows, so this cross-validated score is "
                            "optimistic relative to the untuned models in the "
                            "same table. The decision to adopt tuning rests on "
                            "the held-out comparison, not on this score."),
                    }
                    successful = [best] + [r for r in successful
                                           if not (r["model"] == best["model"]
                                                   and not r.get("tuned"))]
                    successful.sort(key=lambda r: -r["primary_score"])
                    tuning["adopted"] = True
                else:
                    tuning["adopted"] = False
                    tuning["adoption_note"] = (
                        "The tuned parameters won on the held-out split but not "
                        "under cross-validation, so the defaults were kept.")
            except Exception as exc:
                tuning["adopted"] = False
                tuning["adoption_note"] = f"{type(exc).__name__}: {exc}"
        elif tuning and tuning.get("status") == "ok":
            tuning["adopted"] = False
            tuning["adoption_note"] = (
                "The search did not beat the default parameters on held-out data, "
                "so the defaults were kept.")

    # Selection is decided again now that the scores are final. The first pass
    # had to choose a model *to* tune; tuning then changes that model's scores
    # and can move it in or out of the tied group. Leaving the earlier choice
    # standing would report a selection made against numbers that no longer
    # hold.
    best, selection = _select_best(successful, primary, task)
    selection["re_selected_after_tuning"] = bool(tuning and tuning.get("adopted"))

    lift = None
    if baseline:
        lift = round(best["primary_score"] - baseline["primary_score"], 4)

    # A permutation test on the winning model. Comparing against a dummy
    # estimator is not sufficient: a flexible model can find structure in noise
    # that survives cross-validation by chance, and on macro F1 the dummy
    # baselines sit below true chance level. Refitting against repeatedly
    # shuffled targets measures the score distribution when no relationship
    # exists, which is the honest reference for whether signal was learned.
    permutation = None
    if not best["is_baseline"]:
        n_perm = CONFIG.permutation_tests
        try:
            model = _candidate_models(task, n_rows, y.nunique(), _imbalance_ratio(y))[best["model"]]
            # The permutation test must exercise the model actually being
            # reported, so it uses the tuned parameters when those were adopted.
            if best.get("tuned") and best.get("params"):
                model.set_params(**best["params"])
            pipe = Pipeline([("prep", pre), ("model", model)])
            score, perm_scores, p_value = permutation_test_score(
                pipe, X, y, cv=splitter, scoring=scoring[primary],
                n_permutations=n_perm, random_state=random_state, n_jobs=-1)
            permutation = {
                "score": round(float(score), 4),
                "null_mean": round(float(np.mean(perm_scores)), 4),
                "null_std": round(float(np.std(perm_scores)), 4),
                "p_value": round(float(p_value), 4),
                "n_permutations": n_perm,
                "alpha": CONFIG.permutation_alpha,
                "significant": bool(p_value < CONFIG.permutation_alpha),
            }
        except Exception:
            permutation = None  # cost or compatibility failure, not a run failure

    # What the chosen model does in practice, in counts rather than in a score.
    behaviour = None
    if not best["is_baseline"]:
        try:
            model = _candidate_models(task, n_rows, y.nunique(), _imbalance_ratio(y))[best["model"]]
            if best.get("tuned") and best.get("params"):
                model.set_params(**best["params"])
            behaviour = _operating_behaviour(
                Pipeline([("prep", pre), ("model", model)]),
                X, y, splitter, task)
        except Exception:
            behaviour = None

    return {
        "status": "ok",
        "task": task,
        "target": target,
        "cv_folds": folds,
        "operating_behaviour": behaviour,
        "primary_metric": primary,
        "results": successful,
        "failed_models": [r for r in results if r["status"] != "ok"],
        "best_model": best["model"],
        "best_score": best["primary_score"],
        # A baseline winning is a finding, not a recommendation: it means
        # nothing learned anything. The model is not saved in that case.
        "best_is_baseline": bool(best["is_baseline"]),
        "baseline_model": baseline["model"] if baseline else None,
        "baseline_score": baseline["primary_score"] if baseline else None,
        "lift_over_baseline": lift,
        # The improvement must exceed the model's own fold-to-fold variability,
        # otherwise it is within the noise of the split rather than evidence of
        # learned signal. A model that barely beats chance has found little the
        # report should present as predictive, and saying so is a finding.
        "beats_baseline": bool(
            lift is not None
            and lift > 0.02
            and lift > 2 * best["scores"].get(f"{primary}_std", 0.0)
            # When the permutation test ran, it is the deciding evidence: a
            # score indistinguishable from shuffled targets is not signal.
            and (permutation is None or permutation["significant"])
        ),
        "best_score_std": best["scores"].get(f"{primary}_std"),
        "model_selection": selection,
        "tuning": tuning,
        "best_model_tuned": bool(best.get("tuned")),
        "best_model_params": best.get("params"),
        "permutation_test": permutation,
        **meta,
    }


def _param_grid(task: str, model_name: str) -> dict[str, list[Any]]:
    """Search ranges for each model family.

    The ranges are deliberately wide rather than centred on the defaults,
    because the point of searching is to discover that the default was wrong.
    `class_weight` is included for every classifier that supports it: on an
    imbalanced target it is usually the single change that matters most, and it
    costs nothing to try.
    """
    if task == "classification":
        grids = {
            "Logistic Regression": {
                "model__C": [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0],
                "model__class_weight": [None, "balanced"],
            },
            "Decision Tree": {
                "model__max_depth": [3, 4, 5, 6, 8, 10, None],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
                "model__class_weight": [None, "balanced"],
            },
            "Random Forest": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [6, 10, 16, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
            "Gradient Boosting": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.7, 0.85, 1.0],
            },
            "Hist Gradient Boosting": {
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__max_iter": [100, 200, 400],
                "model__max_leaf_nodes": [15, 31, 63],
                "model__l2_regularization": [0.0, 0.1, 1.0],
                "model__class_weight": [None, "balanced"],
            },
            "Extra Trees": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [8, 16, None],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": ["sqrt", "log2", None],
                "model__class_weight": [None, "balanced"],
            },
        }
    else:
        grids = {
            "Ridge Regression": {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            "Decision Tree": {
                "model__max_depth": [3, 4, 5, 6, 8, 10, None],
                "model__min_samples_leaf": [1, 2, 5, 10, 20],
            },
            "Random Forest": {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [6, 10, 16, None],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": ["sqrt", "log2", None],
            },
            "Gradient Boosting": {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "model__max_depth": [2, 3, 4, 5],
                "model__subsample": [0.7, 0.85, 1.0],
            },
        }
    return grids.get(model_name, {})


def tune_model(df: pd.DataFrame, target: str, features: list[str], task: str,
               model_name: str, random_state: int = 42) -> dict[str, Any] | None:
    """Search hyperparameters for the winning model and measure the gain honestly.

    The search must not see the data it is judged on. The frame is split once;
    the search runs with its own inner cross-validation inside the training
    portion, and both the tuned and the untuned model are then scored on the
    held-out portion that neither of them was fitted or selected on. Reporting a
    cross-validated score from the search itself would be reporting the maximum
    of many attempts, which is optimistic by construction.

    Returning the untuned score alongside the tuned one is the point: it makes
    "tuning helped" a measurement rather than an assumption, and tuning does not
    always help.
    """
    grid = _param_grid(task, model_name)
    if not grid:
        return None

    X, y, meta = _prepare(df, target, features, task)
    if X is None:
        return None
    if len(X) > CONFIG.tuning_max_rows:
        return {"status": "skipped",
                "reason": (f"{len(X):,} rows exceeds the tuning limit of "
                           f"{CONFIG.tuning_max_rows:,}; the search would cost more "
                           f"than the expected gain.")}

    numeric, categorical = meta["numeric_features"], meta["categorical_features"]
    pre = _preprocessor(numeric, categorical)

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=stratify)
    except ValueError:
        return None

    if task == "classification":
        metric = "f1_macro"
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    else:
        metric = "r2"
        inner = KFold(n_splits=3, shuffle=True, random_state=random_state)

    base = _candidate_models(task, len(X_train), y.nunique(), _imbalance_ratio(y))
    if model_name not in base:
        return None
    pipe = Pipeline([("prep", pre), ("model", base[model_name])])

    try:
        search = RandomizedSearchCV(
            pipe, grid, n_iter=CONFIG.tuning_iterations, cv=inner, scoring=metric,
            random_state=random_state, n_jobs=-1, error_score="raise")
        search.fit(X_train, y_train)

        tuned_score = float(search.score(X_test, y_test))

        # The same model family with its default settings, fitted on the same
        # training rows and scored on the same held-out rows. Any difference is
        # attributable to the parameters and nothing else.
        untuned = Pipeline([("prep", _preprocessor(numeric, categorical)),
                            ("model", _candidate_models(
                                task, len(X_train), y.nunique())[model_name])])
        untuned.fit(X_train, y_train)
        scorer = get_scorer(metric)
        untuned_score = float(scorer(untuned, X_test, y_test))
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    gain = round(tuned_score - untuned_score, 4)
    return {
        "status": "ok",
        "model": model_name,
        "metric": metric,
        "best_params": {k.replace("model__", ""): v
                        for k, v in search.best_params_.items()},
        "tuned_holdout_score": round(tuned_score, 4),
        "untuned_holdout_score": round(untuned_score, 4),
        "gain": gain,
        # A gain smaller than a fifth of a percentage point is not worth
        # reporting as an improvement; it is within the variation of a single
        # split. Saying so prevents the report from claiming a benefit the
        # measurement does not support.
        "improved": bool(gain > 0.002),
        "n_candidates": CONFIG.tuning_iterations,
        "holdout_rows": int(len(X_test)),
    }


def fit_final_model(df: pd.DataFrame, target: str, features: list[str],
                    task: str, model_name: str,
                    params: dict[str, Any] | None = None,
                    random_state: int = 42,
                    preparation: list[dict[str, Any]] | None = None,
                    ) -> tuple[Any, dict[str, Any]] | None:
    """Fit the chosen model on every row, ready to be saved and reused.

    The cross-validated pipelines above are fitted on folds and discarded: each
    saw only part of the data, and none of them survives the loop. A model a
    report recommends has to be a real object someone can load, so the winner is
    refitted once on the complete dataset.

    Returned with a signature -- the columns it expects, their dtypes, the
    target it predicts, the classes it can emit -- because a pipeline without
    one is a black box that fails confusingly on the first column it does not
    recognise. Preprocessing is inside the pipeline, so the saved object takes
    raw rows in the original shape rather than requiring the caller to
    reproduce imputation and encoding.

    `preparation` is the run's record of what was done to the data before
    modelling: repairs from the quality stage and derived columns from the
    feature stage, each with its fitted parameters. When given, those steps
    become the pipeline's first stage (`RawRowPreparer`), and the signature
    lists the raw columns they are built from rather than the derived ones.
    """
    X, y, meta = _prepare(df, target, features, task)
    if X is None:
        return None

    candidates = _candidate_models(task, len(X), y.nunique(), _imbalance_ratio(y))
    if model_name not in candidates:
        return None

    model = candidates[model_name]
    if params:
        try:
            model.set_params(**params)
        except ValueError:
            # Tuned parameters belong to the model they were searched for; a
            # mismatch means the winner changed after tuning, and defaults are
            # the honest fallback.
            pass

    from tools.preparation import RawRowPreparer, plan_inputs

    model_columns = list(X.columns)
    inputs, steps = (plan_inputs(model_columns, preparation) if preparation
                     else (model_columns, []))

    stages: list[tuple[str, Any]] = [
        ("prep", _preprocessor(meta["numeric_features"],
                               meta["categorical_features"])),
        ("model", model),
    ]
    frame = X
    if steps:
        # Fitted on the working frame's raw columns, which the preparer turns
        # back into exactly `X`: every step is idempotent on data it has
        # already been applied to, so the fit is the same one cross-validated.
        stages.insert(0, ("prepare", RawRowPreparer(steps, inputs, model_columns)))
        frame = df.loc[X.index, inputs]
    pipeline = Pipeline(stages)
    try:
        pipeline.fit(frame, y)
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, {}

    signature = {
        "target": target,
        "task": task,
        "model": model_name,
        "tuned_params": params or None,
        # What a caller passes: raw columns, in the shape of the original file.
        "expects_columns": list(inputs),
        "column_dtypes": {c: str(frame[c].dtype) for c in inputs},
        # What the model itself was fitted on, after the preparation below.
        "model_columns": model_columns,
        "derived_features": [
            {"feature": s["feature"], "kind": s["kind"], "source": s["source"]}
            for s in steps if s.get("stage") == "features"],
        "input_repairs": [
            {"action": s["action"], "column": s["column"]}
            for s in steps if s.get("stage") == "quality"],
        "numeric_features": meta["numeric_features"],
        "categorical_features": meta["categorical_features"],
        "n_training_rows": int(len(X)),
        "classes": ([str(c) for c in sorted(pd.Series(y).unique().tolist(),
                                            key=str)]
                    if task == "classification" else None),
        "fitted_at": datetime.now().isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        # Stated because a pipeline is only valid for data like the data it saw,
        # and a reader who loads it months later has no other way to know.
        "note": ("Fitted on every row of the dataset this report analysed. "
                 "Pass raw rows with the columns in expects_columns: the "
                 "pipeline repeats the repairs and derived features the "
                 "analysis applied before modelling, then its own imputation "
                 "and encoding."),
    }
    return pipeline, signature


def feature_importance(df: pd.DataFrame, target: str, features: list[str],
                       task: str, model_name: str = "Random Forest",
                       random_state: int = 42) -> dict[str, Any]:
    """Permutation importance for the fitted model, measured on held-out data.

    Permutation importance is used rather than a tree's built-in importance
    because the latter is biased toward high-cardinality features. Measuring on a
    held-out split means importance reflects predictive contribution rather than
    how well the model memorised the training rows.
    """
    X, y, meta = _prepare(df, target, features, task)
    if X is None:
        return {"status": "failed", **meta}

    stratify = y if task == "classification" and y.value_counts().min() >= 2 else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=stratify)
    except ValueError as exc:
        return {"status": "failed", "error": f"Could not split the data: {exc}"}

    models = _candidate_models(task, len(X), y.nunique(), _imbalance_ratio(y))
    # An explicit None check: estimators define __len__, so `a or b` would
    # evaluate the estimator's truthiness and raise on an unfitted model.
    model = models[model_name] if model_name in models else models["Random Forest"]
    pipe = Pipeline([("prep", _preprocessor(meta["numeric_features"],
                                            meta["categorical_features"])),
                     ("model", model)])

    try:
        pipe.fit(X_train, y_train)
        scoring = "f1_macro" if task == "classification" else "r2"
        perm = permutation_importance(pipe, X_test, y_test, n_repeats=10,
                                      random_state=random_state, scoring=scoring, n_jobs=1)
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

    importances = sorted(
        ({"feature": col,
          "importance": round(float(m), 5),
          "std": round(float(s), 5),
          # An importance smaller than its own variability is indistinguishable
          # from noise and must not be reported as a driver.
          "informative": bool(m > s and m > 0.001)}
         for col, m, s in zip(X.columns, perm.importances_mean, perm.importances_std)),
        key=lambda d: -d["importance"],
    )

    return {
        "status": "ok",
        "model": model_name,
        "metric": scoring,
        "test_rows": int(len(X_test)),
        "importances": importances,
        "informative_features": [i["feature"] for i in importances if i["informative"]],
        "uninformative_features": [i["feature"] for i in importances if not i["informative"]],
    }
