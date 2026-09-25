"""Feature engineering: derive predictors the raw columns only imply.

The framework cleaned data and then modelled it, with nothing in between. That
is a real hole in a data science workflow: most of the accuracy available in a
tabular problem sits in representation rather than in model choice, and a
pipeline that never transforms its inputs leaves it unclaimed.

Every transformation here is generic. Nothing keys off a particular dataset's
column names, because a framework that knows about "Cabin" or "Pclass" is
demonstrating that its author knows Titanic, not that the system works. The
transformations are chosen instead from what the data shows: a column is
log-transformed because it is skewed, a category is grouped because it is rare,
a missingness flag is added because the absence itself varies with the outcome.

Two deliberate exclusions:

Target encoding is not implemented, despite usually being the single most
effective transformation for high-cardinality categoricals. Computing a category's
mean outcome over the whole frame and then cross-validating on that frame leaks
the validation fold into the training features, and the inflated score that
follows looks like success. Doing it correctly requires the encoding to be refitted
inside every fold, which the evaluation's shared-splitter design does not
currently express. An honest omission is better than a leaking feature.

Interaction terms between arbitrary numeric pairs are also excluded. On a wide
frame they multiply the feature space faster than the sample size can support,
and the resulting overfit is invisible to a single train/test split.
"""
from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np
import pandas as pd


# A column with more distinct values than this is treated as high-cardinality:
# one-hot encoding it produces more columns than most datasets have rows to
# support, so it is frequency-encoded instead.
HIGH_CARDINALITY = 15

# Categories rarer than this share of the column are consolidated. Below roughly
# one percent a category contributes too few rows to estimate anything from, but
# still costs a column when one-hot encoded.
RARE_CATEGORY_FRACTION = 0.01

# Missingness is informative only when it varies. A column that is almost never
# missing, or almost always missing, carries no signal in its absence pattern.
MISSING_FLAG_RANGE = (0.02, 0.95)

# Skew beyond this makes a positive numeric column a candidate for a log
# transform. Below it the transform changes little and costs interpretability.
SKEW_THRESHOLD = 1.0


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _add_datetime_parts(df: pd.DataFrame, out: pd.DataFrame,
                        feature_columns: list[str],
                        applied: list[dict[str, Any]],
                        room: Callable[[], bool]) -> None:
    """Split a timestamp into the parts a model can use.

    A raw timestamp is unusable as a model input, but a seasonal or weekly
    effect lives in the month and the weekday.

    Appends to `applied` and writes into `out`; `room` reports whether
    the feature budget still allows another column.
    """
    # --- 1. Datetime decomposition -----------------------------------------
    # A timestamp is unusable as a model input but the parts of it are: seasonal
    # and weekly effects live in the month and weekday, not in the epoch value.
    for name in feature_columns:
        if not room():
            break
        col = df[name]
        if not pd.api.types.is_datetime64_any_dtype(col):
            continue
        for part, values in (("year", col.dt.year), ("month", col.dt.month),
                             ("dayofweek", col.dt.dayofweek)):
            if not room() or values.nunique(dropna=True) <= 1:
                continue
            new = f"{name}_{part}"
            out[new] = values
            applied.append({
                "feature": new, "kind": "datetime_part", "source": name,
                "rationale": (f"Extracted {part} from {name}; a raw timestamp "
                              f"cannot express a seasonal or weekly effect."),
                "_params": {"part": part},
            })

def _add_missing_indicators(df: pd.DataFrame, out: pd.DataFrame,
                            feature_columns: list[str],
                            applied: list[dict[str, Any]],
                            room: Callable[[], bool]) -> None:
    """Record that a value was absent, before imputation erases the fact.

    Whether something was measured is often a fact about the subject rather
    than an accident of collection.

    Appends to `applied` and writes into `out`; `room` reports whether
    the feature budget still allows another column.
    """
    lo, hi = MISSING_FLAG_RANGE
    for name in feature_columns:
        if not room():
            break
        fraction = float(df[name].isna().mean())
        if not (lo <= fraction <= hi):
            continue
        new = f"{name}_was_missing"
        out[new] = df[name].isna().astype(int)
        applied.append({
            "feature": new, "kind": "missing_indicator", "source": name,
            "rationale": (f"{name} is missing for {fraction:.1%} of rows; the "
                          f"fact of absence is preserved before imputation "
                          f"replaces it."),
        })

def _add_skew_corrections(df: pd.DataFrame, out: pd.DataFrame,
                          feature_columns: list[str],
                          applied: list[dict[str, Any]],
                          room: Callable[[], bool]) -> None:
    """Compress a long right tail so a handful of rows stop dominating the fit.

    Only a right tail: log1p applied to a left-skewed column lengthens the
    tail it should be correcting.

    Appends to `applied` and writes into `out`; `room` reports whether
    the feature budget still allows another column.
    """
    # --- 3. Skew correction -------------------------------------------------
    # A long right tail lets a handful of extreme rows dominate a distance or a
    # linear coefficient. The log compresses the tail without reordering values.
    for name in feature_columns:
        if not room():
            break
        col = df[name]
        if not _is_numeric(col):
            continue
        values = col.dropna()
        if len(values) < 30 or values.min() < 0 or values.nunique() < 10:
            continue
        skew = float(values.skew())
        # Only a right tail. log1p is monotonic and compresses large values, so
        # applied to a left-skewed column it lengthens the tail it should be
        # correcting and makes the distribution worse, not better.
        if skew < SKEW_THRESHOLD:
            continue
        new = f"{name}_log"
        out[new] = np.log1p(col.clip(lower=0))
        applied.append({
            "feature": new, "kind": "log_transform", "source": name,
            "rationale": (f"{name} has skewness {skew:.2f}; log1p compresses the "
                          f"tail so extreme values stop dominating the fit."),
        })

def _add_rare_category_groups(df: pd.DataFrame, out: pd.DataFrame,
                              feature_columns: list[str],
                              applied: list[dict[str, Any]],
                              room: Callable[[], bool]) -> None:
    """Consolidate categories too rare to estimate anything from.

    A category holding a handful of rows supports no estimate but still
    costs a column once encoded.

    Appends to `applied` and writes into `out`; `room` reports whether
    the feature budget still allows another column.
    """
    # --- 4. Rare-category consolidation ------------------------------------
    # A category holding a handful of rows supports no estimate, but still costs
    # a column once encoded.
    for name in feature_columns:
        if not room():
            break
        col = df[name]
        if _is_numeric(col) or pd.api.types.is_datetime64_any_dtype(col):
            continue
        values = col.dropna().astype(str)
        if values.empty:
            continue
        # An identifier has no categories to group: every value is distinct, so
        # the rule below would collapse the entire column into one constant.
        if values.nunique() >= 0.5 * len(values):
            continue
        frequencies = values.value_counts(normalize=True)
        rare = frequencies[frequencies < RARE_CATEGORY_FRACTION]
        if len(rare) < 2:
            continue
        # Grouping is only worth doing if something survives ungrouped. When the
        # rare categories cover most of the column the result is near-constant
        # and carries less information than the original.
        rare_share = float(frequencies[frequencies < RARE_CATEGORY_FRACTION].sum())
        if rare_share > 0.5:
            continue
        new = f"{name}_grouped"
        kept = sorted(set(frequencies.index) - set(rare.index))
        # Masked on `notna()`, because `astype(str)` renders missing values as
        # the literal strings "nan" and "None" -- two fabricated categories,
        # one-hot encoded downstream as genuine predictors. `tools.quality`
        # documents this exact mistake and avoids it the same way; the feature
        # builders reintroduced it. Whether a value was recorded is a fact
        # about the subject, and the missing-indicator transformation exists to
        # capture it deliberately rather than by accident.
        present = col.notna()
        as_text = col.where(present).astype(object)
        grouped = as_text.where(~col.astype(str).isin(rare.index), "__rare__")
        out[new] = grouped.where(present)
        applied.append({
            "feature": new, "kind": "rare_category_grouping", "source": name,
            "rationale": (f"{len(rare)} categories in {name} each cover under "
                          f"{RARE_CATEGORY_FRACTION:.0%} of rows; grouping them "
                          f"avoids estimating from a handful of examples."),
            # The categories kept, rather than the ones grouped: a category
            # first seen at scoring time is rare by definition, and grouping
            # it is what the fitted encoder can use.
            "_params": {"kept": kept},
        })

def _add_frequency_encodings(df: pd.DataFrame, out: pd.DataFrame,
                             feature_columns: list[str],
                             applied: list[dict[str, Any]],
                             room: Callable[[], bool]) -> None:
    """Replace a high-cardinality column with how common each value is.

    How often a value occurs often matters more than which value it is, and
    this holds that in one column instead of hundreds.

    Appends to `applied` and writes into `out`; `room` reports whether
    the feature budget still allows another column.
    """
    # --- 5. Frequency encoding for high-cardinality categoricals -----------
    # How common a value is often matters more than which value it is, and this
    # holds it in one column rather than hundreds.
    for name in feature_columns:
        if not room():
            break
        col = df[name]
        if _is_numeric(col) or pd.api.types.is_datetime64_any_dtype(col):
            continue
        n_unique = col.nunique(dropna=True)
        if n_unique <= HIGH_CARDINALITY or n_unique >= 0.9 * len(df):
            continue
        frequency = col.value_counts()
        counts = col.map(frequency)
        new = f"{name}_frequency"
        out[new] = counts.fillna(0).astype(float)
        applied.append({
            "feature": new, "kind": "frequency_encoding", "source": name,
            "rationale": (f"{name} has {n_unique} distinct values; encoding how "
                          f"often each occurs keeps the information in one "
                          f"column instead of {n_unique}."),
            "_params": {"counts": frequency.to_dict()},
        })

def _add_salient_tokens(df: pd.DataFrame, out: pd.DataFrame,
                        feature_columns: list[str],
                        applied: list[dict[str, Any]],
                        room: Callable[[], bool]) -> None:
    """Extract the repeated structural token inside near-unique text.

    An honorific, a product line, a branch code: discovered from the data
    rather than named in advance.

    Appends to `applied` and writes into `out`; `room` reports whether
    the feature budget still allows another column.
    """
    # --- 6. Salient token extraction from structured text ------------------
    # Free-text identifiers frequently carry a repeated structural token -- an
    # honorific, a product line, a branch code -- that groups rows meaningfully
    # while the full string is unique to each. The token is discovered from the
    # data rather than named in advance.
    for name in feature_columns:
        if not room():
            break
        col = df[name]
        if _is_numeric(col) or pd.api.types.is_datetime64_any_dtype(col):
            continue
        values = col.dropna().astype(str)
        if len(values) < 50 or values.nunique() < 0.5 * len(values):
            continue
        tokens = values.map(_salient_token)
        # The structural token is the one that repeats. Incidental short words --
        # a four-letter surname reached before the honorific -- appear a handful
        # of times each and would otherwise inflate the distinct count past any
        # threshold, discarding a feature that is mostly well-behaved. Keeping
        # the frequent tokens and grouping the rest recovers it.
        frequencies = tokens[tokens != ""].value_counts(normalize=True)
        frequent = frequencies[frequencies >= RARE_CATEGORY_FRACTION].head(HIGH_CARDINALITY)
        coverage = float(frequent.sum())
        if len(frequent) < 2 or coverage < 0.5:
            continue
        keep = set(frequent.index)
        new = f"{name}_token"
        # Same masking as the rare-category grouping above: without it
        # `_salient_token("nan")` returns "nan", which is short enough to be
        # kept as a genuine token.
        present = col.notna()
        extracted = col.astype(str).map(_salient_token).where(present)
        out[new] = extracted.where(extracted.isin(keep) | ~present,
                                   "__other__").where(present)
        applied.append({
            "feature": new, "kind": "salient_token", "source": name,
            "rationale": (f"{name} is near-unique per row but contains a repeated "
                          f"token; the {len(frequent)} frequent values cover "
                          f"{coverage:.0%} of rows and group rows the full string "
                          f"cannot."),
            "_params": {"keep": sorted(keep)},
        })


# Applied in this order. Order matters because the feature budget is
# consumed first-come: when a wide frame cannot afford every derived
# column, the transformations listed first are the ones that survive.
TRANSFORMATIONS = (
    _add_datetime_parts,
    _add_missing_indicators,
    _add_skew_corrections,
    _add_rare_category_groups,
    _add_frequency_encodings,
    _add_salient_tokens,
)


def engineer_features(df: pd.DataFrame, target: str | None,
                      task: str | None = None) -> dict[str, Any]:
    """Derive additional features, returning the new frame and a record of why.

    Every addition carries its rationale, so the report can explain what was
    built rather than presenting unexplained columns. Declined transformations
    are recorded too: knowing that target encoding was considered and rejected
    for leakage is more informative than its silent absence.
    """
    out = df.copy()
    applied: list[dict[str, Any]] = []
    declined: list[dict[str, str]] = []

    feature_columns = [c for c in df.columns if c != target]

    # A budget on how many features may be added. Doubling a wide frame's width
    # costs more in dimensionality than the new columns return, and the models
    # this framework fits are not regularised enough to absorb it.
    budget = max(8, min(30, len(feature_columns)))

    def room() -> bool:
        return len(applied) < budget

    # Each transformation is applied in turn, and each stops itself when the
    # budget is exhausted. The order is load-bearing: the budget is consumed
    # first-come, so the earlier entries get priority when a wide frame cannot
    # afford everything.
    for transformation in TRANSFORMATIONS:
        if not room():
            break
        transformation(df, out, feature_columns, applied, room)



    declined.append({
        "kind": "target_encoding",
        "reason": ("Encoding a category by its mean outcome leaks the validation "
                   "fold into the training features when computed over the whole "
                   "frame, inflating the score. It requires refitting inside each "
                   "fold, which this pipeline does not yet express."),
    })
    declined.append({
        "kind": "numeric_interactions",
        "reason": ("Pairwise products expand the feature space faster than the "
                   "sample supports, and the resulting overfit is not visible to "
                   "a single split."),
    })
    if len(applied) >= budget:
        declined.append({
            "kind": "additional_features",
            "reason": (f"The budget of {budget} engineered features was reached; "
                       f"further additions cost more in dimensionality than they "
                       f"return."),
        })

    # The fitted parameters are separated from the description of what was
    # built. `applied` is shown to the model and kept as evidence; the recipe
    # holds what a saved model needs to rebuild each column on new rows --
    # a frequency table can run to thousands of entries, which belongs in
    # neither a prompt nor the set of numbers a report may cite.
    recipe = []
    for entry in applied:
        params = entry.pop("_params", {})
        recipe.append({"stage": "features", "kind": entry["kind"],
                       "feature": entry["feature"], "source": entry["source"],
                       **params})

    return {
        "frame": out,
        "applied": applied,
        "declined": declined,
        "recipe": recipe,
        "n_added": len(applied),
        "n_columns_before": int(df.shape[1]),
        "n_columns_after": int(out.shape[1]),
        "budget": budget,
    }


def apply_recipe(df: pd.DataFrame, recipe: list[dict[str, Any]]) -> pd.DataFrame:
    """Rebuild derived columns on new rows, using the parameters fitted at the time.

    Each branch repeats the construction in the matching `_add_*` function
    with its fitted values held fixed: the categories kept, the frequency
    table, the token vocabulary. Recomputing them from the new rows instead
    would give a different column from the one the model was fitted on.
    """
    out = df.copy()
    for step in recipe:
        source, new, kind = step["source"], step["feature"], step["kind"]
        if source not in out.columns:
            continue
        col = out[source]
        present = col.notna()
        if kind == "datetime_part":
            stamps = pd.to_datetime(col, errors="coerce")
            out[new] = getattr(stamps.dt, step["part"])
        elif kind == "missing_indicator":
            out[new] = col.isna().astype(int)
        elif kind == "log_transform":
            out[new] = np.log1p(pd.to_numeric(col, errors="coerce").clip(lower=0))
        elif kind == "rare_category_grouping":
            kept = set(step["kept"])
            grouped = col.where(present).astype(object).where(
                col.astype(str).isin(kept), "__rare__")
            out[new] = grouped.where(present)
        elif kind == "frequency_encoding":
            out[new] = col.map(step["counts"]).fillna(0).astype(float)
        elif kind == "salient_token":
            keep = set(step["keep"])
            extracted = col.astype(str).map(_salient_token).where(present)
            out[new] = extracted.where(extracted.isin(keep) | ~present,
                                       "__other__").where(present)
    return out


_TOKEN = re.compile(r"\b([A-Za-z]{2,})\b")


def _salient_token(text: str) -> str:
    """The first short alphabetic token in a string, or empty when there is none.

    Structured identifiers tend to place the grouping token early and keep it
    short: an honorific before a name, a class code before a serial. Taking the
    first token under five characters finds it without naming the field.
    """
    for match in _TOKEN.finditer(text):
        token = match.group(1)
        if len(token) <= 4:
            return token
    return ""


def measure_feature_gain(before: pd.DataFrame, after: pd.DataFrame,
                         target: str, task: str,
                         random_state: int = 42) -> dict[str, Any]:
    """Cross-validate with and without the engineered features and compare.

    Feature engineering is conventionally assumed to help, and on this project's
    own datasets it frequently does not: tree ensembles are invariant to
    monotonic transforms, so a log-transformed column adds a dimension without
    adding information, and a derived feature that duplicates an existing one
    costs sample efficiency for nothing. Measuring rather than assuming is the
    same discipline the framework applies to hyperparameter tuning.

    Both frames are scored with the same model family on the same folds, so the
    difference is attributable to the columns and nothing else.
    """
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline

    from tools.modelling import _prepare, _preprocessor, _candidate_models

    if task == "classification":
        metric = "f1_macro"
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        metric = "r2"
        splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)

    failures: list[str] = []

    def score(frame: pd.DataFrame) -> float | None:
        features = [c for c in frame.columns if c != target]
        X, y, meta = _prepare(frame, target, features, task)
        if X is None:
            return None
        # A single flexible model, not the whole comparison: the question is
        # whether the representation carries more signal, not which family
        # exploits it best, and one model keeps the measurement affordable.
        candidates = _candidate_models(task, len(X), y.nunique())
        name = "Random Forest"
        if name not in candidates:
            return None
        pipe = Pipeline([
            ("prep", _preprocessor(meta["numeric_features"],
                                   meta["categorical_features"])),
            ("model", candidates[name]),
        ])
        try:
            scores = cross_val_score(pipe, X, y, cv=splitter, scoring=metric, n_jobs=1)
        except Exception as exc:
            # Recorded on the closure so the caller can report why no gain was
            # measured. Silently returning None made a failed measurement look
            # identical to a measured absence of benefit -- and the second
            # justifies discarding the features while the first does not.
            failures.append(f"{type(exc).__name__}: {exc}")
            return None
        return float(np.mean(scores))

    base, enriched = score(before), score(after)
    if base is None or enriched is None:
        return {"status": "not_measurable",
                "reason": ("The comparison could not be scored on this data"
                           + (f": {failures[0]}" if failures else ".")),
                "failures": failures}

    gain = round(enriched - base, 4)
    return {
        "status": "ok",
        "metric": metric,
        "score_without_features": round(base, 4),
        "score_with_features": round(enriched, 4),
        "gain": gain,
        # The same threshold the tuning stage uses. Below it the difference is
        # within the variation of a resample and is not evidence of improvement.
        "improved": bool(gain > 0.002),
        "n_features_added": int(after.shape[1] - before.shape[1]),
    }
