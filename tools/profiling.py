"""Dataset profiling: structure, types, distributions, and basic statistics."""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from config import CONFIG


def _semantic_type(s: pd.Series, n_rows: int) -> str:
    """Classify a column beyond its raw dtype.

    The dtype alone is not enough for planning: an integer column holding three
    distinct values is categorical for modelling purposes, and a string column
    where every value is unique is an identifier, not a feature.
    """
    non_null = s.dropna()
    if non_null.empty:
        return "empty"

    n_unique = non_null.nunique()

    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_numeric_dtype(s):
        # Near-unique integers that increase monotonically are usually row ids.
        if n_unique == len(non_null) and pd.api.types.is_integer_dtype(s):
            return "identifier"
        if n_unique <= CONFIG.max_classification_classes:
            return "categorical_numeric"
        return "numeric"

    # Object/string columns.
    if n_unique == len(non_null) and len(non_null) > 0.9 * n_rows:
        return "identifier"
    if n_unique <= CONFIG.max_classification_classes:
        return "categorical"
    return "text"


def _numeric_stats(s: pd.Series) -> dict[str, Any]:
    non_null = s.dropna()
    if non_null.empty:
        return {}
    q1, q3 = non_null.quantile([0.25, 0.75])
    return {
        "mean": float(non_null.mean()),
        "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
        "min": float(non_null.min()),
        "q1": float(q1),
        "median": float(non_null.median()),
        "q3": float(q3),
        "max": float(non_null.max()),
        "skewness": float(non_null.skew()) if len(non_null) > 2 else 0.0,
        "kurtosis": float(non_null.kurtosis()) if len(non_null) > 3 else 0.0,
        "zeros": int((non_null == 0).sum()),
        "negatives": int((non_null < 0).sum()),
    }


def _categorical_stats(s: pd.Series, top_n: int = 5) -> dict[str, Any]:
    non_null = s.dropna()
    if non_null.empty:
        return {}
    counts = non_null.value_counts()
    return {
        "top_values": [
            {"value": str(v), "count": int(c), "fraction": round(float(c) / len(non_null), 4)}
            for v, c in counts.head(top_n).items()
        ],
        "imbalance_ratio": round(float(counts.iloc[0] / counts.iloc[-1]), 3)
        if len(counts) > 1
        else 1.0,
    }


def profile_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return a complete structural profile of the dataframe.

    This is the first tool the framework runs; the orchestrator plans the rest
    of the workflow from what it returns.
    """
    n_rows, n_cols = df.shape

    # Large frames are sampled for the expensive per-column statistics, but
    # shape, dtypes and null counts are always computed on the full frame so
    # data-quality findings are never based on a sample.
    sample = (
        df.sample(CONFIG.profile_sample_rows, random_state=42)
        if n_rows > CONFIG.profile_sample_rows
        else df
    )

    columns = []
    for name in df.columns:
        full_col, samp_col = df[name], sample[name]
        n_missing = int(full_col.isna().sum())
        sem = _semantic_type(samp_col, len(sample))

        info: dict[str, Any] = {
            "name": str(name),
            "dtype": str(full_col.dtype),
            "semantic_type": sem,
            "missing": n_missing,
            "missing_fraction": round(n_missing / n_rows, 4) if n_rows else 0.0,
            "unique": int(full_col.nunique(dropna=True)),
            "unique_fraction": round(float(full_col.nunique(dropna=True)) / n_rows, 4)
            if n_rows
            else 0.0,
        }
        if sem in ("numeric", "categorical_numeric") and pd.api.types.is_numeric_dtype(samp_col):
            info["stats"] = _numeric_stats(samp_col)
        elif sem in ("categorical", "boolean", "text"):
            info["stats"] = _categorical_stats(samp_col)
        columns.append(info)

    n_duplicates = int(df.duplicated().sum())
    memory_mb = round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3)

    type_counts: dict[str, int] = {}
    for c in columns:
        type_counts[c["semantic_type"]] = type_counts.get(c["semantic_type"], 0) + 1

    return {
        "shape": {"rows": n_rows, "columns": n_cols},
        "memory_mb": memory_mb,
        "sampled": n_rows > CONFIG.profile_sample_rows,
        # Which fields the sampling applies to. Missing counts, unique counts
        # and duplicates are computed across every row; the per-column
        # distribution statistics and the semantic type are read from the
        # sample. The two can therefore disagree on a large frame -- a column
        # typed `categorical` from 100,000 sampled rows may show a million
        # distinct values in the exact count beside it. Saying which basis each
        # field has costs one line and makes that legible rather than
        # contradictory.
        "sampling_note": (
            f"Semantic types and distribution statistics were read from a "
            f"{CONFIG.profile_sample_rows:,}-row sample; missing, unique and "
            f"duplicate counts are exact across all {n_rows:,} rows."
            if n_rows > CONFIG.profile_sample_rows else None),
        "duplicate_rows": n_duplicates,
        "duplicate_fraction": round(n_duplicates / n_rows, 4) if n_rows else 0.0,
        "total_missing_cells": int(df.isna().sum().sum()),
        "missing_fraction_overall": round(
            float(df.isna().sum().sum()) / (n_rows * n_cols), 4
        )
        if n_rows and n_cols
        else 0.0,
        "semantic_type_counts": type_counts,
        "columns": columns,
    }


def _one_hot_members(names: list[str]) -> set[str]:
    """Columns in a family of three or more sharing a stem and a number.

    A binary member is one level of a one-hot encoded categorical, or one
    indicator among many like it (`pixel_3_0` in an 8x8 image).

    `Wilderness_Area_0` .. `Wilderness_Area_3` encode one feature across four
    columns, and each is fixed by its siblings: when three are 0, the fourth is
    1. Offered as a target, such a column is predicted perfectly from inputs
    that contain its own encoding -- on Covertype the framework once modelled
    `Wilderness_Area_0` at accuracy 1.00 instead of the stated `Cover_Type`.
    """
    stems: dict[str, list[str]] = {}
    for name in names:
        match = re.fullmatch(r"(.+?)_?(\d+)", name)
        if match:
            stems.setdefault(match.group(1), []).append(name)
    return {n for group in stems.values() if len(group) >= 3 for n in group}


def _named_in_goal(column: str, goal: str | None) -> bool:
    """Whether a stated goal names this column, as written or with spaces."""
    if not goal:
        return False
    text = goal.lower()
    for form in {column.lower(), column.lower().replace("_", " ")}:
        if len(form) >= 3 and re.search(rf"(?<![\w]){re.escape(form)}(?![\w])", text):
            return True
    return False


def suggest_targets(profile: dict[str, Any],
                    goal: str | None = None) -> list[dict[str, Any]]:
    """Propose plausible target columns and the task type each would imply.

    The orchestrator uses this to decide whether a modelling branch is worth
    planning at all, and the Model Recommendation Agent uses it as its starting
    point rather than guessing from column names alone.

    A column the stated goal names outright ranks first: the person asking has
    said what the target is, and a name heuristic must not overrule them. One
    level of a one-hot group is never proposed (see `_one_hot_members`).
    """
    candidates = []
    families = _one_hot_members([c["name"] for c in profile["columns"]])
    # Names that commonly denote an outcome. Short entries are matched exactly
    # rather than as substrings: matching "y" anywhere would score almost every
    # column, so "account_age_days" would outrank a column actually named as an
    # outcome.
    exact_names = {"y", "class", "label", "target", "outcome", "result"}
    partial_names = {
        "target", "label", "outcome", "survived", "churn", "price", "salary",
        "income", "wage", "revenue", "score", "rating", "fraud", "default",
        "status", "success", "response",
    }
    outcome_prefixes = ("is_", "has_", "did_", "will_")

    for col in profile["columns"]:
        sem, name = col["semantic_type"], col["name"].lower()
        if sem in ("identifier", "empty", "text"):
            continue
        if col["missing_fraction"] > 0.2:
            continue
        if col["unique"] < 2:  # a constant has nothing to predict
            continue
        # A binary member of a column family is an indicator feature. Only
        # binary ones: `sales_2021`, `sales_2022` form a family too, and the
        # latest year can be exactly what is being predicted.
        if col["unique"] == 2 and col["name"] in families:
            continue
        named = _named_in_goal(col["name"], goal)

        if sem in ("categorical", "categorical_numeric", "boolean"):
            task = "classification"
        elif sem == "numeric":
            task = "regression"
        else:
            continue

        score = 0.0
        if name in exact_names or any(k in name for k in partial_names):
            score += 0.6
        if name.startswith(outcome_prefixes):
            score += 0.5
        if col is profile["columns"][-1]:  # last column is a common convention
            score += 0.2
        # A binary column is far more often an outcome than a feature, whatever
        # its encoding: "yes"/"no" and 0/1 carry the same evidence. A column with
        # several classes is a weaker candidate, because most multi-class
        # categoricals in a table are descriptive attributes rather than the
        # thing being predicted -- marital status, occupation, region.
        if task == "classification":
            if col["unique"] == 2:
                score += 0.4
            elif col["unique"] <= 10:
                score += 0.15

        if named:
            score += 2.0
        if score > 0:
            candidates.append(
                {
                    "column": col["name"],
                    "task": task,
                    "classes": col["unique"] if task == "classification" else None,
                    "confidence": round(min(score, 1.0), 2),
                    "named_in_goal": named,
                    "_rank": score,
                }
            )

    candidates.sort(key=lambda c: c.pop("_rank"), reverse=True)
    return candidates[:5]
