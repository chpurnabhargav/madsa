"""Exploratory data analysis: relationships, group differences and distributions.

Every function here returns effect sizes alongside p-values. On a large dataset
almost any difference reaches statistical significance, so significance alone
would let the framework report trivia as insight. The agent is asked to judge
relationships on effect size and to treat p-values only as evidence that an
effect is distinguishable from noise.

Nothing here interprets. These functions produce the numbers the EDA Agent
reasons about, and those same numbers become the evidence the Validation Agent
checks the final narrative against.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from config import CONFIG
from scipy import stats


def _numeric_columns(df: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    return [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]


def _categorical_columns(df: pd.DataFrame, max_unique: int = 20,
                         exclude: tuple[str, ...] = ()) -> list[str]:
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            # Numeric columns with very few distinct values behave as categories.
            if df[c].nunique(dropna=True) > 10:
                continue
        if 1 < df[c].nunique(dropna=True) <= max_unique:
            out.append(c)
    return out


def _strength(magnitude: float, thresholds: tuple[float, float, float]) -> str:
    weak, moderate, strong = thresholds
    a = abs(magnitude)
    if a >= strong:
        return "strong"
    if a >= moderate:
        return "moderate"
    if a >= weak:
        return "weak"
    return "negligible"


# --------------------------------------------------------------------------- #
# Numeric-numeric relationships
# --------------------------------------------------------------------------- #

def correlations(df: pd.DataFrame, min_abs: float = 0.3,
                 exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Pairwise correlations between numeric columns.

    Both Pearson and Spearman are computed. A large gap between them indicates a
    monotonic but non-linear relationship, which changes what models are
    appropriate, so the gap is reported rather than hidden by picking one.
    """
    cols = _numeric_columns(df, exclude)
    if len(cols) < 2:
        return {"pairs": [], "n_numeric_columns": len(cols),
                "note": "At least two numeric columns are needed for correlation."}

    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            sub = df[[a, b]].dropna()
            if len(sub) < 10 or sub[a].nunique() < 2 or sub[b].nunique() < 2:
                continue
            pear, p_pear = stats.pearsonr(sub[a], sub[b])
            spear, _ = stats.spearmanr(sub[a], sub[b])
            if abs(pear) < min_abs and abs(spear) < min_abs:
                continue
            pairs.append({
                "column_a": a,
                "column_b": b,
                "pearson": round(float(pear), 4),
                "spearman": round(float(spear), 4),
                "p_value": float(f"{p_pear:.4g}"),
                "n": int(len(sub)),
                "strength": _strength(pear, (0.3, 0.5, 0.7)),
                "direction": "positive" if pear > 0 else "negative",
                "non_linear": bool(abs(spear) - abs(pear) > 0.15),
            })

    pairs.sort(key=lambda p: -abs(p["pearson"]))
    return {"pairs": pairs, "n_numeric_columns": len(cols),
            "threshold": min_abs}


def target_correlations(df: pd.DataFrame, target: str) -> list[dict[str, Any]]:
    """How strongly each numeric feature relates to a numeric target."""
    if target not in df.columns or not pd.api.types.is_numeric_dtype(df[target]):
        return []

    results = []
    for col in _numeric_columns(df, exclude=(target,)):
        sub = df[[col, target]].dropna()
        if len(sub) < 10 or sub[col].nunique() < 2:
            continue
        r, p = stats.pearsonr(sub[col], sub[target])
        results.append({
            "feature": col,
            "pearson": round(float(r), 4),
            "p_value": float(f"{p:.4g}"),
            "n": int(len(sub)),
            "strength": _strength(r, (0.1, 0.3, 0.5)),
            "direction": "positive" if r > 0 else "negative",
        })

    return sorted(results, key=lambda r: -abs(r["pearson"]))


# --------------------------------------------------------------------------- #
# Group differences
# --------------------------------------------------------------------------- #

def group_differences(df: pd.DataFrame, numeric: str, group: str,
                      max_groups: int = 10) -> dict[str, Any] | None:
    """Compare a numeric column across the levels of a categorical column.

    Reports eta-squared, the share of variance in the numeric column explained by
    group membership. Unlike the test statistic it does not grow with sample
    size, so it says whether a difference actually matters.
    """
    if numeric not in df.columns or group not in df.columns:
        return None

    sub = df[[numeric, group]].dropna()
    if len(sub) < 10:
        return None

    levels = sub[group].value_counts()
    levels = levels[levels >= 5]  # groups too small to say anything about
    if not 2 <= len(levels) <= max_groups:
        return None

    sub = sub[sub[group].isin(levels.index)]
    samples = [sub.loc[sub[group] == lv, numeric].values for lv in levels.index]

    try:
        f_stat, p_value = stats.f_oneway(*samples)
    except Exception as exc:
        # A group with zero variance, or one that ended up empty after
        # filtering, makes the F statistic undefined. Returning the reason
        # rather than None lets the caller distinguish "no difference between
        # these groups" from "this comparison could not be made", which are
        # opposite conclusions that an absent result would render identical.
        return {"numeric_column": numeric, "group_column": group,
                "evaluated": False,
                "reason": f"The ANOVA could not be computed ({type(exc).__name__}); "
                          f"absence here is a failed calculation, not evidence "
                          f"that the groups are alike."}
    if not np.isfinite(f_stat):
        return {"numeric_column": numeric, "group_column": group,
                "evaluated": False,
                "reason": "The F statistic was not finite, usually because a "
                          "group has no variation; no difference is reported "
                          "either way."}

    grand_mean = sub[numeric].mean()
    ss_between = sum(len(s) * (s.mean() - grand_mean) ** 2 for s in samples)
    ss_total = float(((sub[numeric] - grand_mean) ** 2).sum())
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0

    stats_by_group = [
        {"group": str(lv), "n": int(len(s)), "mean": round(float(s.mean()), 4),
         "median": round(float(np.median(s)), 4), "std": round(float(s.std()), 4)}
        for lv, s in zip(levels.index, samples)
    ]
    stats_by_group.sort(key=lambda g: -g["mean"])

    return {
        "numeric_column": numeric,
        "group_column": group,
        "f_statistic": round(float(f_stat), 4),
        "p_value": float(f"{p_value:.4g}"),
        "eta_squared": round(eta_sq, 4),
        "effect_size": _strength(eta_sq, (0.01, 0.06, 0.14)),
        "significant": bool(p_value < 0.05),
        "n_groups": int(len(levels)),
        "groups": stats_by_group,
        "spread": {
            "highest": stats_by_group[0]["group"],
            "lowest": stats_by_group[-1]["group"],
            "mean_gap": round(stats_by_group[0]["mean"] - stats_by_group[-1]["mean"], 4),
        },
    }


def categorical_associations(df: pd.DataFrame, target: str,
                             max_features: int = 15) -> list[dict[str, Any]]:
    """Association between each categorical feature and a categorical target.

    Uses Cramer's V, a chi-squared association rescaled to 0-1 so that features
    with different numbers of categories can be compared directly.
    """
    if target not in df.columns:
        return []

    results = []
    for col in _categorical_columns(df, exclude=(target,))[:max_features]:
        sub = df[[col, target]].dropna()
        if len(sub) < 20:
            continue
        table = pd.crosstab(sub[col], sub[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        try:
            chi2, p, _, _ = stats.chi2_contingency(table)
        except Exception:
            # A contingency table that chi-square cannot evaluate (an empty
            # cell pattern, a single observed category) is skipped. The pair is
            # recorded as unevaluated rather than dropped, so a reader is not
            # left to infer that no association was found.
            results.append({
                "feature": col, "target": target, "evaluated": False,
                "reason": "The chi-square test could not be evaluated on this "
                          "contingency table, so no association is reported for "
                          "this feature. Absence here is a failed computation, "
                          "not evidence of independence.",
            })
            continue

        n = table.values.sum()
        min_dim = min(table.shape) - 1
        cramers_v = float(np.sqrt(chi2 / (n * min_dim))) if n and min_dim else 0.0

        # The rate of each target class within each feature level - this is what
        # makes the association readable in a report.
        rates = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
        results.append({
            "feature": col,
            "target": target,
            "cramers_v": round(cramers_v, 4),
            "p_value": float(f"{p:.4g}"),
            "strength": _strength(cramers_v, (0.1, 0.3, 0.5)),
            "significant": bool(p < 0.05),
            "n": int(n),
            "rates_by_level": {
                str(idx): {str(c): float(rates.loc[idx, c]) for c in rates.columns}
                for idx in rates.index[:10]
            },
        })

    # Entries that could not be evaluated carry no `cramers_v` to sort on, so
    # they are ordered last rather than indexed. Sorting on the key directly
    # raised KeyError whenever a contingency table had an empty expected cell.
    return sorted(results, key=lambda r: -r.get("cramers_v", -1.0))


# --------------------------------------------------------------------------- #
# Distributions and balance
# --------------------------------------------------------------------------- #

def distribution_summary(df: pd.DataFrame, max_columns: int = 20) -> list[dict[str, Any]]:
    """Shape of each numeric distribution, and whether it looks normal.

    Normality matters because it governs which statistical tests and models are
    appropriate. The Shapiro-Wilk test is run on a capped sample because it
    rejects normality for almost any large sample regardless of practical shape.
    """
    results = []
    for col in _numeric_columns(df)[:max_columns]:
        s = df[col].dropna()
        if len(s) < 8 or s.nunique() < 3:
            continue

        sample = s.sample(5000, random_state=42) if len(s) > 5000 else s
        try:
            _, p_normal = stats.shapiro(sample)
            normality_note = None
        except Exception as exc:
            # Shapiro-Wilk refuses constant or near-constant input. NaN alone
            # would read as "unknown" without saying why, so the reason travels
            # with it.
            p_normal = float("nan")
            normality_note = (f"The normality test could not be run "
                              f"({type(exc).__name__}), usually because the "
                              f"column barely varies.")

        skew = float(s.skew())
        results.append({
            "column": col,
            "n": int(len(s)),
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "skewness": round(skew, 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "shape": ("right-skewed" if skew > 0.5
                      else "left-skewed" if skew < -0.5
                      else "roughly symmetric"),
            "normality_p": None if np.isnan(p_normal) else float(f"{p_normal:.4g}"),
            "approximately_normal": bool(p_normal > 0.05) if not np.isnan(p_normal) else None,
            # The comment above says the reason travels with the NaN. It was
            # composed and then dropped on the floor, so a null normality result
            # reached the report with no explanation -- exactly the "unknown
            # without saying why" the comment set out to avoid.
            **({"normality_note": normality_note} if normality_note else {}),
            "suggests_transform": bool(abs(skew) > 1),
        })
    return results


def target_balance(df: pd.DataFrame, target: str) -> dict[str, Any] | None:
    """Class distribution of a categorical target.

    Imbalance decides both the models and the evaluation metrics that make sense
    downstream, so it is computed explicitly rather than left to be noticed.
    """
    if target not in df.columns:
        return None
    counts = df[target].value_counts()
    if counts.empty or len(counts) > 50:
        return None

    total = int(counts.sum())
    ratio = float(counts.iloc[0] / counts.iloc[-1]) if len(counts) > 1 else 1.0
    return {
        "target": target,
        "n_classes": int(len(counts)),
        "total": total,
        "classes": [
            {"class": str(k), "count": int(v), "percentage": round(100 * v / total, 2)}
            for k, v in counts.items()
        ],
        "majority_class": str(counts.index[0]),
        "minority_class": str(counts.index[-1]),
        "imbalance_ratio": round(ratio, 2),
        "imbalanced": bool(ratio > CONFIG.imbalance_threshold),
        "severely_imbalanced": bool(ratio > 10),
    }


def _partition_evaluated(
        results: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    """Split analyses that produced a result from those that could not.

    Several analyses here return a short `{"evaluated": False, "reason": ...}`
    dict when the statistic is undefined -- a constant column inside a group, a
    contingency table with an empty expected cell. That is deliberate: a failed
    calculation is not evidence that the groups are alike, and saying so is
    better than silently dropping the column.

    Every consumer, however, indexed `eta_squared` or `cramers_v` on whatever it
    was handed, so a single constant column raised `KeyError` and failed the
    whole EDA stage -- and with it chart selection, which reads the same lists.
    The escape hatch was correct and the contract around it was not.

    Separating them here means no consumer can be handed an incomplete record,
    while the reasons are still reported rather than lost.
    """
    evaluated = [r for r in results if r.get("evaluated") is not False]
    skipped = [r for r in results if r.get("evaluated") is False]
    return evaluated, skipped


def run_eda(df: pd.DataFrame, target: str | None = None,
            task_type: str | None = None) -> dict[str, Any]:
    """Run the analyses appropriate to this dataset and target.

    Which analyses apply depends on the task: correlating features against a
    categorical target is meaningless, as is a class-balance check on a
    continuous one. Selecting here keeps the agent's input free of results that
    could only mislead it.
    """
    results: dict[str, Any] = {
        "correlations": correlations(df, exclude=(target,) if target else ()),
        "distributions": distribution_summary(df),
        "target": target,
        "task_type": task_type,
    }

    if not target or target not in df.columns:
        return results

    if task_type == "regression":
        results["target_correlations"] = target_correlations(df, target)
        measured, skipped = _partition_evaluated([
            g for g in (
                group_differences(df, target, cat)
                for cat in _categorical_columns(df, exclude=(target,))[:10]
            ) if g
        ])
        results["group_differences"] = measured
        if skipped:
            results["group_differences_skipped"] = skipped
    elif task_type == "classification":
        results["target_balance"] = target_balance(df, target)
        associations = categorical_associations(df, target)
        measured, skipped = _partition_evaluated(associations)
        results["categorical_associations"] = measured
        if skipped:
            results["categorical_associations_skipped"] = skipped
        measured, skipped = _partition_evaluated([
            g for g in (
                group_differences(df, num, target)
                for num in _numeric_columns(df, exclude=(target,))[:10]
            ) if g
        ])
        results["group_differences"] = measured
        if skipped:
            results["group_differences_skipped"] = skipped

    return results
