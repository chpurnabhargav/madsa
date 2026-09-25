"""Data quality assessment and cleaning.

Detection and repair are deliberately separate. The Quality Agent reasons over
`assess_quality` findings and decides which repairs to authorise; `apply_cleaning`
then executes only those, recording every action so the final report can state
exactly what was changed and why.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from config import CONFIG

SEVERITY_ORDER = {"critical": 3, "warning": 2, "info": 1}


def _issue(kind: str, severity: str, column: str | None, detail: str, **extra: Any) -> dict:
    return {"kind": kind, "severity": severity, "column": column, "detail": detail, **extra}


def detect_outliers_iqr(s: pd.Series, k: float = 1.5) -> dict[str, Any]:
    """Flag outliers by the interquartile-range rule.

    IQR is used rather than a z-score because it does not assume normality and
    is not itself dragged by the outliers it is meant to find.
    """
    non_null = s.dropna()
    if len(non_null) < 4:
        return {"count": 0, "fraction": 0.0, "lower": None, "upper": None}
    q1, q3 = non_null.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    mask = (non_null < lower) | (non_null > upper)
    return {
        "count": int(mask.sum()),
        "fraction": round(float(mask.sum()) / len(non_null), 4),
        "lower": float(lower),
        "upper": float(upper),
    }


def assess_quality(df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    """Detect quality problems without modifying the dataframe."""
    issues: list[dict] = []

    if profile["duplicate_rows"] > 0:
        issues.append(
            _issue(
                "duplicate_rows",
                "warning" if profile["duplicate_fraction"] < 0.05 else "critical",
                None,
                f"{profile['duplicate_rows']} fully duplicated rows "
                f"({profile['duplicate_fraction']:.1%} of the dataset).",
                count=profile["duplicate_rows"],
            )
        )

    for col in profile["columns"]:
        name, sem = col["name"], col["semantic_type"]
        s = df[name]

        # Missing values.
        mf = col["missing_fraction"]
        if mf >= CONFIG.unusable_missing_fraction:
            issues.append(
                _issue("missing_values", "critical", name,
                       f"{mf:.1%} missing - likely unusable as a feature.",
                       missing_fraction=mf))
        elif mf > 0.05:
            issues.append(
                _issue("missing_values", "warning", name,
                       f"{mf:.1%} missing - imputation required before modelling.",
                       missing_fraction=mf))
        elif mf > 0:
            issues.append(
                _issue("missing_values", "info", name,
                       f"{mf:.1%} missing.", missing_fraction=mf))

        # Constant columns carry no information.
        if col["unique"] <= 1:
            issues.append(
                _issue("constant_column", "warning", name,
                       "Column holds a single value and cannot inform any model."))

        # High-cardinality categoricals blow up one-hot encoding.
        if sem == "categorical" and col["unique"] > 50:
            issues.append(
                _issue("high_cardinality", "warning", name,
                       f"{col['unique']} distinct categories - encoding needs care.",
                       unique=col["unique"]))

        # Outliers and skew, numeric columns only.
        if sem == "numeric" and pd.api.types.is_numeric_dtype(s):
            out = detect_outliers_iqr(s)
            if out["fraction"] > 0.01:
                issues.append(
                    _issue("outliers", "warning" if out["fraction"] > 0.05 else "info",
                           name,
                           f"{out['count']} values ({out['fraction']:.1%}) outside the IQR fence "
                           f"[{out['lower']:.2f}, {out['upper']:.2f}].",
                           **out))
            skew = (col.get("stats") or {}).get("skewness", 0.0)
            if abs(skew) > 2:
                issues.append(
                    _issue("skewed_distribution", "info", name,
                           f"Strongly skewed (skewness {skew:.2f}); a log or power "
                           "transform may help linear models.", skewness=skew))

        # Numbers stored as text prevent any numeric analysis. Identifier-typed
        # columns are included: a near-unique column of numeric strings is still
        # stored in the wrong type.
        if sem in ("categorical", "text", "identifier") and s.dtype == object:
            values = s.dropna().astype(str).str.strip()
            # Placeholders written by an exporter are missing values, not
            # evidence that the column holds non-numeric content, so they are
            # excluded before judging whether the rest is numeric. A column
            # exported with "nan" or "NULL" in place of blanks is exactly the
            # case this check exists to find.
            null_like = {"", "nan", "none", "null", "na", "n/a", "-", "?"}
            real = values[~values.str.lower().isin(null_like)]
            coerced = pd.to_numeric(real, errors="coerce")
            if len(real) > 0 and coerced.notna().sum() > 0.9 * len(real):
                placeholder_note = (
                    f" {len(values) - len(real)} placeholder value(s) are also present."
                    if len(real) < len(values) else ""
                )
                issues.append(
                    _issue("numeric_stored_as_text", "warning", name,
                           "Values are numeric but stored as text; convert before "
                           f"analysis.{placeholder_note}"))

        # Whitespace/case variants inflate category counts.
        if sem == "categorical" and s.dtype == object:
            norm = s.dropna().astype(str).str.strip().str.lower()
            if norm.nunique() < s.dropna().nunique():
                issues.append(
                    _issue("inconsistent_categories", "warning", name,
                           f"{s.dropna().nunique() - norm.nunique()} categories differ only by "
                           "case or surrounding whitespace."))

    counts = {"critical": 0, "warning": 0, "info": 0}
    for i in issues:
        counts[i["severity"]] += 1

    # A single headline number, used by the orchestrator to decide whether the
    # cleaning stage is worth running at all.
    penalty = counts["critical"] * 15 + counts["warning"] * 5 + counts["info"] * 1
    score = max(0, 100 - penalty)

    return {
        "quality_score": score,
        "issue_counts": counts,
        "issues": sorted(issues, key=lambda i: -SEVERITY_ORDER[i["severity"]]),
    }


def apply_cleaning(df: pd.DataFrame,
                   actions: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[dict]]:
    """Apply an authorised list of cleaning actions.

    Each action is `{"action": ..., "column": ...}`. Returns the cleaned frame
    and a log of what actually happened, including failures, so nothing silently
    does or does not occur.
    """
    out = df.copy()
    log: list[dict] = []

    for act in actions:
        kind, col = act.get("action"), act.get("column")
        rows_before = out.shape[0]
        try:
            if kind == "drop_duplicates":
                out = out.drop_duplicates()
                log.append({"action": kind, "column": None, "status": "applied",
                            "detail": f"Removed {rows_before - out.shape[0]} duplicate rows."})

            elif kind == "drop_column":
                if col not in out.columns:
                    raise KeyError(col)
                out = out.drop(columns=[col])
                log.append({"action": kind, "column": col, "status": "applied",
                            "detail": f"Dropped column {col}."})

            elif kind == "impute_median":
                med = out[col].median()
                n = int(out[col].isna().sum())
                out[col] = out[col].fillna(med)
                log.append({"action": kind, "column": col, "status": "applied",
                            "detail": f"Filled {n} missing values with median {med:.4g}."})

            elif kind == "impute_mode":
                mode = out[col].mode()
                if mode.empty:
                    raise ValueError("column has no mode")
                n = int(out[col].isna().sum())
                out[col] = out[col].fillna(mode.iloc[0])
                log.append({"action": kind, "column": col, "status": "applied",
                            "detail": f"Filled {n} missing values with mode {mode.iloc[0]}."})

            elif kind == "to_numeric":
                out[col] = pd.to_numeric(out[col], errors="coerce")
                log.append({"action": kind, "column": col, "status": "applied",
                            "detail": f"Converted {col} to a numeric type."})

            elif kind == "normalise_categories":
                # `astype(str)` renders NaN as the literal string "nan", which
                # then survives as a real category: the column reports zero
                # missing afterwards while the report still states the missing
                # fraction that was measured, and "nan" is one-hot encoded as a
                # level of its own. Missingness is a fact about the data, not a
                # formatting inconsistency for this action to tidy away -- the
                # sibling `to_numeric` action preserves it with
                # `errors="coerce"` for the same reason.
                present = out[col].notna()
                out.loc[present, col] = (
                    out.loc[present, col].astype(str).str.strip().str.lower())
                log.append({"action": kind, "column": col, "status": "applied",
                            "detail": f"Trimmed and lowercased categories in "
                                      f"{col}, leaving missing values missing."})

            elif kind == "clip_outliers":
                fence = detect_outliers_iqr(out[col])
                if fence["lower"] is None:
                    raise ValueError("too few values to compute an IQR fence")
                out[col] = out[col].clip(fence["lower"], fence["upper"])
                # The fence is recorded, not only described, because a model
                # fitted on clipped values must see new rows clipped to the
                # same fence (`replay_cleaning`).
                log.append({"action": kind, "column": col, "status": "applied",
                            "lower": float(fence["lower"]),
                            "upper": float(fence["upper"]),
                            "detail": f"Clipped {col} to "
                                      f"[{fence['lower']:.4g}, {fence['upper']:.4g}]."})
            else:
                log.append({"action": kind, "column": col, "status": "skipped",
                            "detail": f"Unknown action {kind}."})

        except Exception as exc:  # a bad suggestion must not kill the workflow
            log.append({"action": kind, "column": col, "status": "failed",
                        "detail": f"{type(exc).__name__}: {exc}"})

    return out, log


#: Repairs that change a column's type or values, and so must be repeated on
#: every new row a saved model scores. Removing duplicate rows or a whole
#: column leaves nothing to repeat, and imputation happens inside the model.
REPLAYABLE_ACTIONS = frozenset({"to_numeric", "normalise_categories", "clip_outliers"})


def replay_cleaning(df: pd.DataFrame, steps: list[dict[str, Any]]) -> pd.DataFrame:
    """Repeat recorded repairs on new rows, with the parameters fixed at the time.

    A model fitted on repaired data expects repaired input. Raw rows given to
    it without these steps arrive with numbers still stored as text and
    categories in their original case -- values the fitted encoder has never
    seen, which it silently encodes as all zeros rather than rejecting.
    """
    out = df.copy()
    for step in steps:
        col, kind = step.get("column"), step.get("action")
        if col not in out.columns:
            continue
        if kind == "to_numeric":
            out[col] = pd.to_numeric(out[col], errors="coerce")
        elif kind == "normalise_categories":
            present = out[col].notna()
            out[col] = out[col].astype(object)
            out.loc[present, col] = (
                out.loc[present, col].astype(str).str.strip().str.lower())
        elif kind == "clip_outliers":
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(
                step["lower"], step["upper"])
    return out
