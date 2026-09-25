"""Canned responses that let the whole framework run with no model at all.

Extracted from ``llm.py``, where they had grown to 446 of its 1,251 lines --
more than a third of a module whose subject is talking to language-model
providers, given over to pretending to be one. The two concerns are unrelated:
nothing here knows about API keys, retries, rate limits or token accounting.

The mock backend is not a stub. It reads the structural summary in each prompt
and responds to what the dataset actually contains, so planning genuinely adapts
offline and the verification layer has real output to check. Several responses
are deliberately written to be *correct rather than agreeable* -- the feature
mock reports that derived features did not help when the measured gain says so,
because a mock that always claimed success would let the guard against
overstating a rejected result pass without ever being exercised.

This is what makes the project reproducible by someone with no API key and no
budget: every deterministic result -- model scores, permutation tests, the
engine comparison, all twelve verification mechanisms -- is reachable from here.
"""
from __future__ import annotations

import json
from typing import Any, Callable


def _mock_plan(prompt: str) -> dict:
    """Plan from the structural summary in the prompt, not from a fixed list.

    Reading the summary keeps the mock's planning genuinely responsive to the
    dataset, so adaptive behaviour can be exercised offline.
    """
    plan = ["profiling", "quality", "features", "eda", "multivariate",
            "visualization", "modelling", "report", "validation"]
    skipped = []
    try:
        summary = json.loads(prompt[prompt.index("{"):prompt.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        summary = {}

    if summary.get("missing_cells") == 0 and summary.get("duplicate_rows") == 0:
        plan.remove("quality")
        skipped.append({"stage": "quality",
                        "reason": "No missing values or duplicate rows were found."})
    if summary.get("rows", 0) and summary["rows"] < 30:
        plan.remove("modelling")
        skipped.append({"stage": "modelling",
                        "reason": "Too few rows to validate a model."})
    if summary.get("rows", 0) and summary["rows"] < 50 and "features" in plan:
        plan.remove("features")
        skipped.append({"stage": "features",
                        "reason": "Too few rows to measure whether a derived "
                                  "feature contributes."})

    types = summary.get("semantic_type_counts", {})
    n_numeric = types.get("numeric", 0) + types.get("categorical_numeric", 0)
    if n_numeric < 3:
        plan.remove("multivariate")
        skipped.append({"stage": "multivariate",
                        "reason": "Fewer than three numeric columns; PCA and "
                                  "clustering would not be informative."})

    return {"plan": plan, "skipped": skipped,
            "reasoning": "Mock coordinator: stages selected from the structural summary."}


def _json_blocks(prompt: str) -> list[Any]:
    """Every JSON value embedded in a prompt, in order of appearance.

    The mock handlers read the computed statistics the agents put into their own
    prompts. Deriving decisions from real numbers rather than returning fixed
    text means an offline run exercises the same downstream paths a live one
    does, which is what makes the offline evaluation meaningful.
    """
    blocks, depth, start = [], 0, None
    in_string = escaped = False
    for i, ch in enumerate(prompt):
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            if depth == 0:
                start = i
            depth += 1
        elif ch in "}]" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    blocks.append(json.loads(prompt[start:i + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    return blocks


def _mock_profile(prompt: str) -> dict:
    """Interpret a profile using the heuristics the tools already computed."""
    profile: dict = {}
    candidates: list = []
    for block in _json_blocks(prompt):
        if isinstance(block, dict) and "column_details" in block:
            profile = block
        elif isinstance(block, list) and block and isinstance(block[0], dict) \
                and "task" in block[0]:
            candidates = block

    columns = profile.get("column_details", [])
    usable = [
        c["name"] for c in columns
        if c.get("type") not in ("identifier", "empty", "text")
        and c.get("missing_pct", 0) < 60
    ]
    excluded = [
        {"column": c["name"], "reason": f"semantic type '{c.get('type')}', "
                                        f"{c.get('missing_pct', 0)}% missing"}
        for c in columns if c["name"] not in usable
    ]

    top = candidates[0] if candidates else None
    target = top["column"] if top else None
    task = top["task"] if top else "none"
    if target in usable:
        usable.remove(target)

    return {
        "dataset_description": f"Mock backend: {profile.get('rows', '?')} rows and "
                               f"{profile.get('columns', '?')} columns.",
        "domain": "unknown",
        "usable_features": usable,
        "excluded_columns": excluded,
        "target_column": target,
        "task_type": task,
        "target_reasoning": "Selected from the heuristic target candidates.",
        "concerns": [],
        "narrative": f"Profile computed for {profile.get('rows', '?')} rows across "
                     f"{profile.get('columns', '?')} columns.",
    }


def _mock_quality(prompt: str) -> dict:
    """Authorise proportionate repairs from the detected issues."""
    issues = next(
        (b for b in _json_blocks(prompt)
         if isinstance(b, list) and b and isinstance(b[0], dict) and "kind" in b[0]),
        [],
    )
    actions, limitations = [], []
    for issue in issues:
        kind, col, severity = issue.get("kind"), issue.get("column"), issue.get("severity")
        if kind == "duplicate_rows":
            actions.append({"action": "drop_duplicates", "column": None,
                            "reason": "Exact duplicate rows carry no information."})
        elif kind == "constant_column":
            actions.append({"action": "drop_column", "column": col,
                            "reason": "A single value cannot inform any model."})
        elif kind == "numeric_stored_as_text":
            actions.append({"action": "to_numeric", "column": col,
                            "reason": "Numeric values stored as text."})
        elif kind == "inconsistent_categories":
            actions.append({"action": "normalise_categories", "column": col,
                            "reason": "Categories differ only by case or whitespace."})
        elif kind == "missing_values" and severity == "critical":
            actions.append({"action": "drop_column", "column": col,
                            "reason": "Too sparse to impute without fabricating data."})
        elif kind == "missing_values" and severity == "warning":
            # Numeric columns take the median; the guard rejects a wrong choice.
            actions.append({"action": "impute_median", "column": col,
                            "reason": "Moderate missingness; median is robust."})
            limitations.append(f"{col} has missing values; they are imputed inside "
                               f"each model-fitting fold, which reduces its variance.")
        elif kind == "outliers" and severity == "warning":
            limitations.append(f"{col} retains extreme values that may be genuine.")
    return {"actions": actions, "declined": [], "limitations": limitations,
            "narrative": f"Mock backend authorised {len(actions)} cleaning action(s) "
                         f"from {len(issues)} detected issue(s)."}


def _mock_eda(prompt: str) -> dict:
    """State the strongest computed effects as insights."""
    stats = next((b for b in _json_blocks(prompt)
                  if isinstance(b, dict) and "task_type" in b), {})
    insights = []

    for a in (stats.get("categorical_vs_target") or [])[:3]:
        v = a.get("cramers_v", 0)
        insights.append({
            "insight": f"{a['feature']} is associated with the target "
                       f"({a.get('strength', 'unknown')} association).",
            "evidence": f"Cramer's V = {v}, p = {a.get('p_value')}",
            "importance": "high" if v >= 0.3 else "medium" if v >= 0.1 else "low",
        })
    for g in (stats.get("group_differences") or [])[:3]:
        eta = g.get("eta_squared", 0)
        if eta < 0.01:
            continue
        insights.append({
            "insight": f"{g['numeric']} differs across levels of {g['grouped_by']} "
                       f"({g.get('effect_size')} effect).",
            "evidence": f"eta-squared = {eta}, p = {g.get('p_value')}",
            "importance": "high" if eta >= 0.06 else "medium",
        })
    for t in (stats.get("numeric_vs_target") or [])[:3]:
        r = abs(t.get("pearson", 0))
        insights.append({
            "insight": f"{t['feature']} is {t.get('direction')}ly associated with "
                       f"the target ({t.get('strength')}).",
            "evidence": f"Pearson r = {t.get('pearson')}, p = {t.get('p_value')}",
            "importance": "high" if r >= 0.5 else "medium" if r >= 0.3 else "low",
        })

    relationships = [
        f"{c['pair'][0]} and {c['pair'][1]} are correlated "
        f"(Pearson {c['pearson']})." for c in (stats.get("feature_correlations") or [])[:3]
    ]
    notes = [
        f"{d['column']} is {d['shape']} (skewness {d['skewness']})."
        for d in (stats.get("distributions") or []) if d.get("suggests_transform")
    ][:3]

    return {
        "key_insights": insights,
        "relationships": relationships,
        "distribution_notes": notes,
        "data_observations": [],
        "modelling_implications": [],
        "narrative": f"Mock backend ranked {len(insights)} finding(s) by effect size.",
    }


def _mock_visualization(prompt: str) -> dict:
    """Choose charts from the strongest effects reported in the prompt."""
    ctx = next((b for b in _json_blocks(prompt)
                if isinstance(b, dict) and "strongest_effects" in b), {})
    target = ctx.get("target")
    types = ctx.get("column_types", {})
    charts: list[dict] = []

    if target and ctx.get("task_type") == "classification":
        charts.append({"kind": "target_balance", "target": target,
                       "reason": "Establishes the class balance of the target."})

    for column, effect in list((ctx.get("strongest_effects") or {}).items())[:4]:
        kind = types.get(column)
        if not target:
            break
        if kind in ("categorical", "categorical_numeric", "boolean"):
            charts.append({"kind": "grouped_rate", "feature": column, "target": target,
                           "reason": f"Association with the target (effect {effect})."})
        elif kind == "numeric" and ctx.get("task_type") == "classification":
            charts.append({"kind": "box", "numeric": column, "group": target,
                           "reason": f"Distribution differs by target (effect {effect})."})
        elif kind == "numeric":
            charts.append({"kind": "scatter", "x": column, "y": target,
                           "reason": f"Relationship with the target (effect {effect})."})

    charts.append({"kind": "heatmap", "reason": "Overview of numeric correlations."})
    return {"charts": charts[:6],
            "narrative": f"Mock backend selected {min(len(charts), 6)} chart(s) from "
                         f"the strongest computed effects."}


def _mock_modelling(prompt: str) -> dict:
    """Recommend from the measured cross-validation results."""
    results = next((b for b in _json_blocks(prompt)
                    if isinstance(b, dict) and "best_model" in b), {})
    beats = results.get("beats_baseline")
    metric = results.get("primary_metric", "score")
    importance = (results.get("feature_importance") or {}).get("ranked", [])
    informative = [f["feature"] for f in importance
                   if f.get("distinguishable_from_noise")]

    if beats:
        return {
            "recommended_model": results.get("best_model"),
            "recommendation_reason": f"Highest cross-validated {metric} at "
                                     f"{results.get('best_score')}.",
            "expected_performance": f"{metric} of approximately {results.get('best_score')}.",
            "evaluation_metrics": [metric],
            "important_features": informative[:5],
            "signal_assessment": "strong" if (results.get("lift_over_baseline") or 0) > 0.2
                                 else "moderate",
            "improvements": ["Tune the model's hyperparameters.",
                             "Engineer features from the excluded columns."],
            "caveats": ["Baseline models were untuned."],
            "narrative": f"Mock backend: {results.get('best_model')} reached a "
                         f"cross-validated {metric} of {results.get('best_score')}.",
        }
    return {
        "recommended_model": None,
        "recommendation_reason": "No candidate beat the baseline by a margin "
                                 "distinguishable from chance.",
        "expected_performance": "no better than chance",
        "evaluation_metrics": [metric],
        "important_features": [],
        "signal_assessment": "none",
        "improvements": ["Collect features more closely related to the target."],
        "caveats": ["The available features do not predict this target."],
        "narrative": "Mock backend: no model performed better than chance on this target.",
    }


def _mock_report(prompt: str) -> dict:
    """Assemble a report from the stage narratives already in the prompt."""
    material = next((b for b in _json_blocks(prompt)
                     if isinstance(b, dict) and "stages_completed" in b), {})
    sections, takeaways, limitations = [], [], []
    headings = {
        "profiling": "Dataset Overview", "quality": "Data Quality",
        "features": "Feature Engineering",
        "eda": "Exploratory Analysis", "multivariate": "Feature Space Structure",
        "visualization": "Visualizations", "modelling": "Modelling",
    }
    for stage, heading in headings.items():
        block = material.get(stage) or {}
        if narrative := block.get("narrative"):
            sections.append({"heading": heading, "body": narrative})

    for insight in ((material.get("eda") or {}).get("findings") or {}).get("key_insights", []):
        # Guarded as well as fixed at source: a trimmed list can carry an
        # omission marker, and a mock that crashes fails the stage it stands in
        # for.
        if isinstance(insight, dict) and insight.get("importance") == "high":
            takeaways.append(insight.get("insight", ""))

    quality = (material.get("quality") or {}).get("findings") or {}
    modelling = (material.get("modelling") or {}).get("findings") or {}
    limitations.extend(quality.get("limitations", []) or [])
    limitations.extend(modelling.get("caveats", []) or [])
    if modelling.get("beats_baseline") is False:
        limitations.append(
            "No model performed better than chance, so the available features do "
            "not predict the target.")

    return {
        "title": f"Analytical Report: {material.get('dataset', 'dataset')}",
        "executive_summary": (
            f"Mock backend report covering {material.get('rows', '?')} rows and "
            f"{material.get('columns', '?')} columns. "
            + (takeaways[0] if takeaways else
               "No high-importance findings were established.")
        ),
        "sections": sections,
        "key_takeaways": takeaways,
        "limitations": limitations,
        "recommended_next_steps": modelling.get("improvements", []) or [],
    }


def _mock_multivariate(prompt: str) -> dict:
    """State the computed structure of the feature space."""
    ctx = next((b for b in _json_blocks(prompt)
                if isinstance(b, dict) and ("pca" in b or "clustering" in b)), {})
    pca = ctx.get("pca") or {}
    clustering = ctx.get("clustering") or {}
    vif = ctx.get("variance_inflation") or {}
    mi = ctx.get("mutual_information") or {}

    redundancy = []
    if pca:
        if (needed := pca.get("components_for_90pct_variance")) is not None:
            redundancy.append(
                f"{needed} of {pca.get('n_features')} "
                f"components retain 90% of the variance.")
        elif reason := pca.get("components_not_established"):
            # Stating "None of 20 components retain 90% of the variance" would
            # be worse than saying nothing; the reason is the actual finding.
            redundancy.append(reason)
    if groups := (ctx.get("correlation_groups") or {}).get("redundant_groups"):
        redundancy.append(f"Correlated feature groups: {groups}.")

    cluster_findings = []
    if clustering:
        if clustering.get("meaningful"):
            cluster_findings.append(
                f"k-means found {clustering.get('best_k')} groups with a silhouette "
                f"of {clustering.get('silhouette')}.")
        else:
            cluster_findings.append(
                f"The data does not separate into natural groups; the best silhouette "
                f"was only {clustering.get('silhouette')}.")

    return {
        "structure_summary": f"Mock backend: {pca.get('n_features', 'several')} numeric "
                             f"features analysed.",
        "redundancy_findings": redundancy,
        "cluster_findings": cluster_findings,
        "interpretation_warnings": [
            f"{f} is strongly predictable from the other features."
            for f in vif.get("severe", [])],
        "non_linear_signals": (mi.get("informative") or [])[:5],
        "modelling_implications": [],
        "narrative": "Mock backend: multivariate structure computed without interpretation.",
    }


def _mock_features(prompt: str) -> dict:
    """Report the measured gain, and only claim a benefit when there was one.

    The mock is written to be *correct* rather than agreeable, because a mock
    that always reports success would let the agent's guard against overstating
    a rejected feature set pass without ever being exercised.
    """
    adopted = "ADOPTED" in prompt and "NOT adopted" not in prompt
    gain = None
    for block in _json_blocks(prompt):
        if isinstance(block, dict) and "gain" in block:
            gain = block.get("gain")
    features = []
    for block in _json_blocks(prompt):
        if isinstance(block, list):
            features = [b.get("feature") for b in block
                        if isinstance(b, dict) and b.get("feature")]
            break

    if adopted:
        return {
            "summary": (f"Derived features improved the cross-validated score by "
                        f"{gain} and were adopted."),
            "useful_features": features[:3],
            "why_it_helped_or_not": [
                "The derived columns expressed structure the raw columns only implied."],
            "declined_rationale": [
                "Target encoding was declined because computing it over the whole "
                "frame leaks the validation fold into the training features."],
            "narrative": (f"Mock backend: {len(features)} derived feature(s) changed "
                          f"the score by {gain}, which justified adopting them."),
        }
    return {
        "summary": (f"Derived features did not improve the score (change: {gain}) "
                    f"and were not adopted."),
        "useful_features": [],
        "why_it_helped_or_not": [
            "The change was within resampling noise, so it is not evidence of "
            "improvement.",
            "Tree ensembles are invariant to monotonic transforms, so a log "
            "transform adds a dimension without adding information."],
        "declined_rationale": [
            "Target encoding was declined because computing it over the whole "
            "frame leaks the validation fold into the training features."],
        "narrative": (f"Mock backend: the derived features changed the score by "
                      f"{gain}, which does not justify keeping them. The original "
                      f"columns were retained."),
    }


DEFAULT_MOCKS: dict[str, Callable[[str], dict]] = {
    "coordinator": _mock_plan,
    "profiling": _mock_profile,
    "quality": _mock_quality,
    "features": _mock_features,
    "eda": _mock_eda,
    "multivariate": _mock_multivariate,
    "visualization": _mock_visualization,
    "modelling": _mock_modelling,
    "report": _mock_report,
    "validation": lambda p: {
        "issues": [], "verdict": "sound", "confidence_in_analysis": "low",
        "narrative": "Mock backend: automated checks only.",
    },
}
