"""The Data Profiling Agent: the first stage of every workflow.

The agent computes a structural profile with deterministic tools, then asks the
model to interpret it: what the dataset appears to be about, which columns are
usable features, what the analytical target most likely is, and what the rest of
the workflow should watch out for.

The division of labour is the point. Numbers are never produced by the model, and
interpretation is never produced by pandas. Every statistic in `evidence` is
computed, so the Validation Agent can later check the narrative against it.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from agents.llm import LLMError, as_name, as_names, as_target, ask_json
from state import AgentOutput, WorkflowState
from tools.profiling import profile_dataset, suggest_targets
from tools import spark_engine
from verification import hallucinated_columns

SYSTEM_PROMPT = """You are the Data Profiling Agent in a multi-agent data science system.

You receive a computed structural profile of a dataset. You never calculate
statistics yourself and you never invent values that are not in the profile.

Your job is interpretation:
- Infer what the dataset is about from column names, types and value distributions.
- Decide which columns are usable features and which are identifiers, free text,
  or otherwise unsuitable for modelling.
- Choose the most plausible analytical target and the task type it implies.
- Flag anything later stages must be careful about.

Semantic types in the profile mean:
  numeric              continuous measurement
  categorical_numeric  numeric but few distinct values, behaves as a category
  categorical          text with few distinct values
  text                 free text or high-cardinality strings
  identifier           unique per row, carries no predictive information
  datetime, boolean, empty

Respond with a single JSON object and nothing else:
{
  "dataset_description": "one or two sentences on what this data appears to represent",
  "domain": "a short domain label, e.g. healthcare, finance, transport",
  "usable_features": ["column names suitable as model inputs"],
  "excluded_columns": [{"column": "name", "reason": "why it is not a usable feature"}],
  "target_column": "the most plausible target, or null if none is apparent",
  "task_type": "classification | regression | none",
  "target_reasoning": "why that column and task type",
  "concerns": ["issues later stages should account for"],
  "narrative": "a short paragraph a human reader would understand, stating concrete numbers from the profile"
}"""

REQUIRED = (
    "dataset_description", "usable_features", "target_column",
    "task_type", "narrative",
)


def _compact_profile(profile: dict, max_columns: int = 60) -> dict:
    """Shrink the profile to what the model needs.

    Full per-column statistics on a wide dataset would dominate the prompt and
    cost tokens without improving interpretation, so distributional detail is
    summarised and very wide frames are truncated with an explicit note.
    """
    cols = []
    for c in profile["columns"][:max_columns]:
        entry = {
            "name": c["name"],
            "type": c["semantic_type"],
            "missing_pct": round(c["missing_fraction"] * 100, 1),
            "unique": c["unique"],
        }
        stats = c.get("stats") or {}
        if "mean" in stats:
            entry["range"] = [round(stats["min"], 4), round(stats["max"], 4)]
            entry["mean"] = round(stats["mean"], 4)
            if abs(stats.get("skewness", 0)) > 1:
                entry["skew"] = round(stats["skewness"], 2)
        elif "top_values" in stats:
            # Example values leave this machine: they are placed in a prompt and
            # sent to whichever provider is configured. For a category label
            # like "male" or "first class" that is harmless and genuinely helps
            # the interpretation. For an identifier or a free-text field it can
            # be a name, an email address, a patient reference or a clinical
            # note, and sending those to a third party is not something an
            # analysis should do silently.
            #
            # The profiler has already classified every column, so the decision
            # costs nothing: columns typed as identifiers or free text send a
            # description of their shape instead of their contents. Long values
            # are truncated for the same reason -- a real category label is
            # short, and anything long is either sensitive or an injection
            # payload, neither of which belongs in the prompt at full length.
            if c["semantic_type"] in PRIVATE_TYPES:
                entry["examples_withheld"] = (
                    f"{c['unique']:,} distinct values, not shown: a "
                    f"{c['semantic_type']} column's contents are not sent to "
                    f"the model.")
            else:
                entry["examples"] = [
                    _short(v["value"]) for v in stats["top_values"][:4]]
        cols.append(entry)

    compact = {
        "rows": profile["shape"]["rows"],
        "columns": profile["shape"]["columns"],
        "duplicate_rows": profile["duplicate_rows"],
        "overall_missing_pct": round(profile["missing_fraction_overall"] * 100, 2),
        "column_details": cols,
    }
    if profile["shape"]["columns"] > max_columns:
        compact["note"] = (
            f"Only the first {max_columns} of {profile['shape']['columns']} columns are shown."
        )
    return compact


#: Column types whose *contents* are never placed in a prompt. Their values are
#: the ones most likely to identify a person -- names, email addresses, record
#: references, free-text notes -- and the interpretation needs to know such a
#: column exists, not what is in it.
PRIVATE_TYPES = frozenset({"identifier", "text"})

#: Longest example value sent to the model. A genuine category label is short.
MAX_EXAMPLE_CHARS = 60


def _short(value: Any) -> str:
    """One example value, bounded."""
    text = str(value)
    return text if len(text) <= MAX_EXAMPLE_CHARS else text[:MAX_EXAMPLE_CHARS] + "..."


def _account_for_every_column(excluded: Any, features: list[str],
                              target: str | None,
                              profile: dict[str, Any]) -> list[dict[str, str]]:
    """Normalise the model's exclusions and complete them.

    Downstream, `modelling_agent` treats any column in neither `usable_features`
    nor `excluded_columns` as one derived after profiling, and reinstates it as
    a model feature. That inference is only sound if profiling lists every
    column it saw. Two ways it did not:

    - The model names the column under whatever key it likes (`name`, `col`, a
      bare string). Reading only `column` yielded `None` for the others, so the
      real name never entered the set and an identifier was reinstated -- a row
      id fitted as a predictor, reported as a "feature derived after profiling".
    - The model can simply omit a column it judged unusable but did not list.

    Both are properties of model output, not of the data, so neither is fixable
    by prompting. The invariant is restored here instead.
    """
    known = {c["name"] for c in profile["columns"]}
    normalised: list[dict[str, str]] = []
    seen: set[str] = set()

    for entry in (excluded or []):
        if isinstance(entry, str):
            name, reason = entry, "Excluded by the profiling interpretation."
        elif isinstance(entry, dict):
            name = next((str(entry[k]) for k in ("column", "name", "col", "feature")
                         if entry.get(k)), None)
            reason = str(entry.get("reason", "Excluded by the profiling "
                                             "interpretation."))
        else:
            continue
        if not name or name in seen or name not in known:
            continue
        seen.add(name)
        normalised.append({"column": name, "reason": reason})

    # Anything profiling saw but neither kept nor listed. Recording it is the
    # difference between "profiling rejected this" and "this did not exist
    # when profiling ran", which are opposite instructions downstream.
    accounted = set(features) | seen | ({target} if target else set())
    types = {c["name"]: c["semantic_type"] for c in profile["columns"]}
    for name in known - accounted:
        normalised.append({
            "column": name,
            "reason": (f"Present at profiling and not selected as a feature "
                       f"(semantic type: {types.get(name, 'unknown')})."),
        })
    return normalised


class ProfilingAgent(Agent):
    name = "profiling"
    description = (
        "Computes dataset structure, types and distributions, and interprets what "
        "the data represents, which columns are usable, and the likely target."
    )
    requires = ()

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []

        # 1. Deterministic computation. When the orchestrator selected the
        #    distributed engine, the profile is computed across partitions on
        #    the complete dataset rather than on the in-memory sample. Both
        #    engines return the same contract, so nothing below this line
        #    changes; only the scale the numbers describe does.
        if state.spark_df is not None:
            profile = spark_engine.profile_dataset(state.spark_df)
            warnings.append(
                f"Profiled on Spark across the full {profile['shape']['rows']:,} rows; "
                f"later stages use an in-memory sample of {len(df):,} rows."
            )
        else:
            profile = profile_dataset(df)
        candidates = suggest_targets(profile, state.user_goal)

        if profile["sampled"]:
            warnings.append(
                "Distributional statistics were computed on a sample; shape, missing "
                "values and duplicates reflect the full dataset."
            )

        # 2. Model interpretation.
        user_prompt = (
            f"Dataset name: {state.dataset_name}\n"
            f"{f'Stated analysis goal: {state.user_goal}' if state.user_goal else ''}\n\n"
            f"Computed profile:\n{json.dumps(_compact_profile(profile), indent=2)}\n\n"
            f"Target columns suggested by heuristics (advisory only, you may disagree):\n"
            f"{json.dumps(candidates, indent=2)}\n\n"
            "Interpret this profile."
        )

        try:
            interp = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            # Falling back to heuristics keeps the workflow alive without the
            # model, which also makes the framework usable when no model is
            # reachable at all.
            top = candidates[0] if candidates else None
            interp = {
                "dataset_description": f"{profile['shape']['rows']} rows, "
                                       f"{profile['shape']['columns']} columns.",
                "domain": "unknown",
                "usable_features": [
                    c["name"] for c in profile["columns"]
                    if c["semantic_type"] not in ("identifier", "empty", "text")
                ],
                # The columns this fallback deliberately dropped, recorded as
                # exclusions rather than left implicit. An empty list here was
                # read downstream as "profiling excluded nothing", so the
                # modelling agent treated every identifier and free-text column
                # as a feature derived after profiling and reinstated it -- a
                # row id fitted as a predictor, on exactly the runs where the
                # model was unavailable and nobody was writing narrative to
                # notice. What a stage rejected is part of its result.
                "excluded_columns": [
                    {"column": c["name"],
                     "reason": f"Heuristically classified as {c['semantic_type']}, "
                               f"which cannot serve as a model feature."}
                    for c in profile["columns"]
                    if c["semantic_type"] in ("identifier", "empty", "text")
                ],
                "target_column": top["column"] if top else None,
                "task_type": top["task"] if top else "none",
                "target_reasoning": "Heuristic fallback; no model interpretation available.",
                "concerns": [],
                "narrative": "Structural profile computed; interpretation unavailable.",
            }
            warnings.append(f"Model interpretation unavailable, used heuristics. ({exc})")
            status = "degraded"

        # 3. Validate the model's choices against the real dataframe. A target or
        #    feature that does not exist would break every downstream stage, so
        #    hallucinated column names are corrected here rather than propagated.
        #    The names are coerced first: the model has returned them as
        #    objects and as one-element lists, which are unhashable and raised
        #    inside the membership test below.
        columns = list(df.columns)

        proposed = as_names(interp.get("usable_features"))
        invented = hallucinated_columns(proposed, columns)
        features = [c for c in proposed if c not in invented]
        if invented:
            warnings.append(f"Ignored feature names not present in the data: {invented}.")

        target = as_target(interp.get("target_column"))
        if target is not None and hallucinated_columns([target], columns):
            warnings.append(
                f"Proposed target '{target}' is not a column in the dataset; "
                "falling back to the top heuristic candidate."
            )
            target = candidates[0]["column"] if candidates else None
            interp["task_type"] = candidates[0]["task"] if candidates else "none"

        task = as_name(interp.get("task_type")) or "none"
        if task not in ("classification", "regression", "none"):
            warnings.append(f"Unrecognised task type '{task}'; recorded as none.")
            task = "none"
        if target is None:
            task = "none"

        # A target must not also be listed as an input feature.
        if target in features:
            features.remove(target)

        # 4. Publish decisions the rest of the workflow depends on.
        state.target_column = target
        state.task_type = task

        findings = {
            "shape": profile["shape"],
            "dataset_description": interp.get("dataset_description", ""),
            "domain": interp.get("domain", "unknown"),
            "usable_features": features,
            "excluded_columns": _account_for_every_column(
                interp.get("excluded_columns"), features, target, profile),
            "target_column": target,
            "task_type": task,
            "target_reasoning": interp.get("target_reasoning", ""),
            "concerns": interp.get("concerns", []),
            "column_types": {
                c["name"]: c["semantic_type"] for c in profile["columns"]
            },
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=interp.get("narrative", ""),
            evidence={"profile": profile, "target_candidates": candidates},
            warnings=warnings,
        )
