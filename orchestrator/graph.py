"""The orchestration layer: plan a workflow, execute it, and revise it.

This is the component the project contributes. The agents are specialists; the
orchestrator decides which of them should run for a given dataset, in what
order, and whether the result is good enough to deliver.

Three mechanisms make the orchestration adaptive rather than a fixed pipeline:

* Planning. The coordinator inspects a cheap structural profile before any
  analysis and selects the stages that suit the dataset. A dataset with no
  supervised target does not need a modelling stage, and one with no quality
  problems does not need a cleaning stage. Planning is constrained by declared
  dependencies, so a plan that could not execute is repaired before it runs.
* Conditional execution. Some stages become unnecessary because of what an
  earlier stage found, which cannot be known at planning time. Those decisions
  are made as the workflow runs.
* Validated revision. The validation stage can send the workflow back. A report
  that misstates the analysis is re-composed with the validator's objections in
  hand, rather than delivered with a warning attached.

The graph is built with LangGraph so that the workflow is an explicit object
that can be inspected and diagrammed rather than implied by control flow.
"""
from __future__ import annotations

import json
import time
from typing import Any, Callable

import pandas as pd
from langgraph.graph import END, StateGraph

from agents.eda_agent import EDAAgent
from agents.feature_agent import FeatureAgent
from agents.llm import USAGE, LLMError, ask_json
from agents.modelling_agent import ModellingAgent
from agents.multivariate_agent import MultivariateAgent
from agents.profiling_agent import ProfilingAgent
from agents.quality_agent import QualityAgent
from agents.report_agent import ReportAgent
from agents.validation_agent import ValidationAgent
from agents.visualization_agent import VisualizationAgent
from config import CONFIG
from state import WorkflowState
from tools.profiling import profile_dataset
from verification import firings, reset_firings

# The agents available to the coordinator, with their declared dependencies.
REGISTRY = {
    "profiling": ProfilingAgent,
    "quality": QualityAgent,
    "features": FeatureAgent,
    "eda": EDAAgent,
    "multivariate": MultivariateAgent,
    "visualization": VisualizationAgent,
    "modelling": ModellingAgent,
    "report": ReportAgent,
    "validation": ValidationAgent,
}

# Stages that must always run: without profiling nothing else can be planned,
# and without a report and its validation the workflow has produced no output.
MANDATORY = ("profiling", "report", "validation")

# The order stages must respect when they do run. The coordinator may omit a
# stage but may not reorder the ones it keeps, because a later stage's input is
# an earlier stage's output.
CANONICAL_ORDER = [
    "profiling", "quality", "features", "eda", "multivariate", "visualization",
    "modelling", "report", "validation",
]

PLANNER_SYSTEM = """You are the Coordinator Agent in a multi-agent data science system.

You receive a brief structural summary of a dataset, produced before any analysis
has run. You decide which analysis stages are worth executing.

Available stages:
  profiling      structure, types and target identification. Always required.
  quality        detects and repairs data quality problems.
  features       derives additional predictors from existing columns and measures
                 whether they improve prediction, keeping them only if they do.
                 Needs a supervised target; skip when there is none.
  eda            statistical exploration: correlations, group differences, distributions.
  multivariate   PCA, clustering and multicollinearity across the whole feature space.
                 Worth running when there are several numeric columns; skip when
                 there are fewer than three or too few rows.
  visualization  generates charts of the findings.
  modelling      cross-validates baseline models. Only meaningful with a supervised target.
  report         composes the final report. Always required.
  validation     reviews the analysis for unsupported claims. Always required.

How to decide:
- Skip quality when the dataset has no missing values, no duplicates and
  consistent types; there is nothing for it to repair.
- Skip modelling when no plausible target column exists, or when the dataset is
  too small to validate a model.
- Skip eda when there are too few columns for any relationship to exist.
- Skip visualization only when there is nothing worth plotting.
- Including a stage that finds nothing costs time; omitting one that was needed
  costs the analysis. Prefer to include when genuinely uncertain.

Respond with a single JSON object and nothing else:
{
  "plan": ["stage names in execution order"],
  "skipped": [{"stage": "name", "reason": "why it is unnecessary for this dataset"}],
  "reasoning": "one or two sentences on the shape of this dataset and what it needs"
}"""


def _planning_summary(df: pd.DataFrame) -> dict[str, Any]:
    """A cheap structural summary for planning, computed before any stage runs."""
    profile = profile_dataset(df)
    types: dict[str, int] = {}
    for col in profile["columns"]:
        types[col["semantic_type"]] = types.get(col["semantic_type"], 0) + 1

    return {
        "rows": profile["shape"]["rows"],
        "columns": profile["shape"]["columns"],
        "duplicate_rows": profile["duplicate_rows"],
        "missing_cells": profile["total_missing_cells"],
        "overall_missing_pct": round(profile["missing_fraction_overall"] * 100, 2),
        "columns_with_missing": sum(1 for c in profile["columns"] if c["missing"] > 0),
        "semantic_type_counts": types,
        "column_names": [c["name"] for c in profile["columns"]][:40],
    }, profile


def _repair_plan(plan: list[str]) -> tuple[list[str], list[str]]:
    """Make a proposed plan executable.

    A plan is repaired rather than rejected: unknown stages are dropped, mandatory
    stages are restored, a stage whose dependencies were omitted is removed, and
    the remainder is sorted into the canonical order. A model that returns an
    impossible plan therefore degrades the workflow rather than breaking it.
    """
    notes: list[str] = []

    kept = []
    for stage in plan:
        if stage in REGISTRY:
            if stage not in kept:
                kept.append(stage)
        else:
            notes.append(f"Removed unknown stage '{stage}' from the plan.")

    for stage in MANDATORY:
        if stage not in kept:
            kept.append(stage)
            notes.append(f"Restored mandatory stage '{stage}' to the plan.")

    # Drop any stage whose dependencies are not in the plan.
    changed = True
    while changed:
        changed = False
        for stage in list(kept):
            missing = [d for d in REGISTRY[stage].requires if d not in kept]
            if missing:
                kept.remove(stage)
                notes.append(
                    f"Removed '{stage}': its dependency {missing} is not in the plan."
                )
                changed = True

    ordered = [s for s in CANONICAL_ORDER if s in kept]
    if ordered != kept:
        notes.append("Reordered the plan to respect stage dependencies.")

    return ordered, notes


def plan_workflow(df: pd.DataFrame, state: WorkflowState) -> dict[str, Any]:
    """Decide which stages to run for this dataset."""
    summary, profile = _planning_summary(df)

    if not CONFIG.adaptive_planning:
        plan = list(CANONICAL_ORDER)
        state.log("plan_created", mode="fixed", plan=plan)
        return {"plan": plan, "skipped": [], "reasoning": "Fixed pipeline; adaptive planning disabled.",
                "planning_profile": profile}

    user_prompt = (
        f"Dataset: {state.dataset_name}\n"
        f"{f'Stated analysis goal: {state.user_goal}' if state.user_goal else ''}\n\n"
        f"Structural summary:\n{json.dumps(summary, indent=2)}\n\n"
        "Which stages should run for this dataset?"
    )

    try:
        decision = ask_json("coordinator", PLANNER_SYSTEM, user_prompt,
                            required_keys=("plan",))
        proposed = decision.get("plan", [])
        skipped = decision.get("skipped", [])
        reasoning = decision.get("reasoning", "")
        mode = "adaptive"
    except LLMError as exc:
        # Without a planner the safe choice is to run everything: omitting a
        # stage risks losing analysis, while including one only costs time.
        proposed, skipped = list(CANONICAL_ORDER), []
        reasoning = f"Planning unavailable ({exc}); running the full workflow."
        mode = "fallback"

    plan, notes = _repair_plan(proposed)
    state.log("plan_created", mode=mode, plan=plan,
              skipped=[s.get("stage") for s in skipped if isinstance(s, dict)],
              repairs=notes)

    return {"plan": plan, "skipped": skipped, "reasoning": reasoning,
            "repairs": notes, "planning_profile": profile}


def should_run(stage: str, state: WorkflowState) -> tuple[bool, str]:
    """Decide at execution time whether a planned stage is still worth running.

    Planning happens before any analysis, so some stages only become unnecessary
    once an earlier stage has reported. This is where that is handled.
    """
    if stage == "modelling":
        if not state.target_column or state.task_type not in ("classification", "regression"):
            return False, "No supervised target was identified during profiling."
        if len(state.data) < 30:
            return False, f"Only {len(state.data)} rows; too few to validate a model."

    if stage == "quality":
        # Profiling has already computed the quality-relevant structure, so an
        # entirely clean dataset can skip the stage without inspecting it again.
        profile = state.outputs["profiling"].evidence.get("profile", {})
        if (profile.get("total_missing_cells") == 0
                and profile.get("duplicate_rows") == 0):
            return False, "No missing values or duplicate rows were found."

    if stage == "features":
        # Derived features are adopted only when they measurably improve a
        # prediction, so without a target there is nothing to measure against
        # and the stage would be adding columns on faith.
        if not state.target_column or state.task_type in (None, "none"):
            return False, "No target column, so a feature's contribution cannot be measured."
        if len(state.data) < 50:
            return False, f"Only {len(state.data)} rows; too few to measure a feature's contribution."

    if stage == "multivariate":
        numeric = [c for c in state.data.columns
                   if pd.api.types.is_numeric_dtype(state.data[c])
                   and state.data[c].nunique(dropna=True) > 1]
        if len(numeric) < 3:
            return False, f"Only {len(numeric)} usable numeric column(s); multivariate analysis needs at least three."
        if len(state.data) < 30:
            return False, f"Only {len(state.data)} rows; too few for multivariate analysis."

    if stage == "visualization" and len(state.data.columns) < 2:
        return False, "Too few columns to plot a relationship."

    return True, ""


def _notify(payload: dict) -> None:
    """Tell an observer -- the web interface -- that a stage has finished.

    A run takes minutes on a hosted model, and without this the interface could
    only say "running" until the end. A failing observer must never fail the
    analysis it is watching, so its errors are swallowed here.
    """
    if (listener := payload.get("on_progress")) is not None:
        try:
            listener(payload["state"])
        except Exception:  # noqa: BLE001 - display only
            pass


def build_graph() -> Any:
    """Compile the workflow as an explicit LangGraph state machine.

    The graph carries a single mutable `WorkflowState`; each node runs one agent
    and records its output. Routing after validation is where revision happens.
    """
    graph = StateGraph(dict)

    def make_node(stage: str) -> Callable[[dict], dict]:
        def node(payload: dict) -> dict:
            state: WorkflowState = payload["state"]

            # Every stage is a node in the compiled graph, so a stage the
            # coordinator omitted is skipped here rather than removed from the
            # graph. One graph shape for every run keeps traces comparable
            # across datasets, which the evaluation depends on.
            if state.plan and stage not in state.plan:
                return payload

            run, reason = should_run(stage, state)
            if not run:
                state.log("stage_skipped", agent=stage, reason=reason)
                payload.setdefault("skipped_at_runtime", []).append(
                    {"stage": stage, "reason": reason})
                _notify(payload)
                return payload

            output = REGISTRY[stage]().run(state)
            state.record(output)
            _notify(payload)
            return payload

        return node

    for stage in CANONICAL_ORDER:
        graph.add_node(stage, make_node(stage))

    graph.set_entry_point("profiling")
    for a, b in zip(CANONICAL_ORDER, CANONICAL_ORDER[1:]):
        graph.add_edge(a, b)

    # After validation the workflow either ends or returns to the report stage.
    def route(payload: dict) -> str:
        state: WorkflowState = payload["state"]
        validation = state.findings("validation")
        attempts = state.revisions

        if not validation.get("requires_revision"):
            return END
        if attempts >= CONFIG.max_validation_retries:
            state.log("revision_limit_reached", attempts=attempts)
            return END

        state.revisions = attempts + 1
        state.log("revision_requested", attempt=attempts + 1,
                  verdict=validation.get("verdict"),
                  issues=validation.get("issue_counts"))
        return "report"

    graph.add_conditional_edges("validation", route, {"report": "report", END: END})
    return graph.compile()


def _ensure_unique_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Give every column a distinct name, reporting any that had to be renamed.

    Duplicate headers are common in CSVs exported from spreadsheets, and pandas
    permits them. `df[name]` then returns a DataFrame rather than a Series, so
    `int(df[name].isna().sum())` raises `TypeError: cannot convert the series to
    <class 'int'>` in the profiler -- a hard crash that escaped `run_workflow`
    entirely, producing a traceback instead of a report after the file had
    already been read.

    Renaming rather than dropping: a duplicated header usually means two real
    columns that a tool wrote under one name, and discarding one would silently
    lose data. The suffix makes them addressable, and the rename is surfaced as
    a warning so the report can say the input was ambiguous.
    """
    if not df.columns.duplicated().any():
        return df, []

    # A suffix is chosen against every name in use, including the original
    # names still to come: renaming the second "a" to "a.1" collided with a
    # genuine "a.1" later in the header, and the frame left with a duplicate
    # the renaming existed to remove.
    originals = {str(name) for name in df.columns}
    renamed: list[str] = []
    used: set[str] = set()
    names: list[str] = []
    for name in df.columns:
        key = str(name)
        if key not in used:
            used.add(key)
            names.append(key)
            continue
        suffix = 1
        while f"{key}.{suffix}" in used or f"{key}.{suffix}" in originals:
            suffix += 1
        new = f"{key}.{suffix}"
        used.add(new)
        renamed.append(f"{key} -> {new}")
        names.append(new)

    out = df.copy()
    out.columns = names
    return out, renamed


def run_workflow(df: pd.DataFrame, dataset_name: str = "dataset",
                 user_goal: str | None = None,
                 spark_df: Any = None,
                 engine: dict[str, Any] | None = None,
                 on_progress: Callable[[WorkflowState], None] | None = None,
                 ) -> WorkflowState:
    """Run the complete adaptive workflow over a dataset.

    When `spark_df` is supplied the orchestrator has already determined that the
    dataset warrants distributed execution, and `df` is a bounded sample of it.
    Stages that can be distributed read the Spark frame and so describe the full
    dataset; stages that cannot -- charting and scikit-learn modelling -- read
    the sample. Which is which is recorded in the trace so the report can say so.

    `on_progress`, when given, is called with the state once the plan exists
    and again after every stage finishes or is skipped.
    """
    # Before anything reads the frame: pandas allows duplicate column names and
    # every tool here assumes a name selects one column.
    df, renamed_columns = _ensure_unique_columns(df)

    state = WorkflowState(df=df, dataset_name=dataset_name, user_goal=user_goal)
    if renamed_columns:
        state.log("columns_renamed", renamed=renamed_columns)
    if engine:
        state.engine = engine.get("engine", "pandas")
        state.engine_reason = engine.get("reason", "")
    else:
        # Callers that hand over an already-loaded frame (the evaluation harness,
        # the Streamlit app) have implicitly chosen the in-memory engine by
        # loading it, so the record says so rather than leaving the field blank.
        state.engine_reason = (
            "In-memory execution; the caller supplied an already-loaded frame.")
    if spark_df is not None:
        state.spark_df = spark_df
        state.full_row_count = spark_df.count()
    # Usage is tracked in a module-level singleton, so it is cleared here to make
    # the reported cost that of this run alone rather than of every run in the
    # process. The evaluation harness depends on this. The verification layer's
    # firing counts are per-run for the same reason.
    USAGE.reset()
    reset_firings()
    started = time.time()
    state.log("workflow_started", dataset=dataset_name, rows=len(df),
              columns=len(df.columns), engine=state.engine)
    if state.engine == "spark":
        state.log("execution_engine_selected", engine="spark",
                  full_rows=state.full_row_count, sample_rows=len(df),
                  reason=state.engine_reason)

    decision = plan_workflow(df, state)
    state.plan = decision["plan"]

    graph = build_graph()
    payload: dict[str, Any] = {"state": state, "on_progress": on_progress}
    _notify(payload)

    for stage in CANONICAL_ORDER:
        if stage not in state.plan:
            state.log("stage_omitted_by_plan", agent=stage)

    result = graph.invoke(payload, {"recursion_limit": 50})
    final_state: WorkflowState = result["state"]

    final_state.log(
        "workflow_completed",
        elapsed_s=round(time.time() - started, 3),
        revisions=final_state.revisions,
        verdict=final_state.findings("validation").get("verdict"),
    )
    final_state.planning = decision
    final_state.usage = USAGE.summary()
    final_state.verification = firings()
    return final_state
