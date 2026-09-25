"""The shared workflow state passed between agents.

Every agent reads from and writes to a single `WorkflowState`. Two design points
matter for the framework:

* Agents communicate only through this object, never by calling each other. The
  orchestrator therefore fully controls information flow, and an agent can be
  added, removed or reordered without touching any other agent.
* The state records not just results but the sequence of events that produced
  them, so a completed run can be replayed, audited, or presented as a trace of
  how the analysis was reached.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


#: Evidence keys that exist for the live interface and must never reach the
#: serialised record. Plotly figures are the case this was written for: the
#: visualisation agent stores them so the Streamlit app can render them
#: interactively, but `json.dumps(..., default=str)` in `save_report` and the
#: evaluation harness then wrote each figure's full `str()` repr -- every
#: plotted point, up to a scatter's 5,000-point cap -- into the run record.
#: They are not evidence about the data; they are a rendering of it.
LIVE_ONLY_EVIDENCE = frozenset({"figures"})


@dataclass
class AgentOutput:
    """The uniform result contract every agent returns.

    `findings` is consumed by other agents, `narrative` by the report, and
    `evidence` by the Validation Agent, which checks that statements made in the
    narrative are actually supported by computed values.
    """

    agent: str
    status: str = "ok"  # ok | degraded | failed
    findings: dict[str, Any] = field(default_factory=dict)
    narrative: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status != "failed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "status": self.status,
            "findings": self.findings,
            "narrative": self.narrative,
            "evidence": {k: v for k, v in self.evidence.items()
                         if k not in LIVE_ONLY_EVIDENCE},
            "warnings": self.warnings,
            "error": self.error,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class WorkflowState:
    """Everything known about the dataset and the run so far."""

    # Inputs.
    df: pd.DataFrame
    dataset_name: str = "dataset"
    user_goal: str | None = None  # optional natural-language objective

    # The working frame, replaced by the cleaning stage. Agents downstream of
    # cleaning read this; the original `df` is kept so the report can state what
    # changed and the run can be re-executed from the raw data.
    working_df: pd.DataFrame | None = None

    # Results, keyed by agent name.
    outputs: dict[str, AgentOutput] = field(default_factory=dict)

    # What was done to the data before modelling, in order, with the
    # parameters fitted at the time: repairs from the quality stage, derived
    # columns from the feature stage. The saved model replays these on raw
    # rows (`tools.preparation`), which is what lets it accept the original
    # file rather than the working frame. Kept out of findings and evidence:
    # it is machinery for the saved model, and a frequency table belongs in
    # neither a prompt nor the set of numbers a report may cite.
    preparation: list[dict[str, Any]] = field(default_factory=list)

    # The orchestrator's plan and what has actually run.
    plan: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)

    # Decisions the workflow depends on, filled in as they are established.
    target_column: str | None = None
    task_type: str | None = None  # classification | regression | none

    started_at: float = field(default_factory=time.time)

    # How many times validation has sent the report back. Held here rather than
    # in the graph payload because LangGraph passes a copy of the payload to
    # routing functions, so a counter incremented there would not persist; this
    # object is shared by reference and does.
    revisions: int = 0

    # Which execution engine computed the distributed-capable stages, and the
    # evidence for choosing it. Recorded because the choice is made
    # automatically from the size of the data, and a reader of the report needs
    # to be able to see why. When the engine is Spark, `df` holds a bounded
    # sample and `full_row_count` is the true size of the dataset.
    engine: str = "pandas"
    engine_reason: str = ""
    full_row_count: int | None = None
    spark_df: Any = None

    # Filled in by the orchestrator once the run completes.
    planning: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)
    # How often each verification mechanism found something in this run, as
    # counted by the mechanisms themselves (`verification.firings`).
    verification: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.working_df is None:
            self.working_df = self.df.copy()

    # -- access ------------------------------------------------------------- #

    @property
    def data(self) -> pd.DataFrame:
        """The frame agents should analyse: cleaned if cleaning has run."""
        return self.working_df if self.working_df is not None else self.df

    def findings(self, agent: str) -> dict[str, Any]:
        """Findings from a previous agent, or an empty dict if it never ran.

        Returning empty rather than raising lets an agent degrade gracefully when
        the orchestrator chose to skip an upstream stage.
        """
        out = self.outputs.get(agent)
        return out.findings if out and out.ok else {}

    def has_run(self, agent: str) -> bool:
        return agent in self.outputs and self.outputs[agent].ok

    # -- recording ---------------------------------------------------------- #

    def record(self, output: AgentOutput) -> None:
        self.outputs[output.agent] = output
        if output.agent not in self.completed:
            self.completed.append(output.agent)
        self.log(
            "agent_completed",
            agent=output.agent,
            status=output.status,
            duration_s=round(output.duration_s, 3),
            warnings=len(output.warnings),
        )

    def log(self, event: str, **details: Any) -> None:
        self.trace.append(
            {"t": round(time.time() - self.started_at, 3), "event": event, **details}
        )

    def summary(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            # On a distributed run `rows` is the size of the whole dataset, not
            # of the in-memory sample, so the record describes what was analysed.
            "rows": int(self.full_row_count or self.data.shape[0]),
            "columns": int(self.data.shape[1]),
            "engine": self.engine,
            "engine_reason": self.engine_reason,
            "in_memory_rows": int(self.data.shape[0]),
            "target_column": self.target_column,
            "task_type": self.task_type,
            "plan": self.plan,
            "completed": self.completed,
            "revisions": self.revisions,
            "elapsed_s": round(time.time() - self.started_at, 3),
            "planning": {k: v for k, v in self.planning.items() if k != "planning_profile"},
            "usage": self.usage,
            "verification": self.verification,
            "outputs": {k: v.to_dict() for k, v in self.outputs.items()},
            "trace": self.trace,
        }
