"""The Visualization Agent: choose the charts that carry the findings.

The agent selects which charts to draw and why; `tools.visualization` draws them.
Selection is the part that needs judgement, because a report that plots every
column produces volume rather than insight. The agent is asked to justify each
chart against a specific finding, and a chart it cannot justify is not drawn.

Every chart carries the values it displays, so the interpretation attached to it
can be checked against what was actually plotted rather than taken on trust.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from agents.llm import LLMError, ask_json
from state import AgentOutput, WorkflowState
from tools.visualization import build_chart, export_chart
from verification import failed_chart_request

MAX_CHARTS = 8

SYSTEM_PROMPT = """You are the Visualization Agent in a multi-agent data science system.

You receive the statistical findings already produced for a dataset. You choose
which charts to draw. You do not draw them and you do not compute values.

Available chart kinds and their required arguments:
  target_balance  {"target": column}                  class distribution of the target
  grouped_rate    {"feature": column, "target": column}  target rate within each category
  box             {"numeric": column, "group": column}   numeric distribution across groups
  histogram       {"column": numeric column}             distribution of one numeric column
  scatter         {"x": numeric column, "y": numeric column}  relationship between two numerics
  heatmap         {}                                     correlation matrix of numeric columns
  bar             {"column": categorical column}         category frequencies

How to choose:
- Every chart must show a specific finding from the statistics you were given.
  Charts that illustrate nothing are noise.
- Prioritise the strongest effects. A chart of a negligible relationship wastes
  the reader's attention.
- Choose at most 6 charts. Fewer, well-chosen charts beat many weak ones.
- Match the chart to the data: grouped_rate and box for a categorical target,
  scatter and histogram for numeric relationships, heatmap only when several
  numeric columns exist.
- Do not request the same relationship twice in different forms.
- histogram, box and scatter require genuinely numeric columns.

Respond with a single JSON object and nothing else:
{
  "charts": [
    {"kind": "chart kind", "<argument>": "column name", ...,
     "reason": "the specific finding this chart shows"}
  ],
  "narrative": "a short paragraph describing what the chosen charts collectively show"
}"""

REQUIRED = ("charts",)


def _finding_summary(state: WorkflowState) -> dict[str, Any]:
    """The statistical context the agent needs in order to choose charts."""
    eda = state.findings("eda")
    profiling = state.findings("profiling")
    evidence = state.outputs.get("eda")
    raw = evidence.evidence.get("eda_results", {}) if evidence else {}

    summary: dict[str, Any] = {
        "target": state.target_column,
        "task_type": state.task_type,
        "column_types": profiling.get("column_types", {}),
        "usable_features": profiling.get("usable_features", []),
        "strongest_effects": eda.get("strongest_effects", {}),
        "key_insights": [
            {"insight": i.get("insight", "")[:200], "importance": i.get("importance")}
            for i in eda.get("key_insights", [])[:8]
        ],
    }

    if tb := raw.get("target_balance"):
        summary["target_balance"] = {"imbalance_ratio": tb["imbalance_ratio"],
                                     "n_classes": tb["n_classes"]}
    if assoc := raw.get("categorical_associations"):
        summary["categorical_associations"] = [
            {"feature": a["feature"], "cramers_v": a["cramers_v"], "strength": a["strength"]}
            for a in assoc[:8]
        ]
    if gd := raw.get("group_differences"):
        summary["group_differences"] = [
            {"numeric": g["numeric_column"], "grouped_by": g["group_column"],
             "eta_squared": g["eta_squared"], "effect_size": g["effect_size"]}
            for g in gd[:8]
        ]
    if pairs := raw.get("correlations", {}).get("pairs"):
        summary["feature_correlations"] = [
            {"pair": [p["column_a"], p["column_b"]], "pearson": p["pearson"],
             "strength": p["strength"]} for p in pairs[:8]
        ]
    if dists := raw.get("distributions"):
        summary["skewed_columns"] = [
            {"column": d["column"], "skewness": d["skewness"]}
            for d in dists if d["suggests_transform"]
        ][:5]

    return summary


class VisualizationAgent(Agent):
    name = "visualization"
    description = (
        "Selects the charts that best communicate the statistical findings, "
        "generates them, and interprets what each one shows."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []

        # 1. Choose.
        user_prompt = (
            f"Dataset: {state.dataset_name}\n"
            f"Description: {state.findings('profiling').get('dataset_description', '')}\n\n"
            f"Findings to illustrate:\n"
            f"{json.dumps(_finding_summary(state), indent=2, default=str)}\n\n"
            "Choose the charts that best communicate these findings."
        )

        try:
            decision = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            decision = self._fallback_charts(state)
            warnings.append(f"Model selection unavailable, used default charts. ({exc})")
            status = "degraded"

        specs = decision.get("charts", [])[:MAX_CHARTS]
        if not specs:
            warnings.append("No charts were selected.")

        # 2. Draw. A rejected specification is recorded and skipped; one bad
        #    request must not cost the report every other chart.
        charts: list[dict[str, Any]] = []
        seen: set[str] = set()

        for spec in specs:
            chart, error = build_chart(df, spec)
            if error:
                # A chart the data cannot support is never drawn; the
                # mechanism decides only whether the refusal is reported.
                if failed_chart_request(error):
                    warnings.append(f"Skipped chart: {error}")
                continue
            if chart["id"] in seen:
                warnings.append(f"Skipped duplicate chart '{chart['id']}'.")
                continue
            seen.add(chart["id"])

            path = export_chart(chart)
            if path is None:
                # `and not charts` reported only the first failure, because
                # `charts` is non-empty from the second chart onwards. A later
                # export failure was therefore silent, and the report carried a
                # link to an image that was never written -- which Markdown
                # renders as nothing at all rather than as an error.
                warnings.append(
                    f"Static export failed for chart '{chart['id']}'; the report "
                    f"will reference it without an embedded image."
                )

            charts.append({
                "id": chart["id"],
                "kind": chart["kind"],
                "title": chart["title"],
                "columns": chart["columns"],
                "description": chart["description"],
                "reason": spec.get("reason", ""),
                "data": chart["data"],
                "path": chart["path"],
                "figure": chart["figure"],  # consumed by the interface, not the report
            })

        if specs and not charts:
            status = "degraded"
            warnings.append("Every requested chart failed to build.")

        findings = {
            "charts": [{k: v for k, v in c.items() if k != "figure"} for c in charts],
            "n_charts": len(charts),
            "n_requested": len(specs),
            "chart_ids": [c["id"] for c in charts],
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=decision.get("narrative", ""),
            # Figures are live Plotly objects for the interface to render. They
            # are listed in `state.LIVE_ONLY_EVIDENCE`, so `to_dict` drops them
            # from the serialised record -- the comment here previously claimed
            # they were kept out of evidence while the next line put them in,
            # and `json.dumps(..., default=str)` then wrote every plotted point
            # into each run's JSON.
            evidence={"figures": {c["id"]: c["figure"] for c in charts},
                      "chart_data": {c["id"]: c["data"] for c in charts}},
            warnings=warnings,
        )

    @staticmethod
    def _fallback_charts(state: WorkflowState) -> dict[str, Any]:
        """Choose charts from the strongest computed effects, without a model."""
        specs: list[dict[str, Any]] = []
        target = state.target_column
        profiling = state.findings("profiling")
        types = profiling.get("column_types", {})
        eda = state.findings("eda")

        if target and state.task_type == "classification":
            specs.append({"kind": "target_balance", "target": target,
                          "reason": "Class distribution of the target."})

        for col, effect in list(eda.get("strongest_effects", {}).items())[:4]:
            kind_of = types.get(col)
            if state.task_type == "classification" and kind_of in (
                    "categorical", "categorical_numeric", "boolean"):
                specs.append({"kind": "grouped_rate", "feature": col, "target": target,
                              "reason": f"Association with the target (effect {effect:.3f})."})
            elif state.task_type == "classification" and kind_of == "numeric":
                specs.append({"kind": "box", "numeric": col, "group": target,
                              "reason": f"Distribution differs by target (effect {effect:.3f})."})

        specs.append({"kind": "heatmap", "reason": "Overview of numeric correlations."})
        return {"charts": specs[:6],
                "narrative": "Charts selected from the strongest computed effects."}
