"""The Report Agent: assemble the findings of every stage into one document.

This agent introduces no analysis of its own. It receives what the other agents
established and composes a report from it, which makes it the point in the
workflow where unsupported statements are most likely to appear: composing prose
invites filling gaps with plausible-sounding claims.

Two things limit that. The agent is given the findings only, never the raw data,
so it has nothing to compute from. And every number it writes is checked against
the numbers the other agents recorded, with unmatched values reported rather
than silently accepted.
"""
from __future__ import annotations

import json
from typing import Any

from config import CONFIG
from state import AgentOutput, WorkflowState
from agents.base import Agent
from agents.llm import LLMError, ask_json, bound_payload
from verification import (causal_language, computed_values, report_text,
                          suppressed_negative_result, unsupported_numbers)

SYSTEM_PROMPT = """You are the Report Agent in a multi-agent data science system.

You receive the findings of every earlier stage. You write the final analytical
report. You perform no analysis of your own.

Absolute rules:
- Every number you write must appear in the findings you were given. Never
  estimate, round differently, or infer a value that is not there.
- Never introduce a finding no agent reported.
- Never claim causation. Write "is associated with", not "causes" or "drives".
- If the modelling stage found no signal, say so plainly. A negative result is a
  real result and must not be softened into apparent success.
- Report the limitations. A report that hides its caveats is not usable.

Write for someone who understands the subject but not the statistics. Explain
what a finding means, not merely what its value is.

Respond with a single JSON object and nothing else:
{
  "title": "a specific title naming this dataset and analysis",
  "executive_summary": "3-5 sentences a decision-maker could read alone",
  "sections": [
    {"heading": "section heading",
     "body": "the section text, in full sentences, citing concrete numbers"}
  ],
  "key_takeaways": ["the most important conclusions, most important first"],
  "limitations": ["what constrains these conclusions"],
  "recommended_next_steps": ["what a reader should do next"]
}

Cover, in this order and only where the stage ran: what the dataset contains and
its quality; what features were derived and whether they measurably helped; what
the exploratory analysis found; what the structure of the feature space shows;
what the charts show; what the modelling established; and what it all means.

If derived features were not adopted, say so plainly. Feature engineering that
did not improve performance is a real result and must not be presented as though
it did."""

REQUIRED = ("title", "executive_summary", "sections")

class ReportAgent(Agent):
    name = "report"
    description = (
        "Assembles the findings of every stage into a structured analytical "
        "report written for a human reader."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        warnings: list[str] = []

        # 1. Gather what the earlier stages established. Only findings are
        #    passed on: the agent must not have raw data to compute from.
        # On a distributed run `state.data` is a sample, so reporting its length
        # would understate what the analysis covered: profiling and exploratory
        # statistics were computed across every row. The report must state the
        # size of the dataset it describes, and separately say which stages saw
        # only the sample.
        material: dict[str, Any] = {
            "dataset": state.dataset_name,
            "rows": int(state.full_row_count or state.data.shape[0]),
            "columns": int(state.data.shape[1]),
            "target": state.target_column,
            "task_type": state.task_type,
            "stages_completed": state.completed,
        }
        if state.user_goal:
            material["stated_goal"] = state.user_goal
        if state.engine == "spark":
            material["execution"] = {
                "engine": "spark",
                "reason": state.engine_reason,
                "full_rows_analysed": state.full_row_count,
                "in_memory_sample_rows": int(state.data.shape[0]),
                "note": (
                    "Profiling and quality statistics were computed across the "
                    "full dataset. Charting and model evaluation used the sample, "
                    "so model scores describe the sample rather than every row."
                ),
            }

        for agent_name in ("profiling", "quality", "features", "eda", "multivariate",
                           "visualization", "modelling"):
            output = state.outputs.get(agent_name)
            if not output or not output.ok:
                continue
            findings = output.findings
            if agent_name == "visualization":
                # A chart's `data` holds every value it plots, which for a
                # correlation matrix or a contingency table is far larger than
                # anything the report needs. The description already states what
                # the chart shows, so the raw values are withheld from the prompt
                # while remaining available to the number check via evidence.
                findings = {
                    **findings,
                    "charts": [
                        {k: v for k, v in c.items() if k != "data"}
                        for c in findings.get("charts", [])
                    ],
                }
            material[agent_name] = {
                "findings": findings,
                "narrative": output.narrative,
            }
            if output.warnings:
                material[agent_name]["agent_warnings"] = output.warnings

        # 1b. The validator's objections, when this is a rewrite.
        #
        # The orchestrator routes a report that failed validation back to this
        # agent, and the graph describes that as the report being "re-composed
        # with the validator's objections in hand". It was not: this agent read
        # nothing from the validation stage, so the rewrite was built from an
        # essentially identical prompt and could only converge by luck. On a
        # rate-limited provider it also spent two further report and validation
        # round trips to arrive back where it started.
        #
        # The objections are passed as instructions rather than as findings,
        # because they are the one part of the material that is about this
        # report rather than about the data.
        if previous := state.findings("validation"):
            if issues := previous.get("issues"):
                material["revision_required"] = {
                    "why": ("A previous draft of this report was reviewed "
                            "against the computed results and did not pass. "
                            "Each objection below names something the draft "
                            "claimed that the calculations do not support. "
                            "Address every one: correct the claim to match the "
                            "computed value, or remove it. Do not restate a "
                            "claim the review rejected."),
                    "objections": [
                        {"severity": i.get("severity"),
                         "about": i.get("stage") or i.get("location"),
                         "problem": i.get("problem"),
                         "how_to_fix": i.get("suggested_fix")}
                        for i in issues[:12]
                    ],
                }

        # Charts are described rather than shown, so the report can reference
        # them by what they display.
        if viz := state.findings("visualization"):
            material["available_charts"] = [
                {"id": c["id"], "title": c["title"], "shows": c["description"]}
                for c in viz.get("charts", [])
            ]

        # 2. Compose. The material is bounded first: it accumulates the output
        #    of every earlier stage, and a real model writes longer narratives
        #    than a scripted one, so an unbounded prompt eventually exceeds the
        #    provider's per-request limit and fails the one stage whose job is
        #    to produce the deliverable. Trimming affects only what the model
        #    is shown: the number check below reads the computed evidence
        #    directly, never this payload.
        material, was_trimmed = bound_payload(material, CONFIG.report_prompt_char_budget)
        if was_trimmed:
            warnings.append(
                "The findings exceeded the prompt budget and were trimmed for "
                "composition; the number check still runs against the full "
                "computed values."
            )

        user_prompt = (
            f"Write the final analytical report from these findings.\n\n"
            f"{json.dumps(material, indent=2, default=str)}"
        )

        try:
            report = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            report = self._fallback(state, material)
            warnings.append(f"Model composition unavailable, assembled from findings. ({exc})")
            status = "degraded"

        # 3. Check every number in the report against the computed values.
        # The permitted set is every value a tool computed (see
        # `verification.computed_values`), and nothing a language model wrote:
        # not the stages' narratives, and not the validator's objections to
        # the previous draft. Drawing it from the prompt material, as this did,
        # let an invented figure quoted back in an objection become permitted
        # in the next draft.
        full_text = report_text(report)

        unsupported = unsupported_numbers(full_text, computed_values(state))
        if unsupported:
            warnings.append(
                f"{len(unsupported)} number(s) in the report do not match any computed "
                f"value: {unsupported[:10]}"
            )

        causal = causal_language(full_text)
        if causal:
            warnings.append(f"Report uses causal language: {causal}.")

        # A negative modelling result must survive into the report.
        modelling = state.findings("modelling")
        if modelling and suppressed_negative_result(
                full_text, modelling.get("beats_baseline")):
            warnings.append(
                "The modelling stage found no signal, but the report does not "
                "state this clearly."
            )

        findings = {
            "title": report.get("title", f"Analysis of {state.dataset_name}"),
            "executive_summary": report.get("executive_summary", ""),
            "sections": report.get("sections", []),
            "key_takeaways": report.get("key_takeaways", []),
            "limitations": report.get("limitations", []),
            "recommended_next_steps": report.get("recommended_next_steps", []),
            "chart_ids": [c["id"] for c in (state.findings("visualization") or {}).get("charts", [])],
            "unsupported_numbers": unsupported,
            "n_sections": len(report.get("sections", [])),
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=report.get("executive_summary", ""),
            evidence={"source_material": list(material.keys())},
            warnings=warnings,
        )

    @staticmethod
    def _fallback(state: WorkflowState, material: dict[str, Any]) -> dict[str, Any]:
        """Assemble a report from the agents' own narratives, without composition."""
        sections = []
        headings = {
            "profiling": "Dataset Overview",
            "quality": "Data Quality",
            "features": "Feature Engineering",
            "eda": "Exploratory Analysis",
            "multivariate": "Feature Space Structure",
            "visualization": "Visualizations",
            "modelling": "Modelling",
        }
        for agent_name, heading in headings.items():
            output = state.outputs.get(agent_name)
            if output and output.ok and output.narrative:
                sections.append({"heading": heading, "body": output.narrative})

        takeaways = [
            i.get("insight", "") for i in state.findings("eda").get("key_insights", [])
            if i.get("importance") == "high"
        ]
        limitations = list(state.findings("quality").get("limitations", []))
        limitations += list(state.findings("modelling").get("caveats", []))

        return {
            "title": f"Analytical Report: {state.dataset_name}",
            "executive_summary": (
                f"This report covers {material['rows']} rows and {material['columns']} "
                f"columns from {state.dataset_name}. It was assembled directly from "
                f"agent findings without model composition."
            ),
            "sections": sections,
            "key_takeaways": takeaways,
            "limitations": limitations,
            "recommended_next_steps": state.findings("modelling").get("improvements", []),
        }
