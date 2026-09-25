"""The Validation Agent: review the completed workflow before it is delivered.

This agent is the framework's answer to the central weakness of LLM-based
analysis, which is that a fluent report is not necessarily a correct one. It runs
last and reviews the other agents' output rather than the data.

Validation happens in two passes with different strengths. Deterministic checks
catch what can be stated as a rule: a conclusion contradicting a measurement, a
number with no computed source, a stage that failed silently. The model then
reviews what rules cannot express, such as whether the report's emphasis matches
the evidence or whether a section quietly overstates a weak result.

The deterministic checks are authoritative. A model review that disagrees with a
measurement does not overturn it.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from config import CONFIG
from agents.llm import LLMError, as_text, ask_json, bound_payload
from state import AgentOutput, WorkflowState
from verification import overridden_signal_claim

#: The stage names the workflow actually has, used to interpret a
#: model-supplied location. Imported lazily inside the function to avoid a
#: circular import with the orchestrator, which constructs the agents.

SYSTEM_PROMPT = """You are the Validation Agent in a multi-agent data science system.

You review the outputs of the other agents. You do not analyse the dataset and
you do not add findings. Your role is to find problems a reader would be misled
by.

Look for:
- Claims in the report that the findings do not support.
- Conclusions that contradict the measurements, especially a weak or null result
  presented as a successful one.
- Causal language applied to observational associations.
- Important findings that were established but omitted from the report.
- Limitations that were identified earlier but not carried into the report.
- Statements whose confidence exceeds what the evidence justifies.

Do not invent problems. If the analysis is sound, say so. Judge only what you
were given.

Severity levels:
  critical  the report would mislead a reader on a substantive point
  warning   a real weakness that should be corrected but does not invalidate the work
  info      a minor improvement

Respond with a single JSON object and nothing else:
{
  "issues": [
    {"severity": "critical | warning | info",
     "location": "which agent or report section",
     "problem": "what is wrong",
     "evidence": "the finding or measurement that shows it is wrong",
     "suggested_fix": "what should change"}
  ],
  "verdict": "sound | needs_revision | unreliable",
  "confidence_in_analysis": "high | medium | low",
  "narrative": "a short assessment of the analysis as a whole"
}"""

REQUIRED = ("issues", "verdict")

SEVERITY_RANK = {"critical": 3, "warning": 2, "info": 1}


def _stage_named_in(location: Any) -> str:
    """The workflow stage a model-supplied location refers to, or "".

    Taking the first word worked only when the model answered with a bare stage
    name. It also answers with a phrase ("the report stage"), a list of stages,
    or whitespace -- the first two yielding "the", which matches no stage, and
    the third raising. Since this drives the orchestrator's decision about what
    to re-run, a location it cannot parse must produce nothing rather than
    something wrong: re-running the wrong stage is worse than re-running none.

    Searching the text for a name the workflow actually has is both more
    tolerant of phrasing and incapable of inventing a stage.
    """
    # Imported here rather than at module scope: the orchestrator constructs
    # the agents, so a top-level import would be circular.
    from orchestrator.graph import CANONICAL_ORDER

    text = as_text(location).lower()
    return next((stage for stage in CANONICAL_ORDER if stage in text), "")


class ValidationAgent(Agent):
    name = "validation"
    description = (
        "Reviews the completed analysis for unsupported claims, contradictions "
        "and omissions, and judges whether the output is reliable."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        warnings: list[str] = []

        # 1. Deterministic checks. These are facts about the run, not opinions.
        issues = self._mechanical_checks(state)

        # 2. Model review of what rules cannot express.
        material = {
            "dataset": state.dataset_name,
            "target": state.target_column,
            "task_type": state.task_type,
            "stages_completed": state.completed,
            "stages_failed": [
                name for name, o in state.outputs.items() if not o.ok
            ],
        }
        for name in ("profiling", "quality", "features", "eda", "multivariate",
                     "modelling", "report"):
            output = state.outputs.get(name)
            if output and output.ok:
                material[name] = {
                    "findings": output.findings,
                    "warnings": output.warnings,
                }

        # Bounded for the same reason as the report: this stage sees every
        # earlier agent's findings, so its prompt is the largest in the run
        # and is the first to exceed a provider's per-request limit. Losing
        # the reviewer to an oversized prompt would remove the very check
        # the framework exists to perform.
        material, was_trimmed = bound_payload(
            material, CONFIG.report_prompt_char_budget)
        if was_trimmed:
            warnings.append(
                "The findings exceeded the prompt budget and were trimmed for "
                "review; the mechanical checks still ran against the full output."
            )

        try:
            review = ask_json(
                self.name, SYSTEM_PROMPT,
                "Review this completed analysis.\n\n"
                + json.dumps(material, indent=2, default=str),
                required_keys=REQUIRED,
            )
            status = "ok"
            for issue in review.get("issues", []):
                severity = str(issue.get("severity", "info")).lower()
                issues.append({
                    "severity": severity if severity in SEVERITY_RANK else "info",
                    "source": "model_review",
                    "location": issue.get("location", "unknown"),
                    "problem": issue.get("problem", ""),
                    "evidence": issue.get("evidence", ""),
                    "suggested_fix": issue.get("suggested_fix", ""),
                })
        except LLMError as exc:
            review = {"verdict": None, "confidence_in_analysis": None,
                      "narrative": "Automated checks only; no model review was available."}
            warnings.append(f"Model review unavailable, ran mechanical checks only. ({exc})")
            status = "degraded"

        issues.sort(key=lambda i: -SEVERITY_RANK.get(i["severity"], 0))
        counts = {"critical": 0, "warning": 0, "info": 0}
        for issue in issues:
            counts[issue["severity"]] += 1

        # 3. The verdict follows the mechanical findings. A model that judges the
        #    analysis sound while a measurement contradicts its conclusions is
        #    overruled, since the measurement is the stronger evidence.
        mechanical_critical = any(
            i["severity"] == "critical" and i["source"] == "automated_check"
            for i in issues
        )
        if mechanical_critical:
            verdict = "unreliable"
            if review.get("verdict") == "sound":
                warnings.append(
                    "Model review judged the analysis sound despite a critical "
                    "automated finding; verdict overridden."
                )
        elif counts["critical"]:
            verdict = "needs_revision"
        elif counts["warning"] >= 3:
            verdict = "needs_revision"
        else:
            verdict = review.get("verdict") if review.get("verdict") in (
                "sound", "needs_revision", "unreliable") else "sound"

        findings = {
            "verdict": verdict,
            "issue_counts": counts,
            "issues": issues,
            "confidence_in_analysis": review.get("confidence_in_analysis"),
            "n_issues": len(issues),
            # Consumed by the orchestrator when deciding whether to re-run a stage.
            # A revision is warranted only when something a rewrite could
            # actually fix is outstanding. The verdict alone was the trigger,
            # so a run whose only faults were unfixable -- a failed upstream
            # stage, a degraded one -- looped until it ran out of retries and
            # reported the same verdict it started with.
            "requires_revision": (
                verdict in ("needs_revision", "unreliable")
                and any(i.get("revisable", True) for i in issues
                        if i.get("severity") in ("critical", "warning"))
            ),
            # `as_name` rather than `.split()[0]`: the model may return a
            # location as a list or as whitespace, and both raised here. This
            # feeds the orchestrator's choice of which stage to re-run, so a
            # crash here did not merely lose a field -- it failed the validation
            # stage, leaving `requires_revision` unset and shipping the report
            # unreviewed. The check the framework exists to perform was the one
            # thing a malformed location could switch off.
            "stages_to_revisit": sorted({
                name for i in issues
                if i.get("severity") == "critical" and i.get("location")
                and (name := _stage_named_in(i.get("location")))
            }),
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=review.get("narrative", ""),
            evidence={"checks_run": "mechanical and model review"},
            warnings=warnings,
        )

    # -- deterministic checks ------------------------------------------------ #

    def _mechanical_checks(self, state: WorkflowState) -> list[dict[str, Any]]:
        """Checks that can be decided from the recorded state alone."""
        issues: list[dict[str, Any]] = []

        def add(severity: str, location: str, problem: str,
                evidence: str, fix: str, revisable: bool = True) -> None:
            """Record an issue.

            `revisable` says whether re-composing the report could address it.
            Most issues are about what the report *claims* and can be fixed by
            writing it differently. Some are about what the run *did* -- a stage
            that failed, an analysis that never happened -- and no rewrite can
            repair those. Both must be reported; only the first should trigger a
            revision.
            """
            issues.append({
                "severity": severity, "source": "automated_check",
                "location": location, "problem": problem,
                "evidence": evidence, "suggested_fix": fix,
                "revisable": revisable,
            })

        report = state.findings("report")
        modelling = state.findings("modelling")
        eda = state.findings("eda")

        # A failed stage that nothing reported is the most consequential silent
        # error, because the report simply omits whatever it would have found.
        for name, output in state.outputs.items():
            if output.status == "failed":
                # Reported as critical, because the reader must know an analysis
                # is missing -- but not revisable. Re-composing the report
                # cannot restore a stage that never ran, and treating this as a
                # revision trigger spent two further report-and-validation round
                # trips to arrive at the identical verdict. On a rate-limited
                # free tier those round trips are the scarce resource, and the
                # observed titanic run burned both of them this way.
                add("critical", name,
                    f"The {name} stage failed and its analysis is missing entirely.",
                    output.error or "no error recorded",
                    f"Investigate the {name} failure and re-run the workflow.",
                    revisable=False)
            elif output.status == "degraded":
                add("warning", name,
                    f"The {name} stage completed in a degraded state.",
                    "; ".join(output.warnings[:2]) or "no detail recorded",
                    "Confirm the stage's output is still usable.",
                    revisable=False)

        # Numbers in the report with no computed source.
        if unsupported := report.get("unsupported_numbers"):
            add("critical", "report",
                "The report states numbers that do not appear in any computed finding.",
                f"unmatched values: {unsupported[:8]}",
                "Remove or correct these values against the computed findings.")

        # A null modelling result presented as success.
        #
        # Not revisable: both issues are in the modelling stage's findings, and
        # a revision re-runs the report, not modelling. Marked revisable, they
        # sent the workflow back to the report twice -- four further model
        # calls -- to arrive at the same "unreliable" verdict every time. The
        # modelling agent now corrects both itself, so these fire only when
        # that correction did not happen.
        if (modelling and modelling.get("beats_baseline") is False
                and overridden_signal_claim(modelling.get("signal_assessment"),
                                            False,
                                            modelling.get("recommended_model"))):
            if modelling.get("recommended_model"):
                add("critical", "modelling",
                    "A model is recommended although performance did not exceed chance.",
                    f"lift over baseline {modelling.get('lift_over_baseline')}, "
                    f"permutation p = {modelling.get('permutation_p_value')}",
                    "State that the features do not predict the target.",
                    revisable=False)
            if str(modelling.get("signal_assessment", "")).lower() in ("strong", "moderate"):
                add("critical", "modelling",
                    "The signal assessment overstates a result that did not beat chance.",
                    f"signal_assessment = {modelling.get('signal_assessment')}, "
                    f"beats_baseline = False",
                    "Set the signal assessment to none.",
                    revisable=False)

        # High-importance EDA findings dropped from the report.
        if eda and report:
            high = [
                i.get("insight", "") for i in eda.get("key_insights", [])
                if isinstance(i, dict) and i.get("importance") == "high"
            ]
            if high:
                # Coerced, for the same reason `report_agent` coerces: the
                # model returns takeaways as objects and sections as bare
                # strings often enough that both were observed. Indexing them
                # here raised inside the validation stage, which `Agent.run`
                # records as `failed` -- so `findings("validation")` returned
                # empty, `route` saw no `requires_revision`, and the report
                # shipped unreviewed. Malformed model output could switch off
                # the one check the framework exists to perform.
                report_text = as_text([
                    report.get("executive_summary"),
                    [s.get("body", "") if isinstance(s, dict) else s
                     for s in report.get("sections", [])],
                    report.get("key_takeaways"),
                ]).lower()
                # An insight is considered carried through when the columns it
                # concerns are discussed in the report. Matching on column names
                # rather than on wording survives the rephrasing that a report
                # legitimately applies, while still detecting a dropped finding.
                columns = [str(c) for c in state.data.columns]
                missing = []
                for insight in high:
                    mentioned = [c for c in columns if c.lower() in insight.lower()]
                    if mentioned and not any(c.lower() in report_text for c in mentioned):
                        missing.append(insight[:70])
                if missing:
                    add("warning", "report",
                        "High-importance findings from the exploratory analysis do not "
                        "appear in the report.",
                        f"omitted: {missing}",
                        "Include these findings or explain their exclusion.")

        # Limitations established earlier but absent from the report.
        known_limitations = list(state.findings("quality").get("limitations", []))
        known_limitations += list(modelling.get("caveats", []) or [])
        if known_limitations and report is not None:
            if not report.get("limitations"):
                add("warning", "report",
                    "Limitations were identified during the analysis but the report "
                    "lists none.",
                    f"{len(known_limitations)} limitation(s) recorded upstream",
                    "Carry the recorded limitations into the report.")

        # An analysis with no target produces no modelling, which the report must
        # acknowledge rather than leave as an unexplained gap.
        if state.has_run("report") and not state.target_column:
            add("info", "report",
                "No supervised target was identified, so no modelling was performed.",
                "target_column is null",
                "Confirm the report explains why no model was built.")

        return issues
