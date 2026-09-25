"""The Feature Engineering Agent: build better inputs, and prove they are better.

The workflow previously went from cleaning straight to analysis, which left a
real gap: most of the accuracy available in a tabular problem sits in how the
data is represented, not in which model is fitted to it.

This agent is deliberately the most sceptical in the framework, because feature
engineering is where a data science pipeline most easily fools itself. Derived
columns always look like progress. They are trivial to produce, they make the
frame wider, and a wider frame feels richer. Whether they carry information the
model can use is a separate question, and one that is rarely asked.

So the agent measures. Features are engineered deterministically, scored against
the original frame on identical folds, and adopted only when the difference
exceeds what a resample would produce anyway. On this project's own datasets they
are rejected more often than not, which is the finding rather than a failure:
tree ensembles are invariant to monotonic transforms, so a log-transformed
column adds a dimension and no information, and a derived feature that restates
an existing one costs sample efficiency for nothing.

The language model interprets what was built and what the measurement showed. It
never decides whether to adopt: that follows from the number.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from agents.llm import LLMError, ask_json
from state import AgentOutput, WorkflowState
from tools.features import engineer_features, measure_feature_gain
from verification import invented_features, overstated_feature_benefit

SYSTEM_PROMPT = """You are the Feature Engineering Agent in a multi-agent data science system.

You receive a list of features that were derived from the dataset, the rationale
for each, transformations that were declined, and a cross-validated measurement
of whether the derived features improved predictive performance.

All of this has already been computed. You never compute a number, never invent
a feature, and never state a value that is not in the input.

How to read the measurement:
- The gain is the difference in cross-validated score with the derived features
  minus the score without them, on identical folds.
- A positive gain means the representation carries more usable signal.
- A gain at or below 0.002 is within resampling noise. Do NOT describe it as an
  improvement, however sensible the features look.
- A NEGATIVE gain is a real and reportable result. Say plainly that the derived
  features did not help and explain why that is plausible: tree ensembles are
  invariant to monotonic transforms such as a logarithm, and a derived feature
  that duplicates information already present adds dimensionality without
  adding signal.

Never claim a feature is important, predictive, or useful on the strength of its
rationale alone. The rationale explains why it was worth trying, not that it
worked.

Respond with a single JSON object and nothing else:
{
  "summary": "what was derived from this dataset and what the measurement showed",
  "useful_features": ["features most likely to carry signal, or an empty list if the measurement does not support any"],
  "why_it_helped_or_not": ["concrete reasons grounded in the measurement and the data"],
  "declined_rationale": ["why the declined transformations were the right thing to decline"],
  "narrative": "a short paragraph for a human reader, citing the measured gain"
}"""

REQUIRED = ("summary", "narrative")

class FeatureAgent(Agent):
    name = "features"
    description = (
        "Derives additional predictors from the existing columns and measures "
        "whether they improve predictive performance, adopting them only if they do."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []

        if not state.target_column or not state.task_type or state.task_type == "none":
            return AgentOutput(
                agent=self.name,
                findings={"skipped": True,
                          "reason": "No target column, so a gain cannot be measured."},
                narrative=(
                    "Feature engineering was not performed: without a target there "
                    "is no way to measure whether derived features help, and "
                    "adding unmeasured columns would be guesswork."
                ),
            )

        # 1. Derive, deterministically.
        result = engineer_features(df, state.target_column, state.task_type)
        engineered = result["frame"]

        if result["n_added"] == 0:
            return AgentOutput(
                agent=self.name,
                findings={"skipped": True, "n_added": 0,
                          "declined": result["declined"],
                          "reason": "The data offered no transformation worth applying."},
                narrative=(
                    "No features were derived: none of the transformations the "
                    "framework can apply were warranted by this data."
                ),
                evidence={"feature_result": {k: v for k, v in result.items()
                                             if k not in ("frame", "recipe")}},
            )

        # 2. Measure, on identical folds.
        measurement = measure_feature_gain(
            df, engineered, state.target_column, state.task_type)

        # 3. Adopt only on evidence. This assignment is the agent's only effect
        #    on the rest of the workflow, and it is driven by the measurement
        #    rather than by the interpretation that follows.
        adopted = bool(measurement.get("improved"))
        if adopted:
            state.working_df = engineered
            # How each adopted column was built, with its fitted parameters,
            # so a saved model can build it again from raw rows.
            state.preparation.extend(result["recipe"])

        # 4. Interpret.
        user_prompt = (
            f"Dataset: {state.dataset_name}\n"
            f"Target: {state.target_column} ({state.task_type})\n\n"
            f"Derived features:\n{json.dumps(result['applied'], indent=2, default=str)}\n\n"
            f"Declined transformations:\n"
            f"{json.dumps(result['declined'], indent=2, default=str)}\n\n"
            f"Measurement:\n{json.dumps(measurement, indent=2, default=str)}\n\n"
            f"The derived features were {'ADOPTED' if adopted else 'NOT adopted'} "
            f"on the strength of this measurement.\n\n"
            "Interpret what was derived and what the measurement shows."
        )

        try:
            interp = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            interp = self._fallback(result, measurement, adopted)
            warnings.append(f"Model interpretation unavailable, used computed results. ({exc})")
            status = "degraded"

        # 5. Check the interpretation against the measurement. Presenting
        #    rejected features as an improvement is the failure this stage
        #    invites, because derived columns look like progress whether or not
        #    they carry signal.
        if not adopted:
            text = " ".join([
                str(interp.get("summary", "")),
                " ".join(interp.get("why_it_helped_or_not", []) or []),
                str(interp.get("narrative", "")),
            ])
            if claimed := overstated_feature_benefit(text):
                warnings.append(
                    f"Derived features described as beneficial despite a measured "
                    f"gain of {measurement.get('gain')} "
                    f"({measurement.get('metric')}); terms used: {claimed}."
                )

            if interp.get("useful_features"):
                warnings.append(
                    "Features named as useful although the measurement did not "
                    "support adopting any of them."
                )

        # A feature named as useful must actually be one that was built.
        built = {a["feature"] for a in result["applied"]}
        invented = invented_features(interp.get("useful_features") or [], built)
        if invented:
            warnings.append(f"Features cited that were never derived: {invented}.")

        findings = {
            "n_added": result["n_added"],
            "n_columns_before": result["n_columns_before"],
            "n_columns_after": result["n_columns_after"],
            "features": [
                {"feature": a["feature"], "kind": a["kind"], "source": a["source"]}
                for a in result["applied"]
            ],
            "declined": [d["kind"] for d in result["declined"]],
            "adopted": adopted,
            "metric": measurement.get("metric"),
            "score_without_features": measurement.get("score_without_features"),
            "score_with_features": measurement.get("score_with_features"),
            "gain": measurement.get("gain"),
            "summary": interp.get("summary", ""),
            "useful_features": interp.get("useful_features", []),
            "why_it_helped_or_not": interp.get("why_it_helped_or_not", []),
            "declined_rationale": interp.get("declined_rationale", []),
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=interp.get("narrative", ""),
            evidence={"engineering": {k: v for k, v in result.items()
                                      if k not in ("frame", "recipe")},
                      "measurement": measurement},
            warnings=warnings,
        )

    @staticmethod
    def _fallback(result: dict[str, Any], measurement: dict[str, Any],
                  adopted: bool) -> dict[str, Any]:
        """State the measurement without interpretation."""
        gain = measurement.get("gain")
        verdict = (
            f"Derived features changed the cross-validated "
            f"{measurement.get('metric', 'score')} by {gain}, which "
            + ("justified adopting them." if adopted else
               "did not justify adopting them; the original columns were kept.")
        )
        return {
            "summary": verdict,
            "useful_features": [a["feature"] for a in result["applied"]] if adopted else [],
            "why_it_helped_or_not": [verdict],
            "declined_rationale": [d["reason"] for d in result["declined"]],
            "narrative": (
                f"{result['n_added']} feature(s) were derived from the existing "
                f"columns. {verdict} No model interpretation was available."
            ),
        }
