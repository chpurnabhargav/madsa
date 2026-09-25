"""The Model Recommendation Agent: fit baselines, then recommend on the evidence.

The agent does not recommend a model from the dataset's description alone. It
cross-validates a spread of candidates first and reasons about what actually
happened, so a recommendation is backed by a measured score rather than by a
plausible-sounding rule of thumb.

The result it must not soften is a negative one. When the fitted models do not
beat chance, the correct output is that the available features do not predict the
target, and the agent's claim to the contrary is overridden here rather than
trusted.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from agents.base import Agent
from config import MODELS_DIR
from agents.llm import (NULL_WORDS, LLMError, as_name, as_names, as_text,
                        ask_json)
from state import AgentOutput, WorkflowState
from tools.modelling import evaluate_models, feature_importance, fit_final_model
from verification import (hallucinated_columns, overridden_signal_claim,
                          unsupported_importance_claim)

SYSTEM_PROMPT = """You are the Model Recommendation Agent in a multi-agent data science system.

Baseline models have already been cross-validated on this dataset. You receive
their scores. You never invent scores and you never claim performance that was
not measured.

How to read the results:
- Compare every model against the baselines. A model that does not clearly beat
  them has found little usable signal, however high its raw score looks.
- A permutation test result is decisive. If it is not significant, the model's
  performance is indistinguishable from what shuffled labels produce, and you
  must say the features do not predict the target.
- On an imbalanced target, accuracy is misleading. Judge on macro F1 or ROC AUC.
- A large gap between a linear model and a tree ensemble indicates non-linear
  structure. A small gap means the simpler, more interpretable model is the
  better choice.
- Permutation importance below its own standard deviation is noise, not a driver.
- Report honestly. A finding of no predictive signal is a valid and useful result.

Respond with a single JSON object and nothing else:
{
  "recommended_model": "the model you recommend, or null if none is justified",
  "recommendation_reason": "why, citing the measured scores",
  "expected_performance": "what performance a user should expect, with the metric named",
  "evaluation_metrics": ["metrics that should be used for this problem, and why"],
  "important_features": ["features the importance analysis supports, strongest first"],
  "signal_assessment": "strong | moderate | weak | none",
  "improvements": ["concrete next steps that could improve performance"],
  "caveats": ["limitations of these results a reader must know"],
  "narrative": "two or three paragraphs for a human reader, citing concrete measured numbers"
}"""

REQUIRED = ("recommended_model", "signal_assessment", "narrative")


def _compact(evaluation: dict[str, Any], importance: dict[str, Any]) -> dict[str, Any]:
    """Reduce the modelling results to what the recommendation rests on."""
    out: dict[str, Any] = {
        "task": evaluation.get("task"),
        "target": evaluation.get("target"),
        "rows_used": evaluation.get("n_rows"),
        "cv_folds": evaluation.get("cv_folds"),
        "primary_metric": evaluation.get("primary_metric"),
        "numeric_features": evaluation.get("numeric_features", []),
        "categorical_features": evaluation.get("categorical_features", []),
        "excluded_features": evaluation.get("excluded_features", []),
        "models": [
            {"model": m["model"], "is_baseline": m["is_baseline"],
             "scores": m["scores"], "fit_time_s": m["fit_time_s"]}
            for m in evaluation.get("results", [])
        ],
        "best_model": evaluation.get("best_model"),
        "best_score": evaluation.get("best_score"),
        "strongest_baseline": evaluation.get("baseline_model"),
        "baseline_score": evaluation.get("baseline_score"),
        "lift_over_baseline": evaluation.get("lift_over_baseline"),
        "beats_baseline": evaluation.get("beats_baseline"),
    }

    if perm := evaluation.get("permutation_test"):
        out["permutation_test"] = {
            "observed_score": perm["score"],
            "score_on_shuffled_labels": f"{perm['null_mean']} +/- {perm['null_std']}",
            "p_value": perm["p_value"],
            "significant": perm["significant"],
            "interpretation": (
                "The model performs better than it does on shuffled labels, so it "
                "has learned real structure."
                if perm["significant"] else
                "The model performs no better than it does on shuffled labels, so "
                "its apparent performance is not evidence of a real relationship."
            ),
        }

    if importance.get("status") == "ok":
        out["feature_importance"] = {
            "measured_on": f"{importance['test_rows']} held-out rows",
            "metric": importance["metric"],
            "ranked": [
                {"feature": i["feature"], "importance": i["importance"],
                 "std": i["std"], "distinguishable_from_noise": i["informative"]}
                for i in importance["importances"]
            ],
        }
    elif importance.get("error"):
        out["feature_importance_error"] = importance["error"]

    return out


class ModellingAgent(Agent):
    name = "modelling"
    description = (
        "Cross-validates baseline models, tests whether performance exceeds "
        "chance, and recommends a modelling approach with expected performance."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []
        profiling = state.findings("profiling")

        target, task = state.target_column, state.task_type
        if not target or task not in ("classification", "regression"):
            return AgentOutput(
                agent=self.name,
                status="degraded",
                findings={"skipped": True,
                          "reason": "No supervised target was identified for this dataset."},
                narrative=(
                    "No modelling was performed because the dataset has no clear "
                    "target column for supervised learning."
                ),
                warnings=["Modelling skipped: no supervised target identified."],
            )

        features = [f for f in profiling.get("usable_features", []) if f in df.columns]

        # Profiling runs before feature engineering, so its list describes the
        # columns as they were on arrival. Any column present now but absent
        # from that list was derived by a later stage and adopted only because
        # it measurably improved a prediction -- excluding it here would make
        # that whole stage inert, engineering features and then declining to
        # use them.
        # Everything profiling saw: the columns it judged usable, plus the ones
        # it deliberately excluded. Anything outside both sets did not exist
        # when profiling ran. Using only `usable_features` here would silently
        # reinstate the identifiers and unusable columns profiling rejected.
        # `c.get("column")` alone contributed `None` for any entry naming the
        # column under a different key, so the real name never entered the set
        # and the column was reinstated as "derived after profiling" -- with a
        # warning that actively misdescribed what had happened. Profiling now
        # normalises these entries, and this reads them defensively anyway,
        # because the cost of being wrong here is a row identifier fitted as a
        # predictor.
        excluded_names = {
            c if isinstance(c, str)
            else next((str(c[k]) for k in ("column", "name", "col", "feature")
                       if isinstance(c, dict) and c.get(k)), None)
            for c in (profiling.get("excluded_columns") or [])
        }
        seen_at_profiling = (
            set(profiling.get("usable_features") or [])
            | {name for name in excluded_names if name}
        )
        derived = [c for c in df.columns
                   if c != target and c not in seen_at_profiling and c not in features]
        if derived:
            features = features + derived
            warnings.append(
                f"Included {len(derived)} feature(s) derived after profiling: {derived}."
            )

        if not features:
            features = [c for c in df.columns if c != target]
            warnings.append("No feature list from profiling; used all remaining columns.")

        # 1. Fit and measure.
        evaluation = evaluate_models(df, target, features, task)
        if evaluation.get("status") != "ok":
            return AgentOutput(
                agent=self.name,
                status="failed",
                error=evaluation.get("error", "Model evaluation failed."),
                findings={"evaluation_failed": True, "reason": evaluation.get("error")},
                narrative=f"Modelling could not be completed: {evaluation.get('error')}",
                evidence={"evaluation": evaluation},
            )

        importance = feature_importance(df, target, features, task,
                                        model_name=evaluation["best_model"])
        if importance.get("status") != "ok":
            warnings.append(f"Feature importance unavailable: {importance.get('error')}")

        if excluded := evaluation.get("excluded_features"):
            warnings.append(
                f"Excluded from modelling as unusable or too high-cardinality: {excluded}."
            )
        if dropped := evaluation.get("rows_dropped_missing_target"):
            warnings.append(f"{dropped} rows were dropped for having no target value.")

        # 2. Interpret.
        user_prompt = (
            f"Dataset: {state.dataset_name}\n"
            f"Description: {profiling.get('dataset_description', '')}\n"
            f"Task: {task} on target '{target}'\n\n"
            f"Measured results:\n"
            f"{json.dumps(_compact(evaluation, importance), indent=2, default=str)}\n\n"
            "Recommend a modelling approach based on these measurements."
        )

        try:
            rec = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            rec = self._fallback(evaluation, importance)
            warnings.append(f"Model interpretation unavailable, used measured results. ({exc})")
            status = "degraded"

        # The model's values are coerced before anything compares them. It has
        # returned feature lists as objects and assessments as one-element
        # lists; `str(["strong"])` is "['strong']", which matched no level and
        # let an overstated claim straight past the override below, and an
        # unhashable feature entry raised in the membership test after every
        # model had already been fitted.
        rec["important_features"] = as_names(rec.get("important_features"))
        recommended = as_text(rec.get("recommended_model")).strip()
        rec["recommended_model"] = (None if recommended.lower() in NULL_WORDS
                                    else recommended)
        rec["signal_assessment"] = as_name(rec.get("signal_assessment")) or None

        # 3. Override the agent where the measurements contradict it. A negative
        #    result is the one an LLM is most likely to soften, and it is the one
        #    that most misleads a reader if softened.
        beats = evaluation.get("beats_baseline", False)
        perm = evaluation.get("permutation_test")
        claimed = rec["signal_assessment"] or ""

        if overridden_signal_claim(claimed, beats, rec.get("recommended_model")):
            measured = (f"lift over baseline {evaluation.get('lift_over_baseline')}"
                        + (f", permutation p = {perm['p_value']}" if perm else ""))
            if claimed in ("strong", "moderate"):
                warnings.append(
                    f"Signal assessment '{claimed}' contradicts the measurements "
                    f"({measured}); corrected to 'none'.")
                rec["signal_assessment"] = "none"
            if rec.get("recommended_model"):
                # Withdrawn whatever the assessment said. Only 'strong' and
                # 'moderate' triggered this before, so a model assessed as
                # 'weak' stayed recommended although nothing beat guessing --
                # and the validator's objection to it could never be cleared,
                # because a revision re-runs the report, not this stage.
                warnings.append(
                    f"'{rec['recommended_model']}' was recommended although no "
                    f"model beat the baseline ({measured}); the recommendation "
                    f"was withdrawn.")
            rec["recommended_model"] = None
            rec["recommendation_reason"] = (
                "No model is recommended: cross-validated performance is not "
                "distinguishable from chance on this target with these features."
            )

        # A feature named as important must exist, and must be one the
        # importance analysis found distinguishable from noise.
        claimed_features = rec["important_features"]
        invented = hallucinated_columns(claimed_features, list(df.columns))
        if invented:
            warnings.append(f"Ignored feature names not present in the data: {invented}.")
            rec["important_features"] = [f for f in claimed_features
                                         if f not in invented]
        if importance.get("status") == "ok":
            unsupported = unsupported_importance_claim(
                [f for f in rec["important_features"] if f in df.columns],
                set(importance["informative_features"]))
            if unsupported:
                warnings.append(
                    f"Features cited as important but not distinguishable from noise: "
                    f"{unsupported}."
                )

        # Persist the winning model. Every pipeline fitted above was fitted on
        # a fold and discarded, so a report recommending a model was pointing
        # at something that no longer existed. This refits the winner on the
        # whole dataset and saves it next to the run, with a signature naming
        # the columns it expects.
        #
        # Only a model that beat the baseline is saved. A baseline is a
        # yardstick, not a recommendation -- and neither is a model that failed
        # to clear it. The winner on pure noise used to be saved anyway, so
        # `predict.py --latest` would score new data with a model the report
        # had explicitly rejected, without any warning.
        saved, not_saved = None, None
        if beats and not evaluation.get("best_is_baseline") and evaluation.get("best_model"):
            saved = self._persist(state, df, target, features, task, evaluation,
                                  warnings)
        elif not evaluation.get("best_is_baseline"):
            not_saved = ("No model was saved: none beat the baseline by more "
                         "than the variation between folds, so there is "
                         "nothing here a reader should deploy.")

        findings = {
            "task": task,
            "target": target,
            "recommended_model": rec.get("recommended_model"),
            "recommendation_reason": rec.get("recommendation_reason", ""),
            "expected_performance": rec.get("expected_performance", ""),
            "evaluation_metrics": rec.get("evaluation_metrics", []),
            "important_features": rec.get("important_features", []),
            "signal_assessment": rec.get("signal_assessment"),
            "improvements": rec.get("improvements", []),
            "caveats": rec.get("caveats", []),
            # The measured facts, kept beside the interpretation so the report
            # and the validator read the same numbers the agent saw.
            "best_model": evaluation["best_model"],
            "best_score": evaluation["best_score"],
            "primary_metric": evaluation["primary_metric"],
            "baseline_score": evaluation["baseline_score"],
            "lift_over_baseline": evaluation["lift_over_baseline"],
            "beats_baseline": beats,
            "permutation_p_value": perm["p_value"] if perm else None,
            # Counts rather than a score: how many of each outcome the model
            # actually finds, and how often it raises a false alarm. Computed
            # out-of-sample, and carried in findings so the report and the
            # validator both see it.
            "operating_behaviour": evaluation.get("operating_behaviour"),
            # Where the fitted model was saved, so the recommendation in the
            # report is something a reader can actually obtain and use -- or
            # why nothing was saved.
            "saved_model": saved,
            "saved_model_note": not_saved,
            "model_scores": {
                m["model"]: m["primary_score"] for m in evaluation["results"]
            },
            "informative_features": importance.get("informative_features", []),
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=rec.get("narrative", ""),
            evidence={"evaluation": evaluation, "feature_importance": importance},
            warnings=warnings,
        )

    @staticmethod
    def _fallback(evaluation: dict[str, Any], importance: dict[str, Any]) -> dict[str, Any]:
        """State the measured outcome without interpretation."""
        beats = evaluation.get("beats_baseline", False)
        metric = evaluation.get("primary_metric", "score")
        return {
            "recommended_model": evaluation["best_model"] if beats else None,
            "recommendation_reason": (
                f"{evaluation['best_model']} achieved the highest cross-validated "
                f"{metric} of {evaluation['best_score']}."
                if beats else
                "No model beat the baseline by a margin distinguishable from noise."
            ),
            "expected_performance": f"{metric} of approximately {evaluation['best_score']}"
                                    if beats else "no better than chance",
            "evaluation_metrics": [metric],
            "important_features": importance.get("informative_features", []),
            "signal_assessment": "moderate" if beats else "none",
            "improvements": [],
            "caveats": ["Interpretation was generated from measurements without a model."],
            "narrative": "Models were cross-validated; results are reported as measured.",
        }

    @staticmethod
    def _persist(state: WorkflowState, df, target, features, task,
                 evaluation: dict, warnings: list[str]) -> dict[str, Any] | None:
        """Refit the winner on all rows and write it beside the run record."""
        best_params = next(
            (r.get("params") for r in evaluation.get("results", [])
             if r.get("model") == evaluation["best_model"] and r.get("tuned")),
            None)
        try:
            # The run's preparation record goes with the model, so the saved
            # pipeline repeats it on raw rows: without it the signature listed
            # derived columns like `Fare_log` that no raw file contains.
            result = fit_final_model(df, target, features, task,
                                     evaluation["best_model"], best_params,
                                     preparation=state.preparation)
        except Exception as exc:
            warnings.append(f"The model could not be refitted for saving. ({exc})")
            return None
        if not result or (isinstance(result[0], dict)
                          and result[0].get("status") == "failed"):
            warnings.append("The model could not be refitted for saving.")
            return None

        pipeline, signature = result
        signature["dataset"] = state.dataset_name
        stem = f"{Path(state.dataset_name).stem}_{datetime.now():%Y%m%d_%H%M%S}"
        path = MODELS_DIR / f"{stem}.joblib"
        try:
            joblib.dump(pipeline, path)
            (MODELS_DIR / f"{stem}_signature.json").write_text(
                json.dumps(signature, indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            # Saving is a convenience; failing to save must not fail a stage
            # that has already produced every number the report needs.
            warnings.append(f"The model could not be written to disk. ({exc})")
            return None

        return {"path": str(path),
                "signature": str(MODELS_DIR / f"{stem}_signature.json"),
                "model": signature["model"],
                "expects_columns": signature["expects_columns"]}
