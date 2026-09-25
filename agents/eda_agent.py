"""The EDA Agent: turn computed statistics into stated findings.

The statistics are produced by `tools.eda`; the agent decides which of them are
worth reporting and what they mean. Two failure modes are guarded against, and
both are known weaknesses of language models used for analysis:

* Reporting significance as importance. On a large dataset a negligible effect
  can have a very small p-value, so the agent is instructed to rank findings by
  effect size and every claim it makes is checked against the computed effect.
* Asserting causation from association. The prompt forbids causal language, and
  observational limitations are recorded as an explicit finding.

Each insight the agent returns must name the statistic it rests on, so the
Validation Agent can later verify the claim rather than trust it.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from agents.llm import LLMError, ask_json
from state import AgentOutput, WorkflowState
import pandas as pd

from tools.eda import run_eda
from tools import spark_engine
from verification import causal_language, overstated_effect

SYSTEM_PROMPT = """You are the Exploratory Data Analysis Agent in a multi-agent data science system.

You receive statistics that have already been computed. You never calculate
values, and you never state a number that does not appear in the input.

How to judge what matters:
- Rank findings by EFFECT SIZE, not by p-value. A p-value says an effect is
  distinguishable from noise; it does not say the effect is large enough to care
  about. On a large sample a negligible effect can have a tiny p-value.
- Effect size guides: Cramer's V and correlation - 0.1 weak, 0.3 moderate,
  0.5 strong. Eta-squared - 0.01 weak, 0.06 moderate, 0.14 strong.
- An effect that is significant but negligible should be described as
  detectable but unimportant, or omitted.
- When Spearman is much larger than Pearson the relationship is monotonic but
  non-linear, which matters for model choice. Say so.

Rules you must follow:
- Never claim causation. These are observational associations. Write "is
  associated with", not "causes", "leads to", "drives" or "results in".
- Every insight must cite the specific statistic supporting it.
- Report notable absences of relationship too; they are informative.
- Do not speculate beyond what the statistics show.

Respond with a single JSON object and nothing else:
{
  "key_insights": [
    {"insight": "what the data shows, in plain language",
     "evidence": "the exact statistic supporting it, including its value",
     "importance": "high | medium | low"}
  ],
  "relationships": ["notable associations between features, phrased non-causally"],
  "distribution_notes": ["anything about distribution shape that affects modelling"],
  "data_observations": ["patterns worth noting, including absences of relationship"],
  "modelling_implications": ["what these findings imply for the modelling stage"],
  "narrative": "two or three paragraphs a human reader would understand, citing concrete numbers"
}"""

REQUIRED = ("key_insights", "narrative")

def _compact(results: dict[str, Any]) -> dict[str, Any]:
    """Reduce the statistics to what the model needs to judge them.

    Full contingency tables and every group's standard deviation would dominate
    the prompt. What survives is the effect size, the direction, and enough
    detail to describe the pattern concretely.
    """
    out: dict[str, Any] = {"target": results.get("target"),
                           "task_type": results.get("task_type")}

    if tb := results.get("target_balance"):
        out["target_balance"] = {
            "classes": tb["classes"],
            "imbalance_ratio": tb["imbalance_ratio"],
            "imbalanced": tb["imbalanced"],
            "severely_imbalanced": tb["severely_imbalanced"],
        }

    if assoc := results.get("categorical_associations"):
        out["categorical_vs_target"] = [
            {"feature": a["feature"], "cramers_v": a["cramers_v"],
             "strength": a["strength"], "p_value": a["p_value"],
             "rates_by_level": a["rates_by_level"]}
            for a in assoc[:10]
        ]

    if tc := results.get("target_correlations"):
        out["numeric_vs_target"] = [
            {"feature": t["feature"], "pearson": t["pearson"],
             "strength": t["strength"], "direction": t["direction"],
             "p_value": t["p_value"]}
            for t in tc[:10]
        ]

    if gd := results.get("group_differences"):
        out["group_differences"] = [
            {"numeric": g["numeric_column"], "grouped_by": g["group_column"],
             "eta_squared": g["eta_squared"], "effect_size": g["effect_size"],
             "p_value": g["p_value"], "significant": g["significant"],
             "highest": g["spread"]["highest"], "lowest": g["spread"]["lowest"],
             "group_means": {x["group"]: x["mean"] for x in g["groups"]}}
            for g in gd[:10]
        ]

    if pairs := results.get("correlations", {}).get("pairs"):
        out["feature_correlations"] = [
            {"pair": [p["column_a"], p["column_b"]], "pearson": p["pearson"],
             "spearman": p["spearman"], "strength": p["strength"],
             "non_linear": p["non_linear"]}
            for p in pairs[:12]
        ]

    if dists := results.get("distributions"):
        out["distributions"] = [
            {"column": d["column"], "shape": d["shape"], "skewness": d["skewness"],
             "mean": d["mean"], "median": d["median"],
             "approximately_normal": d["approximately_normal"],
             "suggests_transform": d["suggests_transform"]}
            for d in dists[:15]
        ]

    return out


def _largest_effects(results: dict[str, Any]) -> dict[str, float]:
    """The strongest effect found by each analysis, used to check the agent's claims."""
    effects: dict[str, float] = {}
    for a in results.get("categorical_associations", []) or []:
        effects[a["feature"]] = max(effects.get(a["feature"], 0), a["cramers_v"])
    for t in results.get("target_correlations", []) or []:
        effects[t["feature"]] = max(effects.get(t["feature"], 0), abs(t["pearson"]))
    for g in results.get("group_differences", []) or []:
        col = g["numeric_column"]
        effects[col] = max(effects.get(col, 0), g["eta_squared"])
    return effects


class EDAAgent(Agent):
    name = "eda"
    description = (
        "Performs statistical exploration - correlations, group differences, "
        "distributions and target balance - and states what the results mean."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []
        profiling = state.findings("profiling")

        # Exploratory analysis describes the data as collected, not the
        # representation built for the model. The feature stage runs first and
        # replaces the working frame, so its derived columns were being
        # correlated alongside the originals they are functions of -- on
        # titanic, 8 of 13 reported pairs involved one, and the single strongest
        # correlation in the report was `Fare` against `Fare_log` at 0.79: a
        # column against its own logarithm, a tautology presented as the
        # headline finding.
        #
        # Reordering the stages is the alternative, but modelling wants the
        # engineered frame and EDA wants the original, so the frame is not the
        # thing to move. Excluding the derived columns here says exactly what is
        # meant: these are a modelling device, and describing them as
        # relationships in the data would be describing an artefact of this
        # pipeline.
        derived = {f["feature"] for f in
                   (state.findings("features").get("features") or [])}
        described = df.drop(columns=[c for c in derived if c in df.columns])
        if derived:
            warnings.append(
                f"{len(derived)} derived feature(s) were excluded from the "
                f"exploratory analysis; they are functions of columns already "
                f"described here."
            )

        # 1. Compute.
        results = run_eda(described, target=state.target_column,
                          task_type=state.task_type)

        # On a distributed run the correlation matrix is recomputed across every
        # row rather than the in-memory sample. Correlations are the finding this
        # stage is most often quoted on, and a coefficient from a 50,000-row
        # sample of three million is an estimate where the full one is a fact.
        # The Spark result uses the identical contract, so it is substituted
        # directly and nothing downstream changes.
        if state.spark_df is not None:
            # `column_types` is what profiling exposes in its findings -- a
            # name-to-semantic-type mapping. This read `columns`, which lives in
            # profiling's *evidence*, so the lookup always returned an empty
            # list and control always fell through to the dtype fallback below.
            # The semantic filter never ran, and a distributed run correlated
            # every numeric column including row identifiers, then reported that
            # matrix as the headline finding.
            column_types = state.findings("profiling").get("column_types", {})
            numeric = [name for name, kind in column_types.items()
                       if kind in ("numeric", "categorical_numeric")
                       and name in df.columns]
            if not numeric:
                # Only when profiling produced no types at all. Identifiers are
                # still excluded by dtype here where they can be recognised.
                numeric = [c for c in df.columns
                           if pd.api.types.is_numeric_dtype(df[c])]
            try:
                distributed = spark_engine.correlations(
                    state.spark_df, numeric,
                    exclude=(state.target_column,) if state.target_column else ())
                results["correlations"] = distributed
                warnings.append(
                    f"Correlations were computed on Spark across all "
                    f"{state.full_row_count:,} rows rather than the in-memory sample."
                )
            except Exception as exc:
                warnings.append(
                    f"Distributed correlation failed, used the in-memory sample. ({exc})"
                )

        n_findings = (
            len(results.get("correlations", {}).get("pairs", []))
            + len(results.get("categorical_associations", []) or [])
            + len(results.get("group_differences", []) or [])
            + len(results.get("target_correlations", []) or [])
        )
        if n_findings == 0:
            warnings.append(
                "No relationships passed the reporting thresholds; the dataset may "
                "have few usable numeric or categorical columns."
            )

        # 2. Interpret.
        user_prompt = (
            f"Dataset: {state.dataset_name}\n"
            f"Description: {profiling.get('dataset_description', 'unknown')}\n"
            f"Target: {state.target_column or 'none identified'} "
            f"({state.task_type or 'unknown task'})\n"
            f"{f'Stated analysis goal: {state.user_goal}' if state.user_goal else ''}\n\n"
            f"Computed statistics:\n{json.dumps(_compact(results), indent=2, default=str)}\n\n"
            "Interpret these results. Rank by effect size, not significance."
        )

        try:
            interp = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            interp = self._fallback(results)
            warnings.append(f"Model interpretation unavailable, used ranked statistics. ({exc})")
            status = "degraded"

        # 3. Check the interpretation against the computed statistics.
        insights = interp.get("key_insights", [])
        effects = _largest_effects(results)

        # Insights are model-shaped: an entry that is not an object carries no
        # importance or evidence to check, and reading one raised here.
        insights = [ins for ins in insights if isinstance(ins, dict)]

        overstated = []
        for ins in insights:
            if ins.get("importance") != "high":
                continue
            # A high-importance claim should rest on a column with a real effect.
            text = str(ins.get("insight", ""))
            mentioned = {c: effects[c] for c in effects if c.lower() in text.lower()}
            if overstated_effect(mentioned):
                overstated.append(text[:60])
        if overstated:
            warnings.append(
                f"{len(overstated)} insight(s) marked high importance rest on negligible "
                f"effect sizes: {overstated}"
            )

        causal = [
            str(ins.get("insight", ""))[:60] for ins in insights
            if causal_language(str(ins.get("insight", "")))
        ]
        if causal:
            warnings.append(
                f"{len(causal)} insight(s) use causal language for observational "
                f"associations: {causal}"
            )

        uncited = sum(1 for ins in insights if not str(ins.get("evidence", "")).strip())
        if uncited:
            warnings.append(f"{uncited} insight(s) cite no supporting statistic.")

        findings = {
            "key_insights": insights,
            "relationships": interp.get("relationships", []),
            "distribution_notes": interp.get("distribution_notes", []),
            "data_observations": interp.get("data_observations", []),
            "modelling_implications": interp.get("modelling_implications", []),
            "n_statistics_computed": n_findings,
            "target_balance": results.get("target_balance"),
            "strongest_effects": dict(
                sorted(effects.items(), key=lambda kv: -kv[1])[:5]
            ),
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=interp.get("narrative", ""),
            evidence={"eda_results": results,
                      "n_statistics_computed": n_findings},
            warnings=warnings,
        )

    @staticmethod
    def _fallback(results: dict[str, Any]) -> dict[str, Any]:
        """Rank the computed statistics without interpretation."""
        insights = []
        for a in (results.get("categorical_associations") or [])[:3]:
            insights.append({
                "insight": f"{a['feature']} is associated with the target "
                           f"({a['strength']} association).",
                "evidence": f"Cramer's V = {a['cramers_v']}, p = {a['p_value']}",
                "importance": "high" if a["cramers_v"] >= 0.3 else "medium",
            })
        for g in (results.get("group_differences") or [])[:3]:
            if g["eta_squared"] < 0.01:
                continue
            insights.append({
                "insight": f"{g['numeric_column']} differs across levels of "
                           f"{g['group_column']} ({g['effect_size']} effect).",
                "evidence": f"eta-squared = {g['eta_squared']}, p = {g['p_value']}",
                "importance": "high" if g["eta_squared"] >= 0.06 else "medium",
            })
        return {
            "key_insights": insights,
            "relationships": [],
            "distribution_notes": [],
            "data_observations": [],
            "modelling_implications": [],
            "narrative": "Statistics computed and ranked by effect size; "
                         "no model interpretation was available.",
        }
