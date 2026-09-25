"""The Multivariate Agent: analyse the feature space as a whole.

Where the EDA Agent examines columns one or two at a time, this agent reports on
structure that only appears across many features at once: how much redundancy the
feature space carries, whether the rows fall into natural groups, which
predictors cannot be interpreted independently, and which features carry
information a correlation would miss.

The agent was added to the framework after the other seven were complete, and
adding it required no change to any existing agent. That is the modularity claim
the architecture makes, demonstrated rather than asserted.

Two results here are easy to overstate and are guarded against. Clusters are
reported as segments only when the silhouette score supports it, and a
mutual-information ranking is not allowed to be described as importance, since it
measures dependence rather than predictive contribution.
"""
from __future__ import annotations

import json
from typing import Any

from agents.base import Agent
from agents.llm import LLMError, ask_json
from state import AgentOutput, WorkflowState
from tools.multivariate import run_multivariate
from verification import (overstated_segmentation,
                         unsupported_non_linear_signal)

SYSTEM_PROMPT = """You are the Multivariate Analysis Agent in a multi-agent data science system.

You receive the results of principal component analysis, k-means clustering,
variance inflation factors, correlation grouping and mutual information. All of
them have already been computed. You never calculate values and never state a
number that is not in the input.

How to read the results:
- PCA. If far fewer components than features retain most of the variance, the
  feature space is redundant. Describe what the leading components represent by
  reading their loadings; features loading together vary together.
- Clustering. Judge on the silhouette score, not on the fact that k-means
  returned clusters. Below 0.25 the groupings are weak and must NOT be presented
  as segments; say the data does not separate into natural groups.
- Variance inflation. Above 10 means a feature is strongly predictable from the
  others, so its coefficient in a linear model cannot be interpreted alone. This
  affects interpretation, not predictive accuracy.
- Mutual information. It captures non-linear dependence. A feature ranked high
  here but low by correlation is related to the target in a way a linear model
  cannot exploit, which is a finding worth stating. It measures dependence, not
  importance; do not call it importance.

Respond with a single JSON object and nothing else:
{
  "structure_summary": "what the feature space looks like as a whole",
  "redundancy_findings": ["what is redundant, and what could be dropped or combined"],
  "cluster_findings": ["what the clustering shows, or that no meaningful grouping exists"],
  "interpretation_warnings": ["features whose coefficients cannot be read individually"],
  "non_linear_signals": ["features related to the target in a way correlation misses"],
  "modelling_implications": ["what this means for feature selection and model choice"],
  "narrative": "a short paragraph for a human reader, citing concrete numbers"
}"""

REQUIRED = ("structure_summary", "narrative")

def _compact(results: dict[str, Any]) -> dict[str, Any]:
    """Reduce the computed results to what the interpretation needs."""
    out: dict[str, Any] = {}

    if pca := results.get("pca"):
        out["pca"] = {
            "n_features": pca["n_features"],
            "components_for_90pct_variance": pca["components_for_target"],
            "redundancy_fraction": pca["redundancy"],
            # Null when the variance target was not reached within the
            # components computed. The reason travels with it so the
            # interpretation states why the figure is absent rather than
            # guessing at one.
            "components_not_established": pca.get("target_not_reached"),
            "explained_variance_ratio": pca["explained_variance_ratio"][:6],
            "leading_components": pca["leading_components"],
        }

    if clustering := results.get("clustering"):
        out["clustering"] = {
            "best_k": clustering["best_k"],
            "silhouette": clustering["silhouette"],
            "structure_strength": clustering["structure_strength"],
            "meaningful": clustering["meaningful"],
            "scores_by_k": clustering["scores_by_k"],
            "clusters": [
                {k: v for k, v in c.items() if k != "labels"}
                for c in clustering["clusters"]
            ],
        }

    if vif := results.get("multicollinearity"):
        out["variance_inflation"] = {
            "factors": vif["factors"][:10],
            "severe": vif["severe_multicollinearity"],
        }

    if groups := results.get("correlation_groups"):
        out["correlation_groups"] = {
            "redundant_groups": groups["redundant_groups"],
            "n_redundant_features": groups["n_redundant_features"],
        }

    if relevance := results.get("feature_relevance"):
        out["mutual_information"] = {
            "ranked": relevance["ranked"][:12],
            "informative": relevance["informative_features"],
            "uninformative": relevance["uninformative_features"],
        }

    return out


class MultivariateAgent(Agent):
    name = "multivariate"
    description = (
        "Analyses the feature space as a whole: dimensionality and redundancy, "
        "natural groupings, multicollinearity, and non-linear dependence on the target."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []

        # 1. Compute.
        results = run_multivariate(df, state.target_column, state.task_type)

        if not results.get("applicable"):
            return AgentOutput(
                agent=self.name,
                findings={"skipped": True,
                          "reason": "Too few numeric features or rows for multivariate analysis."},
                narrative=(
                    "Multivariate analysis was not performed: the dataset does not "
                    "have enough numeric columns or rows for it to be informative."
                ),
                evidence={"multivariate_results": results},
            )

        # 2. Interpret.
        user_prompt = (
            f"Dataset: {state.dataset_name}\n"
            f"Description: {state.findings('profiling').get('dataset_description', '')}\n"
            f"Target: {state.target_column or 'none'} ({state.task_type or 'unknown'})\n\n"
            f"Computed multivariate results:\n"
            f"{json.dumps(_compact(results), indent=2, default=str)}\n\n"
            "Interpret the structure of this feature space."
        )

        try:
            interp = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            interp = self._fallback(results)
            warnings.append(f"Model interpretation unavailable, used computed results. ({exc})")
            status = "degraded"

        # 3. Check the interpretation against the measurements. Weak clustering
        #    described as segmentation is the failure this stage invites, because
        #    k-means always returns clusters whether or not any exist.
        clustering = results.get("clustering")
        if clustering and not clustering["meaningful"]:
            text = " ".join([
                str(interp.get("structure_summary", "")),
                " ".join(interp.get("cluster_findings", []) or []),
                str(interp.get("narrative", "")),
            ]).lower()
            claimed = overstated_segmentation(text)
            if claimed:
                warnings.append(
                    f"Clustering described as segmentation despite a silhouette of "
                    f"{clustering['silhouette']} ({clustering['structure_strength']}); "
                    f"terms used: {claimed}."
                )

        # A feature named as carrying non-linear signal must actually rank as
        # informative by mutual information.
        if relevance := results.get("feature_relevance"):
            informative = set(relevance["informative_features"])
            claimed_features = [
                f for f in (interp.get("non_linear_signals") or [])
                if isinstance(f, str) and f in df.columns
            ]
            unsupported = unsupported_non_linear_signal(claimed_features, informative)
            if unsupported:
                warnings.append(
                    f"Features cited as carrying non-linear signal but not informative "
                    f"by mutual information: {unsupported}."
                )

        pca = results.get("pca") or {}
        vif = results.get("multicollinearity") or {}

        findings = {
            "structure_summary": interp.get("structure_summary", ""),
            "redundancy_findings": interp.get("redundancy_findings", []),
            "cluster_findings": interp.get("cluster_findings", []),
            "interpretation_warnings": interp.get("interpretation_warnings", []),
            "non_linear_signals": interp.get("non_linear_signals", []),
            "modelling_implications": interp.get("modelling_implications", []),
            # The measured facts, kept beside the interpretation.
            "n_features_analysed": pca.get("n_features"),
            "components_for_90pct_variance": pca.get("components_for_target"),
            "redundancy_fraction": pca.get("redundancy"),
            "best_k": (clustering or {}).get("best_k"),
            "silhouette": (clustering or {}).get("silhouette"),
            "clustering_meaningful": (clustering or {}).get("meaningful"),
            "severe_multicollinearity": vif.get("severe_multicollinearity", []),
            "mutual_information_top": [
                r["feature"] for r in
                (results.get("feature_relevance") or {}).get("ranked", [])[:5]
            ],
        }

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=interp.get("narrative", ""),
            evidence={"multivariate_results": results},
            warnings=warnings,
        )

    @staticmethod
    def _fallback(results: dict[str, Any]) -> dict[str, Any]:
        """State the computed structure without interpretation."""
        pca = results.get("pca") or {}
        clustering = results.get("clustering") or {}
        vif = results.get("multicollinearity") or {}
        relevance = results.get("feature_relevance") or {}

        redundancy = []
        if pca:
            redundancy.append(
                f"{pca['components_for_target']} of {pca['n_features']} components "
                f"retain 90% of the variance."
            )
        if groups := (results.get("correlation_groups") or {}).get("redundant_groups"):
            redundancy.append(f"Correlated feature groups: {groups}.")

        cluster_findings = []
        if clustering:
            cluster_findings.append(
                f"k-means selected k={clustering['best_k']} with a silhouette of "
                f"{clustering['silhouette']} ({clustering['structure_strength']}); "
                + ("the grouping is weak and should not be treated as segmentation."
                   if not clustering["meaningful"] else
                   "the grouping is strong enough to report.")
            )

        return {
            "structure_summary": "Computed without model interpretation.",
            "redundancy_findings": redundancy,
            "cluster_findings": cluster_findings,
            "interpretation_warnings": (
                [f"{f} has severe multicollinearity."
                 for f in vif.get("severe_multicollinearity", [])]),
            "non_linear_signals": relevance.get("informative_features", [])[:5],
            "modelling_implications": [],
            "narrative": "Multivariate structure computed; no model interpretation "
                         "was available.",
        }
