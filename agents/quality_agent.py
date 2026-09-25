"""The Data Quality Agent: assess problems, then authorise proportionate repairs.

Detection is deterministic; the judgement is which repairs are warranted. That
judgement is the reason this stage needs a model at all. Imputing a column that
is 77% missing would manufacture data rather than recover it, while dropping a
column missing 2% of its values discards evidence needlessly. The correct action
depends on how much is missing, what the column means, and whether it is the
analytical target - which is exactly the kind of context-dependent call that
fixed rules handle badly.

Every authorised action is checked against the dataset before execution, and the
resulting log records what actually happened, including refusals and failures.
"""
from __future__ import annotations

import json

import pandas as pd

from agents.base import Agent
from agents.llm import LLMError, ask_json
from state import AgentOutput, WorkflowState
from tools.profiling import profile_dataset
from tools.quality import REPLAYABLE_ACTIONS, apply_cleaning, assess_quality
from verification import refused_cleaning_action

#: Authorised here but performed inside each cross-validation fold by the
#: modelling pipeline, where the fill cannot see held-out rows.
DEFERRED_ACTIONS = frozenset({"impute_median", "impute_mode"})

# Actions the agent is permitted to authorise. Anything outside this set is
# refused rather than executed, so the model cannot invent a transformation the
# framework has not implemented and validated.
ALLOWED_ACTIONS = {
    "drop_duplicates",
    "drop_column",
    "impute_median",
    "impute_mode",
    "to_numeric",
    "normalise_categories",
    "clip_outliers",
}

SYSTEM_PROMPT = """You are the Data Quality Agent in a multi-agent data science system.

You receive computed quality findings for a dataset. You do not calculate
anything and you do not invent problems that are not in the findings. You decide
which repairs are justified.

Available actions, and when each is appropriate:
  drop_duplicates       exact duplicate rows exist and are not meaningful repeats
  drop_column           the column is unusable: mostly missing, constant, or an identifier
  impute_median         a numeric column with a modest amount missing
  impute_mode           a categorical column with a small amount missing
  to_numeric            numeric values are stored as text
  normalise_categories  categories differ only by case or surrounding whitespace
  clip_outliers         extreme values are implausible measurement artefacts

Principles you must follow:
- Prefer the least destructive action that resolves the problem.
- Never impute a column missing more than about half its values; that fabricates
  data rather than recovering it. Drop it, or leave it and note the limitation.
- Never drop or impute the target column. Rows with a missing target should be
  reported as a limitation instead.
- Do not clip outliers that are plausible real measurements. Genuine variation is
  a finding to report, not an error to remove.
- Leave small amounts of missing data alone when the analysis tolerates it.
- It is acceptable to authorise no actions at all.

Respond with a single JSON object and nothing else:
{
  "actions": [{"action": "one of the allowed actions", "column": "column name or null", "reason": "why this action on this column"}],
  "declined": [{"issue": "issue kind", "column": "name", "reason": "why no repair is appropriate"}],
  "limitations": ["quality problems that remain after cleaning and affect conclusions"],
  "narrative": "a short paragraph for a human reader, stating concrete numbers from the findings"
}"""

REQUIRED = ("actions", "narrative")


def _compact_issues(assessment: dict, max_issues: int = 40) -> list[dict]:
    """Trim issues to the fields the model needs to make a decision."""
    return [
        {
            "kind": i["kind"],
            "severity": i["severity"],
            "column": i["column"],
            "detail": i["detail"],
        }
        for i in assessment["issues"][:max_issues]
    ]


class QualityAgent(Agent):
    name = "quality"
    description = (
        "Detects data quality problems - missing values, duplicates, outliers, "
        "inconsistent categories - and authorises proportionate cleaning actions."
    )
    requires = ("profiling",)

    def _execute(self, state: WorkflowState) -> AgentOutput:
        df = state.data
        warnings: list[str] = []
        profiling = state.findings("profiling")
        profile = state.outputs["profiling"].evidence["profile"]

        # 1. Deterministic detection.
        #
        # On a distributed run these two disagree about what they describe.
        # `profile` was computed by Spark across every row, so its missing
        # fractions, duplicate count and cardinalities are facts about the whole
        # dataset; `df` is the in-memory sample, so the checks `assess_quality`
        # performs itself -- IQR outlier fences, numeric-stored-as-text,
        # category variants -- describe the sample. Repairs are then applied to
        # the sample too, so "Removed 412 duplicate rows" could be written
        # beneath an issue quoting the full dataset's duplicate count.
        #
        # Cleaning the full dataset through Spark is the real answer and a much
        # larger change. Until then the mixture is stated rather than hidden,
        # which is the same thing the report already does for every other
        # sampled figure: a reader can discount a number they know is an
        # estimate, but not one they believe is a census.
        assessment = assess_quality(df, profile)
        sampled = state.spark_df is not None
        if sampled:
            warnings.append(
                f"Distributed run: issue *rates* were measured across all "
                f"{int(state.full_row_count or 0):,} rows, while the checks "
                f"performed here and the repairs applied describe the "
                f"{len(df):,}-row sample. Counts stated below are sample counts."
            )

        # A clean dataset needs no reasoning and no tokens spent on it.
        if not assessment["issues"]:
            return AgentOutput(
                agent=self.name,
                findings={
                    "quality_score": assessment["quality_score"],
                    "issue_counts": assessment["issue_counts"],
                    "actions_applied": [],
                    "limitations": [],
                    "rows_before": int(df.shape[0]),
                    "rows_after": int(df.shape[0]),
                    "columns_dropped": [],
                },
                narrative=(
                    f"No data quality problems were detected. The dataset has "
                    f"{df.shape[0]} rows and {df.shape[1]} columns with no missing "
                    f"values, duplicates or inconsistent categories."
                ),
                evidence={"assessment": assessment,
                          "cleaning_summary": {"rows_before": int(df.shape[0]),
                                               "rows_after": int(df.shape[0])}},
            )

        # 2. Model judgement on which repairs are warranted.
        user_prompt = (
            f"Dataset: {state.dataset_name}\n"
            f"Target column: {state.target_column or 'not identified'}\n"
            f"Task type: {state.task_type or 'unknown'}\n"
            f"Dataset description: {profiling.get('dataset_description', 'unknown')}\n\n"
            f"Overall quality score: {assessment['quality_score']}/100\n"
            f"Issue counts: {json.dumps(assessment['issue_counts'])}\n\n"
            f"Detected issues:\n{json.dumps(_compact_issues(assessment), indent=2)}\n\n"
            f"Column semantic types:\n"
            f"{json.dumps(profiling.get('column_types', {}), indent=2)}\n\n"
            "Decide which repairs are justified."
        )

        try:
            decision = ask_json(self.name, SYSTEM_PROMPT, user_prompt, required_keys=REQUIRED)
            status = "ok"
        except LLMError as exc:
            decision = self._fallback_decision(assessment, state)
            warnings.append(f"Model judgement unavailable, used conservative rules. ({exc})")
            status = "degraded"

        # 3. Screen every authorised action before executing it.
        approved, rejected = self._screen(decision.get("actions", []), df, state)
        warnings.extend(rejected)

        # Imputation is authorised here but not performed here. Filling a
        # column with its median across the whole frame, before modelling
        # cross-validates on that frame, lets every held-out row inform the
        # value its own training folds were filled with -- the leakage the
        # modelling pipeline is built to prevent, reintroduced one stage
        # earlier. It also erased the missingness itself before the feature
        # stage could record it, so a missing-value indicator could never fire
        # on an imputed column. The modelling pipeline already imputes inside
        # each fold (median for numbers, most frequent value for categories),
        # and the descriptive stages use only recorded values, so the fill is
        # left to the one place it can be done without leakage.
        # (A request naming a column that does not exist -- which reaches here
        # only when screening is switched off -- is left to `apply_cleaning`,
        # which records it as failed.)
        deferred = [a for a in approved if a["action"] in DEFERRED_ACTIONS
                    and isinstance(a["column"], str) and a["column"] in df.columns]
        approved = [a for a in approved if a not in deferred]

        # 4. Execute, and replace the working frame so downstream agents analyse
        #    the cleaned data while the original stays available for the report.
        cleaned, log = apply_cleaning(df, approved)
        for act in deferred:
            column = act["column"]
            log.append({
                "action": act["action"], "column": column, "status": "deferred",
                "detail": (f"{int(df[column].isna().sum())} missing values in "
                           f"{column} are left in place. The modelling stage "
                           f"fills them inside each cross-validation fold from "
                           f"that fold's training rows only, and the "
                           f"descriptive statistics use recorded values."),
            })
        rows_before, cols_before = df.shape
        state.working_df = cleaned

        # Repairs that change a column's type or values are recorded with the
        # parameters used, so a saved model can repeat them on new rows.
        state.preparation.extend(
            {"stage": "quality",
             **{k: entry[k] for k in ("action", "column", "lower", "upper")
                if k in entry}}
            for entry in log
            if entry["status"] == "applied" and entry["action"] in REPLAYABLE_ACTIONS)

        # Copies, because each gets the model's stated reason attached below.
        # Attaching it to the log entries themselves put model-written text --
        # with any figures it quoted -- into `cleaning_log`, which is evidence,
        # and evidence is the set of values a report is permitted to cite.
        applied = [dict(entry) for entry in log if entry["status"] == "applied"]
        failed = [entry for entry in log if entry["status"] in ("failed", "skipped")]
        for entry in failed:
            warnings.append(f"Action '{entry['action']}' on {entry['column']}: {entry['detail']}")

        # Attach each action's stated reason to what actually happened, so the
        # report can explain why the data was changed and not merely how.
        reasons = {
            (a.get("action"), a.get("column")): a.get("reason", "")
            for a in decision.get("actions", [])
        }
        for entry in applied:
            entry["reason"] = reasons.get((entry["action"], entry["column"]), "")
        deferred_log = [dict(entry, reason=reasons.get((entry["action"], entry["column"]), ""))
                        for entry in log if entry["status"] == "deferred"]

        dropped = [e["column"] for e in applied if e["action"] == "drop_column"]

        # Re-assess so the report can state the improvement rather than assert it.
        post = assess_quality(cleaned, profile_dataset(cleaned))

        findings = {
            # Stated in the findings, not only as a warning: warnings can be
            # trimmed out of an oversized prompt, and a reader who is told a
            # count is a sample count will read it correctly, while one who is
            # not will read it as a census.
            "measured_on": ("in-memory sample; rates from the full dataset"
                            if sampled else "the full dataset"),
            "quality_score": assessment["quality_score"],
            "quality_score_after": post["quality_score"],
            "issue_counts": assessment["issue_counts"],
            "issue_counts_after": post["issue_counts"],
            "actions_applied": applied,
            # Authorised, and carried out inside the modelling pipeline rather
            # than on this frame; see step 3.
            "actions_deferred": deferred_log,
            "actions_declined": decision.get("declined", []),
            "limitations": decision.get("limitations", []),
            "rows_before": int(rows_before),
            "rows_after": int(cleaned.shape[0]),
            "columns_before": int(cols_before),
            "columns_after": int(cleaned.shape[1]),
            "columns_dropped": dropped,
            "missing_cells_before": int(df.isna().sum().sum()),
            "missing_cells_after": int(cleaned.isna().sum().sum()),
        }

        # Keep the feature list consistent with the frame that now exists.
        if dropped:
            surviving = [f for f in profiling.get("usable_features", []) if f not in dropped]
            state.outputs["profiling"].findings["usable_features"] = surviving

        return AgentOutput(
            agent=self.name,
            status=status,
            findings=findings,
            narrative=decision.get("narrative", ""),
            evidence={"assessment": assessment, "assessment_after": post,
                      "cleaning_log": log,
                      # Computed here and nowhere else, so recorded as evidence:
                      # a report stating "8,412 rows after cleaning" must be
                      # able to find 8,412 among the computed values.
                      "cleaning_summary": {
                          key: findings[key] for key in (
                              "rows_before", "rows_after", "columns_before",
                              "columns_after", "missing_cells_before",
                              "missing_cells_after")}},
            warnings=warnings,
        )

    # -- guards -------------------------------------------------------------- #

    def _screen(self, actions: list[dict], df: pd.DataFrame,
                state: WorkflowState) -> tuple[list[dict], list[str]]:
        """Reject actions that are unsupported, misdirected, or destructive.

        These checks are deliberately independent of the prompt: the rules the
        model is asked to follow are enforced rather than trusted. The rules
        themselves live in `verification.refused_cleaning_action`, so the
        verification layer's ablation switch reaches them -- they previously ran
        here, where switching verification off left every refusal in place.

        The target is protected from every action, not from a list of three.
        The previous allow-list named `drop_column`, `impute_median` and
        `impute_mode`, which left `clip_outliers` free to compress a regression
        target's tails -- silently changing the very values every score in the
        report is measured against. Naming the one column that must not change
        cannot be forgotten when an action is added; an allow-list can.
        """
        approved: list[dict] = []
        rejected: list[str] = []

        for act in actions:
            if not isinstance(act, dict):
                continue
            kind, col = act.get("action"), act.get("column")
            try:
                present = col is not None and col in df.columns
            except TypeError:  # an unhashable name from the model
                present = False
            refusal = refused_cleaning_action(
                kind, ALLOWED_ACTIONS, column=col, columns=df.columns,
                target=state.target_column,
                missing_fraction=float(df[col].isna().mean()) if present else None,
                numeric=(bool(pd.api.types.is_numeric_dtype(df[col]))
                         if present else None))
            if refusal:
                rejected.append(refusal)
                continue
            approved.append({"action": kind,
                             "column": None if kind == "drop_duplicates" else col})

        return approved, rejected

    @staticmethod
    def _fallback_decision(assessment: dict, state: WorkflowState) -> dict:
        """Conservative rule-based cleaning used when the model is unreachable."""
        actions, limitations = [], []
        for issue in assessment["issues"]:
            col, kind = issue["column"], issue["kind"]
            if col == state.target_column:
                continue
            if kind == "duplicate_rows":
                actions.append({"action": "drop_duplicates", "column": None,
                                "reason": "Exact duplicate rows removed."})
            elif kind == "missing_values" and issue["severity"] == "critical":
                actions.append({"action": "drop_column", "column": col,
                                "reason": "Too sparse to impute."})
            elif kind == "missing_values" and issue["severity"] == "warning":
                limitations.append(f"{col} retains missing values.")
            elif kind == "inconsistent_categories":
                actions.append({"action": "normalise_categories", "column": col,
                                "reason": "Categories differ only by case or whitespace."})
            elif kind == "numeric_stored_as_text":
                actions.append({"action": "to_numeric", "column": col,
                                "reason": "Numeric values stored as text."})
        return {
            "actions": actions,
            "declined": [],
            "limitations": limitations,
            "narrative": "Cleaning applied using conservative rules; no model judgement available.",
        }
