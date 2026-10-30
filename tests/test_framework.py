"""Test suite for the framework.

The tests concentrate on the framework's reliability guarantees rather than on
restating what each function computes. The claim the project makes is that model
output is verified before it reaches the report, so the cases worth testing are
the ones where a model returns something wrong: a column that does not exist, a
destructive cleaning action, an overstated effect, an invented number, a null
result described as a success.

Everything runs on the mock backend, so the suite needs no API key and no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.eda_agent import EDAAgent
from agents.llm import LLMError, ask_json, extract_json, get_backend
from agents.modelling_agent import ModellingAgent
from agents.multivariate_agent import MultivariateAgent
from verification import (as_floats, is_supported, normalise,
                         overstated_feature_benefit,
                         overstated_segmentation)
from agents.profiling_agent import ProfilingAgent
from agents.quality_agent import QualityAgent
from agents.report_agent import ReportAgent
from agents.validation_agent import ValidationAgent
from config import CONFIG
from orchestrator.graph import _repair_plan, run_workflow, should_run
from state import WorkflowState
from tools.modelling import evaluate_models
from tools.multivariate import (cluster_structure, feature_relevance,
                                multicollinearity, principal_components)
from tools.profiling import profile_dataset, suggest_targets
from tools.quality import apply_cleaning, assess_quality
from tools.visualization import build_chart


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def titanic() -> pd.DataFrame:
    """A small dataset with genuine missingness, mixed types and real signal."""
    rng = np.random.default_rng(42)
    n = 300
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    # Survival depends on sex and class, so the tests have a real effect to find.
    p_survive = np.where(sex == "female", 0.74, 0.19) * np.where(pclass == 1, 1.3, 1.0)
    df = pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived": (rng.random(n) < np.clip(p_survive, 0, 1)).astype(int),
        "Pclass": pclass,
        "Name": [f"Passenger {i}" for i in range(n)],
        "Sex": sex,
        "Age": rng.normal(29, 14, n).clip(1, 80).round(1),
        "Fare": rng.gamma(2, 20, n).round(2),
        "Embarked": rng.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]),
    })
    df.loc[rng.choice(n, 60, replace=False), "Age"] = np.nan       # 20% missing
    # Created as an object column: pandas 3 refuses to write text into the
    # float column an all-NaN assignment would otherwise produce.
    df["Cabin"] = pd.Series(np.nan, index=df.index, dtype=object)
    df.loc[rng.choice(n, 40, replace=False), "Cabin"] = "C85"      # 87% missing
    return df


@pytest.fixture
def noise() -> pd.DataFrame:
    """A dataset with no relationship between features and target."""
    rng = np.random.default_rng(0)
    n = 400
    return pd.DataFrame({
        "a": rng.normal(size=n), "b": rng.normal(size=n),
        "c": rng.choice(list("xyz"), n), "target": rng.integers(0, 2, n),
    })


@pytest.fixture
def mock():
    """The mock backend, restored to its defaults after each test."""
    backend = get_backend("mock")
    original = dict(backend._handlers)
    yield backend
    backend._handlers = original


def profiled(df: pd.DataFrame, **overrides) -> WorkflowState:
    """A state with profiling already run, for testing downstream agents."""
    state = WorkflowState(df=df.copy(), dataset_name="test.csv")
    state.record(ProfilingAgent().run(state))
    for key, value in overrides.items():
        setattr(state, key, value)
    return state


# --------------------------------------------------------------------------- #
# LLM layer
# --------------------------------------------------------------------------- #

class TestLLMLayer:
    def test_extracts_json_from_messy_responses(self):
        assert extract_json('{"a": 1}') == {"a": 1}
        assert extract_json('Here:\n```json\n{"a": 2}\n```\nDone.') == {"a": 2}
        # A closing brace inside a string value must not end the object early.
        assert extract_json('Sure! {"a": 3, "n": "has } brace"} ok')["a"] == 3

    def test_raises_when_no_json_present(self):
        with pytest.raises(LLMError):
            extract_json("there is no json here at all")

    def test_retries_with_the_specific_error(self, mock):
        attempts = {"n": 0}

        def flaky(_):
            attempts["n"] += 1
            return {"b": 1} if attempts["n"] == 1 else {"a": 1, "b": 1}

        mock.register("t", flaky)
        result = ask_json("t", "sys", "user", required_keys=("a", "b"))
        assert attempts["n"] == 2 and result["a"] == 1

    def test_gives_up_cleanly_when_output_stays_invalid(self, mock):
        mock.register("t", lambda _: {"wrong": "shape"})
        with pytest.raises(LLMError, match="failed after"):
            ask_json("t", "sys", "user", required_keys=("a",), retries=1)


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #

class TestTools:
    def test_profiling_separates_identifiers_from_features(self, titanic):
        profile = profile_dataset(titanic)
        types = {c["name"]: c["semantic_type"] for c in profile["columns"]}
        assert types["PassengerId"] == "identifier"
        assert types["Survived"] == "categorical_numeric"
        assert types["Age"] == "numeric"

    def test_suggest_targets_ranks_the_real_target_first(self, titanic):
        candidates = suggest_targets(profile_dataset(titanic))
        assert candidates and candidates[0]["column"] == "Survived"
        assert candidates[0]["task"] == "classification"

    def test_a_target_named_in_the_goal_ranks_first(self, titanic):
        """The person asking has said what the target is; a heuristic must not overrule them."""
        candidates = suggest_targets(profile_dataset(titanic), "Predict which port, Embarked")
        assert candidates[0]["column"] == "Embarked"
        assert candidates[0]["named_in_goal"] is True

    def test_one_hot_levels_and_constants_are_never_targets(self):
        """Covertype once modelled Wilderness_Area_0 -- fixed by its siblings -- at accuracy 1.00."""
        rng = np.random.default_rng(0)
        area = rng.integers(0, 4, 400)
        df = pd.DataFrame({f"Wilderness_Area_{i}": (area == i).astype(int) for i in range(4)})
        df["Elevation"] = rng.normal(2800, 200, 400)
        df["constant"] = 1
        df["Cover_Type"] = rng.integers(1, 8, 400)
        names = [c["column"] for c in suggest_targets(profile_dataset(df))]
        assert names[0] == "Cover_Type"
        assert not any(n.startswith("Wilderness_Area") or n == "constant" for n in names)

    def test_quality_scores_severity_by_missingness(self, titanic):
        assessment = assess_quality(titanic, profile_dataset(titanic))
        by_column = {(i["kind"], i["column"]): i for i in assessment["issues"]}
        assert by_column[("missing_values", "Cabin")]["severity"] == "critical"
        assert by_column[("missing_values", "Age")]["severity"] == "warning"

    def test_cleaning_logs_a_failure_rather_than_raising(self, titanic):
        _, log = apply_cleaning(titanic, [{"action": "impute_median", "column": "Ghost"}])
        assert log[0]["status"] == "failed"

    def test_chart_requests_are_validated_before_building(self, titanic):
        for spec, expected in [
            ({"kind": "piechart", "column": "Sex"}, "Unknown chart kind"),
            ({"kind": "histogram", "column": "Ghost"}, "not in the dataset"),
            ({"kind": "histogram", "column": "Sex"}, "requires a numeric column"),
            ({"kind": "box", "numeric": "Fare"}, "missing required argument"),
        ]:
            chart, error = build_chart(titanic, spec)
            assert chart is None and expected in error

    def test_modelling_finds_real_signal(self, titanic):
        result = evaluate_models(
            titanic, "Survived", ["Pclass", "Sex", "Age", "Fare"], "classification")
        assert result["status"] == "ok"
        assert result["beats_baseline"] is True

    def test_modelling_rejects_noise(self, noise):
        """The framework must not present chance performance as a finding."""
        result = evaluate_models(noise, "target", ["a", "b", "c"], "classification")
        assert result["status"] == "ok"
        assert result["beats_baseline"] is False

    def test_modelling_reports_why_it_cannot_run(self, titanic):
        assert "not in the dataset" in evaluate_models(
            titanic, "Ghost", ["Age"], "classification")["error"]
        assert "too few" in evaluate_models(
            titanic.head(10), "Survived", ["Age"], "classification")["error"]


# --------------------------------------------------------------------------- #
# Agent verification: the framework's central claim
# --------------------------------------------------------------------------- #

class TestProfilingAgentVerification:
    def test_corrects_a_hallucinated_target(self, titanic, mock):
        mock.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x",
            "usable_features": ["Sex", "InventedColumn"], "excluded_columns": [],
            "target_column": "DidSurvive", "task_type": "classification",
            "target_reasoning": "x", "concerns": [], "narrative": "x"})
        state = WorkflowState(df=titanic, dataset_name="t")
        output = ProfilingAgent().run(state)

        assert output.findings["target_column"] == "Survived"   # recovered
        assert "InventedColumn" not in output.findings["usable_features"]
        assert any("not present in the data" in w for w in output.warnings)

    def test_degrades_rather_than_fails_without_a_model(self, titanic, mock):
        def unavailable(_):
            raise LLMError("simulated outage")

        mock.register("profiling", unavailable)
        output = ProfilingAgent().run(WorkflowState(df=titanic, dataset_name="t"))
        assert output.status == "degraded"
        assert output.findings["target_column"] == "Survived"


class TestQualityAgentGuards:
    @pytest.mark.parametrize("action,column,expected", [
        ("drop_column", "Survived", "target column"),
        ("impute_median", "Cabin", "too much to impute"),
        ("impute_median", "Sex", "non-numeric"),
        ("delete_everything", None, "unsupported action"),
        ("drop_column", "Ghost", "no such column"),
    ])
    def test_refuses_destructive_or_invalid_actions(self, titanic, mock,
                                                    action, column, expected):
        mock.register("quality", lambda _: {
            "actions": [{"action": action, "column": column, "reason": "x"}],
            "declined": [], "limitations": [], "narrative": "x"})
        state = profiled(titanic)
        output = QualityAgent().run(state)

        assert any(expected in w for w in output.warnings)
        assert "Survived" in state.data.columns   # the target always survives

    def test_spends_nothing_on_a_clean_dataset(self, mock):
        clean = pd.DataFrame({"a": range(20), "b": list("xy") * 10,
                              "c": np.linspace(0, 1, 20)})
        state = profiled(clean)
        called = {"n": 0}

        def counting(prompt):
            called["n"] += 1
            return {"actions": [], "declined": [], "limitations": [], "narrative": ""}

        mock.register("quality", counting)
        output = QualityAgent().run(state)
        assert called["n"] == 0
        assert output.findings["quality_score"] == 100


class TestEDAAgentVerification:
    def test_flags_causal_language_and_overstatement(self, titanic, mock):
        mock.register("eda", lambda _: {
            "key_insights": [
                {"insight": "Being female causes survival.",
                 "evidence": "Cramer's V = 0.5", "importance": "high"},
                {"insight": "Fare is a critical determinant of survival.",
                 "evidence": "p = 0.04", "importance": "high"},
                {"insight": "Something matters.", "evidence": "", "importance": "medium"},
            ],
            "relationships": [], "distribution_notes": [], "data_observations": [],
            "modelling_implications": [], "narrative": "x"})
        output = EDAAgent().run(profiled(titanic))

        assert any("causal language" in w for w in output.warnings)
        assert any("cite no supporting statistic" in w for w in output.warnings)


class TestModellingAgentOverride:
    def test_overrides_a_signal_claim_contradicted_by_measurement(self, noise, mock):
        mock.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x",
            "usable_features": ["a", "b", "c"], "excluded_columns": [],
            "target_column": "target", "task_type": "classification",
            "target_reasoning": "x", "concerns": [], "narrative": "x"})
        mock.register("modelling", lambda _: {
            "recommended_model": "Decision Tree",
            "recommendation_reason": "Strong performance.",
            "expected_performance": "high", "evaluation_metrics": ["f1"],
            "important_features": ["a", "GhostFeature"],
            "signal_assessment": "moderate", "improvements": [], "caveats": [],
            "narrative": "x"})
        output = ModellingAgent().run(profiled(noise))

        assert output.findings["signal_assessment"] == "none"
        assert output.findings["recommended_model"] is None
        assert any("contradicts the measurements" in w for w in output.warnings)
        assert any("not present in the data" in w for w in output.warnings)


class TestSuspectedLeakage:
    """A near-perfect cross-validated score is flagged, not reported as skill."""

    def test_a_target_copied_into_the_inputs_is_flagged(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=400)
        df = pd.DataFrame({"x": x, "noise": rng.normal(size=400),
                           "label": np.where(x > 0, "yes", "no")})
        output = ModellingAgent().run(profiled(df))
        assert output.findings["suspected_leakage"] is True
        assert any("Near-perfect" in w for w in output.warnings)

    def test_an_ordinary_score_is_not_flagged(self, titanic):
        output = ModellingAgent().run(profiled(titanic))
        assert output.findings["suspected_leakage"] is False


class TestLiveRunFindings:
    """Two defects a live-provider run exposed, which the mock never produced."""

    def test_a_single_object_wrapped_in_a_list_is_accepted(self, mock):
        mock.register("multivariate", lambda _: [{"narrative": "ok"}])
        assert ask_json("multivariate", "s", "u", required_keys=("narrative",)) == {
            "narrative": "ok"}

    def test_a_fence_beyond_the_observed_range_clips_nothing_on_that_side(self):
        """An IQR fence of -6.69 for Age was reported, though no age is near it."""
        ages = pd.DataFrame({"Age": [0.42, 2, 20, 24, 28, 30, 35, 38, 45, 80]})
        out, log = apply_cleaning(ages, [{"action": "clip_outliers", "column": "Age"}])
        assert log[0]["lower"] is None and log[0]["upper"] is not None
        assert "below" not in log[0]["detail"]
        assert out["Age"].min() == 0.42 and out["Age"].max() < 80


class TestDistributedValidation:
    """On a Spark run the selected family is retrained on every row with MLlib."""

    def test_full_data_scores_are_produced_for_both_tasks(self, tmp_path):
        from tools import spark_engine

        ok, _ = spark_engine.spark_available()
        if not ok:
            pytest.skip("Spark is not available in this environment")
        rng = np.random.default_rng(3)
        x = rng.normal(size=2000)
        df = pd.DataFrame({"x": x, "g": rng.choice(["a", "b"], 2000),
                           "label": np.where(x + rng.normal(0, 0.3, 2000) > 0, "yes", "no"),
                           # A boolean target, as MiniBooNE's `signal` is.
                           "flag": x + rng.normal(0, 0.3, 2000) > 0,
                           "value": 3 * x + rng.normal(0, 0.5, 2000)})
        path = tmp_path / "d.csv"
        df.to_csv(path, index=False)
        try:
            sdf = spark_engine.load(str(path))
            cls = spark_engine.distributed_validation(sdf, "label", ["x", "g"],
                                                      "classification", "Random Forest")
            reg = spark_engine.distributed_validation(sdf, "value", ["x", "g"],
                                                      "regression", "Ridge Regression")
            flag = spark_engine.distributed_validation(sdf, "flag", ["x", "g"],
                                                       "classification", "LightGBM")
        finally:
            spark_engine.stop_session()
        assert cls["status"] == "ok" and cls["accuracy"] > 0.8
        assert cls["rows_train"] + cls["rows_test"] == 2000
        assert reg["status"] == "ok" and reg["r2"] > 0.9
        assert reg["mllib_model"] == "LinearRegression"
        assert flag["status"] == "ok" and flag["mllib_model"] == "GBTClassifier"


class TestReportAgentNumberVerification:
    def test_detects_invented_numbers(self, titanic, mock):
        mock.register("report", lambda _: {
            "title": "R",
            "executive_summary": "The model reached 94.3% accuracy in 3.7 seconds.",
            "sections": [{"heading": "S", "body": "Some 88.2% of cases improved."}],
            "key_takeaways": [], "limitations": [], "recommended_next_steps": []})
        output = ReportAgent().run(profiled(titanic))

        unsupported = output.findings["unsupported_numbers"]
        assert {"94.3", "88.2"} <= set(unsupported)

    def test_accepts_rounded_forms_of_computed_values(self, titanic, mock):
        state = profiled(titanic)
        state.record(QualityAgent().run(state))
        rows = state.data.shape[0]

        mock.register("report", lambda _: {
            "title": "R",
            "executive_summary": f"The dataset contains {rows} rows.",
            "sections": [], "key_takeaways": [], "limitations": [],
            "recommended_next_steps": []})
        output = ReportAgent().run(state)
        assert output.findings["unsupported_numbers"] == []

    def test_flags_a_hidden_null_result(self, noise, mock):
        mock.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x",
            "usable_features": ["a", "b", "c"], "excluded_columns": [],
            "target_column": "target", "task_type": "classification",
            "target_reasoning": "x", "concerns": [], "narrative": "x"})
        state = profiled(noise)
        state.record(ModellingAgent().run(state))

        mock.register("report", lambda _: {
            "title": "R",
            "executive_summary": "The analysis produced a strong predictive model.",
            "sections": [{"heading": "S", "body": "Performance was excellent."}],
            "key_takeaways": [], "limitations": [], "recommended_next_steps": []})
        output = ReportAgent().run(state)
        assert any("does not state this" in w for w in output.warnings)


class TestValidationAgent:
    def test_overrides_a_model_review_that_contradicts_a_measurement(self, noise, mock):
        mock.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x",
            "usable_features": ["a", "b", "c"], "excluded_columns": [],
            "target_column": "target", "task_type": "classification",
            "target_reasoning": "x", "concerns": [], "narrative": "x"})
        mock.register("report", lambda _: {
            "title": "R", "executive_summary": "Accuracy reached 99.7%.",
            "sections": [], "key_takeaways": [], "limitations": [],
            "recommended_next_steps": []})
        mock.register("validation", lambda _: {
            "issues": [], "verdict": "sound", "confidence_in_analysis": "high",
            "narrative": "Looks fine."})

        state = profiled(noise)
        for agent in (ModellingAgent, ReportAgent):
            state.record(agent().run(state))
        output = ValidationAgent().run(state)

        assert output.findings["verdict"] == "unreliable"
        assert output.findings["requires_revision"] is True
        assert any("overridden" in w for w in output.warnings)


class TestMultivariateTools:
    """PCA, clustering, VIF and mutual information - the unsupervised analyses."""

    @pytest.fixture
    def correlated(self) -> pd.DataFrame:
        """Three underlying factors expressed through nine observed columns."""
        rng = np.random.default_rng(5)
        n = 300
        a, b, c = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        data = {}
        for i, factor in enumerate((a, b, c)):
            for j in range(3):
                data[f"f{i}_{j}"] = factor + rng.normal(0, 0.25, n)
        data["target"] = (a * 1.5 + rng.normal(0, 0.5, n) > 0).astype(int)
        return pd.DataFrame(data)

    def test_pca_detects_redundancy(self, correlated):
        """Nine columns built from three factors should compress to about three."""
        result = principal_components(correlated, exclude=("target",))
        assert result["n_features"] == 9
        assert result["components_for_target"] <= 4
        assert result["redundancy"] > 0.5

    def test_clustering_reports_weak_structure_honestly(self):
        """Unstructured data must not be presented as having natural groups."""
        rng = np.random.default_rng(3)
        blob = pd.DataFrame(rng.normal(size=(300, 4)),
                            columns=[f"v{i}" for i in range(4)])
        result = cluster_structure(blob)
        assert result is not None
        assert result["meaningful"] is False
        assert result["structure_strength"] == "weak"

    def test_clustering_finds_real_separation(self):
        """Well-separated groups should be reported as meaningful."""
        rng = np.random.default_rng(3)
        separated = pd.DataFrame(
            np.vstack([rng.normal(-6, 0.5, (150, 4)), rng.normal(6, 0.5, (150, 4))]),
            columns=[f"v{i}" for i in range(4)])
        result = cluster_structure(separated)
        assert result["best_k"] == 2
        assert result["meaningful"] is True
        assert result["silhouette"] > 0.5

    def test_vif_flags_a_derived_column(self):
        """A column that is a sum of others is perfectly predictable from them."""
        rng = np.random.default_rng(9)
        n = 200
        df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        df["c"] = df["a"] + df["b"]          # exactly collinear
        df["d"] = rng.normal(size=n)         # independent
        result = multicollinearity(df)
        assert "d" not in result["severe_multicollinearity"]
        assert len(result["severe_multicollinearity"]) >= 2

    def test_mutual_information_finds_a_non_linear_relationship(self):
        """Correlation reports nothing for a U-shape; mutual information does not."""
        rng = np.random.default_rng(11)
        n = 500
        x = rng.uniform(-3, 3, n)
        df = pd.DataFrame({"x": x, "noise": rng.normal(size=n),
                           "y": x ** 2 + rng.normal(0, 0.5, n)})
        assert abs(df["x"].corr(df["y"])) < 0.2        # linear correlation misses it
        result = feature_relevance(df, "y", "regression")
        ranked = {r["feature"]: r["mutual_information"] for r in result["ranked"]}
        assert ranked["x"] > ranked["noise"]
        assert "x" in result["informative_features"]


class TestMultivariateAgent:
    @pytest.mark.parametrize("text,flagged", [
        ("The data does not separate into natural groups.", False),
        ("There are no distinct groups here.", False),
        ("The silhouette was only 0.2, so no clear clusters exist.", False),
        ("The data separates into three natural groups.", True),
        ("Customers fall into well-separated segments.", True),
    ])
    def test_negation_aware_segmentation_check(self, text, flagged):
        """Reporting an absence of clusters must not be flagged as claiming them."""
        assert bool(overstated_segmentation(text)) is flagged

    def test_flags_weak_clustering_sold_as_segmentation(self, mock):
        rng = np.random.default_rng(3)
        blob = pd.DataFrame(rng.normal(size=(300, 5)),
                            columns=[f"v{i}" for i in range(5)])
        blob["target"] = rng.integers(0, 2, 300)

        mock.register("multivariate", lambda _: {
            "structure_summary": "The data separates into two distinct groups.",
            "redundancy_findings": [], "cluster_findings": ["Two clear clusters emerge."],
            "interpretation_warnings": [], "non_linear_signals": [],
            "modelling_implications": [], "narrative": "Well-separated segments were found."})
        output = MultivariateAgent().run(profiled(blob))

        assert output.findings["clustering_meaningful"] is False
        assert any("segmentation" in w for w in output.warnings)

    def test_skips_when_too_few_numeric_columns(self, mock):
        df = pd.DataFrame({"a": range(50), "b": ["x", "y"] * 25})
        output = MultivariateAgent().run(profiled(df))
        assert output.findings.get("skipped") is True


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

class TestOrchestration:
    def test_repairs_an_impossible_plan(self):
        plan, notes = _repair_plan(["eda", "nonsense_stage", "profiling"])
        assert plan[0] == "profiling"                      # ordered correctly
        assert {"report", "validation"} <= set(plan)       # mandatory restored
        assert any("unknown stage" in n for n in notes)

    def test_drops_a_stage_whose_dependency_was_omitted(self):
        # Every stage depends on profiling, which _repair_plan restores; a stage
        # is only dropped when a dependency cannot be satisfied at all.
        plan, notes = _repair_plan(["quality", "eda"])
        assert "profiling" in plan

    def test_skips_modelling_without_a_target(self, titanic):
        state = profiled(titanic, target_column=None, task_type="none")
        run, reason = should_run("modelling", state)
        assert run is False and "no supervised target" in reason.lower()

    def test_skips_modelling_on_too_few_rows(self, titanic):
        state = profiled(titanic.head(20), target_column="Survived",
                         task_type="classification")
        run, reason = should_run("modelling", state)
        assert run is False and "too few" in reason.lower()

    def test_runs_end_to_end_and_produces_a_report(self, titanic):
        state = run_workflow(titanic, "titanic.csv")
        assert "profiling" in state.completed
        assert state.has_run("report") and state.has_run("validation")
        assert state.findings("report").get("title")
        assert state.findings("validation").get("verdict") in (
            "sound", "needs_revision", "unreliable")

    def test_adapts_the_plan_to_a_clean_unsupervised_dataset(self, mock):
        clean = pd.DataFrame({
            "sensor_a": np.linspace(0, 100, 200),
            "sensor_b": np.linspace(50, 0, 200),
            "station": ["north", "south"] * 100,
        })
        state = run_workflow(clean, "sensors.csv")
        # With no missing values or duplicates the cleaning stage is unnecessary.
        assert "quality" not in state.plan or not state.has_run("quality")

    def test_validation_can_send_the_report_back(self, titanic, mock):
        attempts = {"n": 0}

        def report_handler(_):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return {"title": "R",
                        "executive_summary": "Accuracy was 97.3% with 88.1% recall.",
                        "sections": [], "key_takeaways": [], "limitations": [],
                        "recommended_next_steps": []}
            return {"title": "Revised", "executive_summary": "A revised summary.",
                    "sections": [], "key_takeaways": [],
                    "limitations": ["Stated limitation."], "recommended_next_steps": []}

        mock.register("report", report_handler)
        mock.register("validation", lambda _: {
            "issues": [], "verdict": "sound", "confidence_in_analysis": "high",
            "narrative": "x"})

        state = run_workflow(titanic, "titanic.csv")
        assert attempts["n"] > 1                     # a revision was requested
        assert state.revisions >= 1

    def test_revision_is_bounded(self, titanic, mock):
        """A report that never improves must not loop forever."""
        mock.register("report", lambda _: {
            "title": "R", "executive_summary": "Accuracy was 97.3%.",
            "sections": [], "key_takeaways": [], "limitations": [],
            "recommended_next_steps": []})
        state = run_workflow(titanic, "titanic.csv")
        assert state.revisions <= CONFIG.max_validation_retries


class TestWorkflowState:
    def test_cleaning_replaces_the_working_frame_only(self, titanic, mock):
        mock.register("quality", lambda _: {
            "actions": [{"action": "drop_column", "column": "Cabin", "reason": "x"}],
            "declined": [], "limitations": [], "narrative": "x"})
        state = profiled(titanic)
        state.record(QualityAgent().run(state))

        assert "Cabin" not in state.data.columns      # agents see the cleaned frame
        assert "Cabin" in state.df.columns            # the original is preserved

    def test_findings_of_a_skipped_stage_are_empty_not_an_error(self, titanic):
        state = profiled(titanic)
        assert state.findings("modelling") == {}
        assert state.has_run("modelling") is False


class TestExecutionEngine:
    """The distributed backend and the rule that decides when to use it.

    The correctness claim these tests defend is that the engine is invisible
    above the tool layer: an agent given a Spark-computed profile and one given
    a pandas-computed profile must be unable to tell the difference. If the two
    engines disagreed, switching between them automatically would silently
    change results.
    """

    def test_small_data_stays_in_memory(self, tmp_path):
        """Spark must not be chosen for data that fits comfortably in memory."""
        from tools.spark_engine import choose_engine

        path = tmp_path / "small.csv"
        path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        decision = choose_engine(str(path))
        assert decision["engine"] == "pandas"
        assert "threshold" in decision["reason"]

    def test_large_data_selects_spark_when_available(self, tmp_path, monkeypatch):
        """Past the threshold the distributed engine is chosen, and says why."""
        from tools import spark_engine

        path = tmp_path / "big.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        monkeypatch.setattr(spark_engine, "spark_available", lambda: (True, "available"))
        monkeypatch.setattr(CONFIG, "spark_size_threshold_mb", 0.0)
        decision = spark_engine.choose_engine(str(path))
        assert decision["engine"] == "spark"

    def test_missing_spark_degrades_to_pandas(self, tmp_path, monkeypatch):
        """A large dataset with no Spark must still run, not fail.

        Degrading here matters more than elsewhere: the alternative is refusing
        to analyse a dataset that pandas may well be able to handle.
        """
        from tools import spark_engine

        path = tmp_path / "big.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        monkeypatch.setattr(spark_engine, "spark_available",
                            lambda: (False, "pyspark is not installed"))
        monkeypatch.setattr(CONFIG, "spark_size_threshold_mb", 0.0)
        decision = spark_engine.choose_engine(str(path))
        assert decision["engine"] == "pandas"
        assert "unavailable" in decision["reason"]

    def test_the_two_engines_produce_the_same_profile(self, tmp_path):
        """The central claim: identical results from either execution path."""
        from tools import spark_engine
        from tools.profiling import profile_dataset

        ok, _ = spark_engine.spark_available()
        if not ok:
            pytest.skip("Spark is not available in this environment")

        df = pd.DataFrame({
            "n": list(range(200)),
            "value": [float(i % 17) for i in range(200)],
            "group": ["a", "b", "c", "d"] * 50,
        })
        path = tmp_path / "compare.csv"
        df.to_csv(path, index=False)

        expected = profile_dataset(pd.read_csv(path))
        actual = spark_engine.profile_dataset(spark_engine.load(str(path)))

        assert actual["shape"] == expected["shape"]
        assert actual["duplicate_rows"] == expected["duplicate_rows"]
        assert actual["total_missing_cells"] == expected["total_missing_cells"]

        by_name = {c["name"]: c for c in actual["columns"]}
        for column in expected["columns"]:
            other = by_name[column["name"]]
            assert other["semantic_type"] == column["semantic_type"]
            assert other["unique"] == column["unique"]
            if "mean" in column.get("stats", {}):
                assert other["stats"]["mean"] == pytest.approx(
                    column["stats"]["mean"], rel=1e-6)

    def test_a_distributed_run_reports_the_full_row_count(self, titanic):
        """The report must describe the dataset, not the sample it held."""
        state = WorkflowState(df=titanic.head(50), dataset_name="t.csv")
        state.engine = "spark"
        state.full_row_count = 3_000_000
        assert state.summary()["rows"] == 3_000_000
        assert state.summary()["in_memory_rows"] == 50


@pytest.mark.slow
class TestHyperparameterTuning:
    """Tuning must be measured, not assumed.

    The failure this guards against is adopting tuned parameters because tuning
    is conventionally beneficial. On small datasets a randomised search overfits
    its own inner folds often enough that unconditional adoption makes the
    reported score worse.
    """

    def test_the_gain_is_measured_on_unseen_rows(self, titanic):
        """Tuned and untuned are compared on rows neither was fitted on."""
        from tools.modelling import tune_model

        features = [c for c in titanic.columns if c != "Survived"]
        result = tune_model(titanic, "Survived", features, "classification",
                            "Random Forest")
        assert result["status"] == "ok"
        assert result["holdout_rows"] > 0
        assert result["gain"] == pytest.approx(
            result["tuned_holdout_score"] - result["untuned_holdout_score"], abs=1e-4)

    def test_a_negligible_gain_is_not_called_an_improvement(self, titanic):
        """A difference inside split noise must not be reported as a win."""
        from tools.modelling import tune_model

        features = [c for c in titanic.columns if c != "Survived"]
        result = tune_model(titanic, "Survived", features, "classification",
                            "Random Forest")
        if abs(result["gain"]) <= 0.002:
            assert result["improved"] is False

    def test_tuning_is_skipped_on_data_too_large_to_search(self, monkeypatch, titanic):
        """The search is declined with a reason rather than run for hours."""
        from tools.modelling import tune_model

        monkeypatch.setattr(CONFIG, "tuning_max_rows", 10)
        features = [c for c in titanic.columns if c != "Survived"]
        result = tune_model(titanic, "Survived", features, "classification",
                            "Random Forest")
        assert result["status"] == "skipped"
        assert "exceeds the tuning limit" in result["reason"]

    def test_untuned_families_are_left_alone(self, titanic):
        """A model with no search range defined is not tuned by accident."""
        from tools.modelling import tune_model

        features = [c for c in titanic.columns if c != "Survived"]
        assert tune_model(titanic, "Survived", features, "classification",
                          "Majority baseline") is None

    def test_a_tuned_winner_reports_its_parameters(self, titanic):
        """Whatever is adopted must be inspectable, not an unexplained score."""
        from tools.modelling import evaluate_models

        features = [c for c in titanic.columns if c != "Survived"]
        result = evaluate_models(titanic, "Survived", features, "classification")
        if result["best_model_tuned"]:
            assert result["best_model_params"]
            assert result["tuning"]["adopted"] is True
        else:
            assert result["best_model_params"] is None


class TestProviderBackends:
    """Every provider must be reachable through one interface.

    The framework's claim is that the reasoning engine is swappable. That is
    only true if selecting a provider needs no change anywhere but a flag, and
    if a missing key fails with an instruction rather than a stack trace.
    """

    def test_every_provider_is_selectable_by_name(self, monkeypatch):
        from agents import llm

        llm.reset_backends()
        # The Anthropic backend reads its key from the config object, which is
        # populated from the environment once at import; the others read the
        # environment when constructed. Both are set so the test does not depend
        # on which mechanism a provider happens to use.
        monkeypatch.setattr(CONFIG.llm, "api_key", "test-key-not-used")
        for name, key in (("anthropic", "ANTHROPIC_API_KEY"),
                          ("gemini", "GEMINI_API_KEY"),
                          ("grok", "GROK_API_KEY")):
            monkeypatch.setenv(key, "test-key-not-used")
            backend = llm.get_backend(name)
            assert backend.name == name
            assert hasattr(backend, "complete")
        llm.reset_backends()

    def test_a_missing_key_explains_how_to_supply_one(self, monkeypatch):
        from agents import llm

        llm.reset_backends()
        for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        with pytest.raises(llm.LLMError, match="GEMINI_API_KEY is not set"):
            llm.get_backend("gemini")
        llm.reset_backends()

    def test_an_unknown_backend_lists_the_valid_ones(self):
        from agents import llm

        with pytest.raises(llm.LLMError, match="gemini"):
            llm.get_backend("not-a-provider")


class TestProfilingAccountsForEveryColumn:
    """A column profiling rejected must be recorded as rejected.

    The heuristic fallback, used whenever the model is unreachable, dropped
    identifier and free-text columns from `usable_features` but reported
    `excluded_columns: []`. Downstream that reads as "profiling excluded
    nothing", so the modelling agent classified every dropped column as a
    feature derived after profiling and reinstated it -- fitting a row id as a
    predictor, on precisely the runs where no narrative was being written to
    make it visible.
    """

    def test_the_fallback_records_what_it_rejected(self, monkeypatch):
        import agents.profiling_agent as profiling_agent
        from agents.llm import LLMError

        def unreachable(*args, **kwargs):
            raise LLMError("simulated provider outage")

        monkeypatch.setattr(profiling_agent, "ask_json", unreachable)

        df = pd.DataFrame({
            "row_id": range(40),
            "note": [f"free text {i}" for i in range(40)],
            "score": [i % 7 for i in range(40)],
            "outcome": [0, 1] * 20,
        })
        output = profiling_agent.ProfilingAgent().run(
            WorkflowState(df=df, dataset_name="t.csv"))

        assert output.status == "degraded"
        excluded = {c["column"] for c in output.findings["excluded_columns"]}
        assert "row_id" in excluded
        assert "note" in excluded

    def test_no_column_is_silently_unaccounted_for(self, monkeypatch):
        """The invariant the modelling agent depends on.

        It reinstates any column absent from both lists, on the reasoning that
        such a column did not exist when profiling ran and must therefore have
        been derived since. That inference is only sound if profiling lists
        every column it saw.
        """
        import agents.profiling_agent as profiling_agent
        from agents.llm import LLMError

        monkeypatch.setattr(
            profiling_agent, "ask_json",
            lambda *a, **k: (_ for _ in ()).throw(LLMError("outage")))

        df = pd.DataFrame({
            "row_id": range(40),
            "note": [f"text {i}" for i in range(40)],
            "score": [i % 7 for i in range(40)],
            "outcome": [0, 1] * 20,
        })
        findings = profiling_agent.ProfilingAgent().run(
            WorkflowState(df=df, dataset_name="t.csv")).findings

        accounted = (set(findings["usable_features"])
                     | {c["column"] for c in findings["excluded_columns"]}
                     | {findings.get("target_column")})

        assert set(df.columns) <= accounted


class TestPercentageEquivalence:
    """A fraction and the same value as a percentage are the same number.

    The reverse direction of this check divided where it needed to multiply, so
    it could never match: a report correctly writing 0.15 for a computed 15.0
    was recorded as citing a figure no stage produced. The validator promotes
    that to a critical issue, which forces a revision round that cannot succeed
    because nothing is wrong. A checker that flags correct numbers is worse than
    no checker, for the reason `normalise` gives -- it teaches the reader to
    ignore the warnings.
    """

    def test_a_computed_fraction_may_be_written_as_a_percentage(self):
        from verification import is_supported

        assert is_supported(15.0, 0.05, [0.15])

    def test_a_computed_percentage_may_be_written_as_a_fraction(self):
        from verification import is_supported

        assert is_supported(0.15, 0.005, [15.0])

    def test_a_fabricated_number_is_still_rejected(self):
        """The conversions must not become a way for anything to pass."""
        from verification import is_supported

        assert not is_supported(0.9999, 0.00005, [0.8311])
        assert not is_supported(42.0, 0.05, [0.8311])


class TestTargetIsProtectedFromCleaning:
    """No cleaning action may modify the column being predicted.

    The guard was an allow-list of three actions, which left `clip_outliers`
    free to compress a regression target's tails -- changing the values every
    score in the report is measured against, and flattering the model by
    removing the hardest cases. An allow-list must be extended whenever an
    action is added, and will not be; naming the one column that must not
    change cannot be forgotten.
    """

    @pytest.mark.parametrize("action", ["clip_outliers", "to_numeric",
                                        "normalise_categories", "drop_column",
                                        "impute_median", "impute_mode"])
    def test_every_action_is_refused_on_the_target(self, action, titanic):
        from agents.profiling_agent import ProfilingAgent
        from agents.quality_agent import QualityAgent

        state = WorkflowState(df=titanic, dataset_name="t.csv")
        state.outputs["profiling"] = ProfilingAgent().run(state)
        state.target_column, state.task_type = "Survived", "classification"

        approved, rejected = QualityAgent()._screen(
            [{"action": action, "column": "Survived", "reason": "r"}],
            titanic, state)

        assert approved == []
        assert any("target column" in r for r in rejected)

    def test_the_same_action_is_allowed_on_a_feature(self, titanic):
        """The guard must protect the target without disabling cleaning."""
        from agents.profiling_agent import ProfilingAgent
        from agents.quality_agent import QualityAgent

        state = WorkflowState(df=titanic, dataset_name="t.csv")
        state.outputs["profiling"] = ProfilingAgent().run(state)
        state.target_column, state.task_type = "Survived", "classification"

        approved, _ = QualityAgent()._screen(
            [{"action": "clip_outliers", "column": "Fare", "reason": "r"}],
            titanic, state)

        assert [a["column"] for a in approved] == ["Fare"]


class TestChartGuidanceCoverage:
    """Every chart the framework draws must have a note on what to look for.

    The guidance is keyed by chart kind, and two keys named kinds that do not
    exist -- "box_plot" where the builder is "box", and "feature_importance",
    which this framework does not draw. A lookup miss returns the empty string,
    so box plots silently lost their note and nothing reported it. Comparing the
    two sets is the only way this stays true as chart kinds are added.
    """

    def test_every_chart_kind_has_guidance(self):
        from orchestrator.plain_language import chart_guidance
        from tools.visualization import BUILDERS

        missing = [kind for kind in BUILDERS if not chart_guidance(kind, {})]

        assert not missing, f"chart kinds with no guidance: {missing}"

    def test_guidance_says_what_to_look_for_not_what_it_is(self):
        """The note must add something the reader could not already see.

        A caption naming the chart tells them what is in front of them; the
        useful part is which feature of it carries the finding.
        """
        from orchestrator.plain_language import chart_guidance
        from tools.visualization import BUILDERS

        for kind in BUILDERS:
            note = chart_guidance(kind, {}).lower()
            assert "look" in note or "compare" in note or "note where" in note


class TestInterfaceRenders:
    """The Streamlit app must actually render, not merely import.

    Streamlit enforces rules that no linter or import check sees -- expanders
    may not nest, for one -- and they raise at render time. A nested expander
    shipped in the agent-detail tab and was found by a person opening the page,
    which is the wrong place to find it.

    `AppTest` runs the script the way Streamlit does and collects any exception,
    so these constraints are checked with everything else.
    """

    def test_the_app_renders_without_a_dataset(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                default_timeout=60)
        app.run()

        assert not app.exception, [e.value for e in app.exception]

    def test_a_failed_permutation_test_is_reported_not_hidden(self, titanic, monkeypatch):
        import tools.modelling as modelling

        def boom(*args, **kwargs):
            raise MemoryError("simulated")

        monkeypatch.setattr(CONFIG, "stacking_max_rows", 0)
        monkeypatch.setattr(CONFIG, "tune_models", False)
        monkeypatch.setattr(modelling, "permutation_test_score", boom)
        df = titanic[["Survived", "Pclass", "Sex", "Fare"]].dropna()
        result = modelling.evaluate_models(df, "Survived", ["Pclass", "Sex", "Fare"],
                                           "classification")
        assert result["permutation_test"] is None
        assert result["permutation_test_error"].startswith("MemoryError")

    def test_every_result_tab_renders(self, titanic, monkeypatch, tmp_path):
        """The tabs only exist once a run has produced state.

        The failure that prompted this test was in the agent-detail tab, which
        is not reached until an analysis has completed -- so rendering the empty
        page would not have caught it.
        """
        from streamlit.testing.v1 import AppTest

        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        monkeypatch.setattr(CONFIG, "stacking_max_rows", 0)
        monkeypatch.setattr(CONFIG, "tune_models", False)
        monkeypatch.setattr(CONFIG, "permutation_tests", 0)

        state = run_workflow(titanic.head(120), "titanic.csv")

        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"),
                                default_timeout=120)
        app.session_state["state"] = state
        app.run()

        assert not app.exception, [e.value for e in app.exception]


class TestTrimKeepsListsHomogeneous:
    """A truncated list must stay the shape its consumers expect.

    `_trim` appended a bare string marker to every shortened list, including
    lists of objects. Every consumer that iterates such a list and reads a
    field then crashed on the final element -- and because trimming only
    happens on payloads large enough to need it, the report stage failed on
    exactly the bigger datasets and nowhere else, which read as an
    intermittent fault rather than a type error.
    """

    def test_a_truncated_list_of_objects_stays_objects(self):
        from agents.llm import _trim

        trimmed = _trim({"k": [{"insight": "a", "importance": "high"}] * 10},
                        2000, 3)["k"]

        assert all(isinstance(item, dict) for item in trimmed)
        # The omission is still reported, just in the list's own shape.
        assert any("_omitted" in item for item in trimmed)

    def test_a_truncated_list_of_strings_stays_strings(self):
        from agents.llm import _trim

        trimmed = _trim({"k": ["a"] * 10}, 2000, 3)["k"]

        assert all(isinstance(item, str) for item in trimmed)
        assert any("more omitted" in item for item in trimmed)

    def test_the_report_survives_a_trimmed_payload(self, monkeypatch):
        """The failure this was found through, guarded end to end."""
        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        # A budget small enough to force trimming on any real dataset.
        monkeypatch.setattr(CONFIG, "report_prompt_char_budget", 1500)

        rng = np.random.default_rng(0)
        df = pd.DataFrame({f"c{i}": rng.normal(size=200) for i in range(8)})
        df["y"] = (df["c0"] + df["c1"] > 0).astype(int)

        state = run_workflow(df, "trimmed.csv")

        assert state.outputs["report"].status != "failed"


class TestImbalanceHandling:
    """A skew the EDA stage reports must be one the modelling stage acts on.

    `class_weight="balanced"` was searched only during tuning, which is
    optional and is skipped above `tuning_max_rows`. On a large imbalanced
    dataset nothing compensated at all: the framework measured the imbalance,
    said so in the report, and then fitted with default weights.
    """

    @staticmethod
    def _imbalanced(ratio_target=0.09, n=800):
        rng = np.random.default_rng(0)
        y = (rng.random(n) < ratio_target).astype(int)
        return pd.DataFrame({
            "x1": rng.normal(size=n) + y * 1.1,
            "x2": rng.normal(size=n) + y * 0.6,
            "y": y,
        })

    def test_weighted_variants_are_offered_on_a_skewed_target(self, monkeypatch):
        from tools.modelling import evaluate_models

        monkeypatch.setattr(CONFIG, "tune_models", False)
        monkeypatch.setattr(CONFIG, "stacking_max_rows", 0)

        result = evaluate_models(self._imbalanced(), "y", ["x1", "x2"],
                                 "classification")
        names = [m["model"] for m in result["results"]]

        assert any("balanced" in n for n in names)

    def test_they_are_added_not_substituted(self, monkeypatch):
        """Weighting is a trade, not an improvement.

        Measured on a 10:1 target it moved macro F1 down and recall sharply up:
        the model finds far more of the minority class and is wrong more often
        when it calls one. Which a reader wants depends on what a miss costs
        against what a false alarm costs, which is not in the data -- so both
        are fitted and both appear, and the choice stays with the reader.
        """
        from tools.modelling import evaluate_models

        monkeypatch.setattr(CONFIG, "tune_models", False)
        monkeypatch.setattr(CONFIG, "stacking_max_rows", 0)

        result = evaluate_models(self._imbalanced(), "y", ["x1", "x2"],
                                 "classification")
        names = [m["model"] for m in result["results"]]

        assert "Logistic Regression" in names
        assert "Logistic Regression (balanced)" in names

    def test_a_balanced_target_is_not_charged_for_them(self, monkeypatch):
        """Extra fits on data that does not need them are wasted time."""
        from tools.modelling import evaluate_models

        monkeypatch.setattr(CONFIG, "tune_models", False)
        monkeypatch.setattr(CONFIG, "stacking_max_rows", 0)
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        df = pd.DataFrame({"x1": rng.normal(size=400) + y, "y": y})

        result = evaluate_models(df, "y", ["x1"], "classification")
        names = [m["model"] for m in result["results"]]

        assert not any("balanced" in n for n in names)

    def test_the_threshold_is_shared_with_the_eda_stage(self):
        """One number, so the report cannot describe a skew nothing acted on."""
        from tools.eda import target_balance

        df = pd.DataFrame({"y": [0] * 90 + [1] * 10})
        balance = target_balance(df, "y")

        assert balance["imbalance_ratio"] > CONFIG.imbalance_threshold
        assert balance["imbalanced"] is True


class TestModelPersistence:
    """A recommended model must be one the reader can actually obtain.

    Every pipeline fitted during evaluation is fitted on a fold and discarded,
    so a report saying "a Random Forest predicts this reliably" was pointing at
    an object that no longer existed. The winner is refitted on all rows and
    saved with a signature naming the columns it expects.
    """

    def test_the_winner_is_refitted_on_every_row(self, titanic):
        from tools.modelling import fit_final_model

        features = ["Pclass", "Age", "Fare", "Sex"]
        result = fit_final_model(titanic, "Survived", features,
                                 "classification", "Logistic Regression")

        assert result is not None
        pipeline, signature = result
        assert signature["n_training_rows"] == len(titanic)
        assert signature["expects_columns"] == features
        assert signature["target"] == "Survived"

    def test_the_saved_pipeline_scores_raw_rows(self, titanic):
        """Preprocessing is inside the pipeline.

        A caller must not have to reproduce imputation and encoding to use the
        model; the whole point of saving the pipeline rather than the estimator
        is that raw rows go in.
        """
        from tools.modelling import fit_final_model

        features = ["Pclass", "Age", "Fare", "Sex"]
        pipeline, _ = fit_final_model(titanic, "Survived", features,
                                      "classification", "Logistic Regression")

        # Raw rows, including the missing ages the original data contains.
        predictions = pipeline.predict(titanic[features].head(10))

        assert len(predictions) == 10

    def test_a_baseline_winner_is_not_saved(self, monkeypatch):
        """A baseline winning means nothing learned anything.

        That is a finding, not a recommendation, and saving it would invite
        someone to use a model the report is telling them has no signal.
        """
        import pandas as pd_

        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        rng = np.random.default_rng(0)
        noise = pd_.DataFrame(rng.normal(size=(120, 4)),
                              columns=[f"c{i}" for i in range(4)])
        noise["y"] = rng.integers(0, 2, 120)

        state = run_workflow(noise, "noise.csv")
        modelling = state.findings("modelling")

        if modelling.get("best_is_baseline"):
            assert modelling.get("saved_model") is None

    def test_the_signature_records_what_the_model_expects(self, titanic):
        """A pipeline without a signature is a black box.

        It fails confusingly on the first column it does not recognise, so the
        contract is written down beside it.
        """
        from tools.modelling import fit_final_model

        _, signature = fit_final_model(
            titanic, "Survived", ["Pclass", "Age", "Fare"],
            "classification", "Logistic Regression")

        for key in ("target", "task", "model", "expects_columns",
                    "column_dtypes", "n_training_rows", "sklearn_version"):
            assert key in signature
        assert signature["classes"] is not None  # a classifier lists its labels


class TestMalformedInput:
    """Real files are malformed in ordinary ways, and must degrade not crash.

    These were found by probing rather than by a test. The duplicate-header
    case raised `TypeError: cannot convert the series to <class 'int'>` out of
    `run_workflow` entirely -- a traceback instead of a report, after the file
    had already been read. Duplicate headers are common in spreadsheet exports.
    """

    @pytest.mark.parametrize("name,frame", [
        ("empty", pd.DataFrame()),
        ("single row", pd.DataFrame({"a": [1], "b": [2]})),
        ("all null column", pd.DataFrame({"a": [None] * 30,
                                          "b": list(range(30))})),
        ("unicode headers", pd.DataFrame({"名前": range(30),
                                          "ü": [0, 1] * 15})),
    ])
    def test_the_workflow_completes(self, name, frame, monkeypatch):
        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")

        state = run_workflow(frame, "edge.csv")

        assert state is not None

    def test_duplicate_column_names_do_not_crash(self, monkeypatch):
        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        frame = pd.DataFrame(np.arange(90).reshape(30, 3),
                             columns=["x", "x", "y"])

        state = run_workflow(frame, "dup.csv")

        # Renamed rather than dropped: a duplicated header usually means two
        # real columns written under one name, and discarding one loses data.
        assert list(state.data.columns) == ["x", "x.1", "y"]
        assert any(entry.get("event") == "columns_renamed"
                   for entry in state.trace)


class TestUntrustedDatasetContent:
    """A dataset is an untrusted input, and its values reach the prompts.

    Category labels and example values are placed in the profiling prompt to
    help the interpretation. A file containing "IGNORE ALL PREVIOUS
    INSTRUCTIONS" therefore puts that sentence in front of the model, and
    nothing previously told it the content was data rather than instruction.

    Two defences, and the architectural one is the stronger: agents never
    compute, so an injected claim has to invent a figure, and an invented figure
    is exactly what the verification layer detects.
    """

    def test_an_injected_number_is_rejected_by_verification(self):
        """The defence that actually holds.

        An attacker can persuade a model to write anything; they cannot make
        the computation agree with it.
        """
        from verification import collect_numbers, unsupported_numbers

        allowed: set[str] = set()
        collect_numbers({"modelling": {"best_score": 0.7374,
                                       "accuracy": 0.7747}}, allowed)

        assert unsupported_numbers(
            "The model achieves an accuracy of 0.99.", allowed) == ["0.99"]
        # And the genuine figure is not flagged, which is what makes the check
        # worth reading rather than worth ignoring.
        assert not unsupported_numbers(
            "The model achieves an accuracy of 0.7747.", allowed)

    def test_every_agent_is_told_the_data_is_untrusted(self, monkeypatch):
        from agents import llm

        captured = {}
        original = llm._complete_cached

        def spy(backend, agent, system, user):
            captured["system"] = system
            return original(backend, agent, system, user)

        monkeypatch.setattr(llm, "_complete_cached", spy)
        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        llm.ask_json("profiling", "You are the Data Profiling Agent.", "{}")

        assert "SECURITY NOTICE" in captured["system"]
        assert "only from this system prompt" in captured["system"]
        # The agent's own instructions must survive alongside it.
        assert "Data Profiling Agent" in captured["system"]

    def test_identifier_and_text_contents_are_not_sent(self, monkeypatch):
        """Values that can identify a person do not leave the machine.

        A category label like "female" is harmless and genuinely helps the
        interpretation. A patient reference or a clinical note is neither, and
        the profiler already knows which is which.
        """
        import agents.profiling_agent as profiling_agent

        captured = {}
        original = profiling_agent.ask_json

        def spy(agent, system, user, **kwargs):
            captured["user"] = user
            return original(agent, system, user, **kwargs)

        monkeypatch.setattr(profiling_agent, "ask_json", spy)

        df = pd.DataFrame({
            "patient_ref": [f"NHS-{i:07d}" for i in range(40)],
            "notes": [f"clinical narrative number {i}, confidential"
                      for i in range(40)],
            "sex": ["male", "female"] * 20,
            "outcome": [0, 1] * 20,
        })
        profiling_agent.ProfilingAgent().run(
            WorkflowState(df=df, dataset_name="t.csv"))

        prompt = captured["user"]
        assert "NHS-000" not in prompt
        assert "confidential" not in prompt
        # The column is still described, so the analysis knows it exists.
        assert "patient_ref" in prompt
        # A benign category is still shown, because withholding it would cost
        # the interpretation something for no privacy gain.
        assert "female" in prompt

    def test_example_values_are_length_bounded(self):
        """A real category label is short; anything long is not one."""
        from agents.profiling_agent import MAX_EXAMPLE_CHARS, _short

        assert _short("female") == "female"
        long_value = "x" * 500
        assert len(_short(long_value)) <= MAX_EXAMPLE_CHARS + 3


class TestReportStructure:
    """Evidence must sit beside the claim it supports, in both renderers.

    Computed blocks were appended after the whole document, which scattered the
    modelling material across four non-adjacent sections with the charts wedged
    between them: the reader met a conclusion, then unrelated pictures, then the
    numbers the conclusion rested on. Placement is now driven by one table that
    both renderers read.
    """

    @staticmethod
    def _state():
        from state import AgentOutput

        df = pd.DataFrame({"a": range(40), "b": [0, 1] * 20})
        state = WorkflowState(df=df, dataset_name="t.csv")
        state.target_column, state.task_type = "b", "classification"
        state.outputs["profiling"] = AgentOutput(
            agent="profiling", findings={"usable_features": ["a"]})
        state.outputs["modelling"] = AgentOutput(
            agent="modelling",
            findings={"best_model": "Logistic Regression", "best_score": 0.73,
                      "primary_metric": "f1_macro", "beats_baseline": True,
                      "operating_behaviour": {
                          "kind": "classification", "n_predictions": 40,
                          "per_class": [{"class": "1", "actual_count": 20,
                                         "correctly_identified": 14,
                                         "found_per_100": 70.0,
                                         "false_alarms": 3,
                                         "false_alarms_per_100_others": 15.0}]}},
            evidence={"evaluation": {
                "primary_metric": "f1_macro",
                "results": [{"model": "Logistic Regression", "status": "ok",
                             "scores": {"f1_macro": 0.73, "f1_macro_std": 0.01}}]}})
        state.outputs["report"] = AgentOutput(
            agent="report",
            findings={"sections": [
                {"heading": "Exploratory Analysis", "body": "text"},
                {"heading": "Modelling", "body": "text"},
            ]})
        return state

    @staticmethod
    def _headings(rendered: str, html: bool):
        import re as _re

        if html:
            return [h.strip() for h in _re.findall(r"<h2>(.*?)</h2>", rendered)]
        return [l[3:].strip() for l in rendered.split(chr(10))
                if l.startswith("## ")]

    def test_the_evidence_follows_the_section_it_supports(self):
        from orchestrator.export import to_markdown

        headings = self._headings(to_markdown(self._state()), html=False)

        modelling = headings.index("Modelling")
        # Immediately after the narrative, not at the end of the document.
        assert headings[modelling + 1] == "What This Means in Practice"
        assert headings[modelling + 2] == "Model Comparison"

    def test_both_renderers_order_the_document_identically(self):
        """The two have drifted on this project every time one was edited alone.

        Ordering is now driven by a single table, so a divergence here means
        that table stopped being the only source of placement.
        """
        from orchestrator.export import to_html, to_markdown

        state = self._state()
        markdown = [h.replace("Appendix: ", "")
                    for h in self._headings(to_markdown(state), html=False)]
        html = self._headings(to_html(state), html=True)

        assert markdown == html

    def test_evidence_still_appears_when_no_section_anchors_it(self):
        """A computed comparison must not vanish over a heading's wording.

        The model titles its own sections, so the anchor may simply not be
        there. The block is then emitted at the end rather than dropped.
        """
        from orchestrator.export import to_markdown
        from state import AgentOutput

        state = self._state()
        state.outputs["report"] = AgentOutput(
            agent="report",
            findings={"sections": [{"heading": "Findings", "body": "text"}]})

        headings = self._headings(to_markdown(state), html=False)

        assert "Model Comparison" in headings


class TestPracticalBehaviour:
    """A score must be translated into what using the model looks like.

    "macro F1 0.7374" is a number on a scale no reader has intuition for. It
    does not say how many cases of the outcome the model finds, nor how often
    it raises a false alarm -- and a decision turns entirely on those two,
    which trade against each other and cannot be collapsed into one figure.

    The translation is deterministic rather than model-written, for the same
    reason every other number is: a translation that can drift from what was
    computed is not a translation.
    """

    def test_it_states_how_many_of_each_outcome_are_found(self):
        from orchestrator.plain_language import practical_behaviour

        behaviour = {"kind": "classification", "n_predictions": 768,
                     "per_class": [{"class": "1", "actual_count": 268,
                                    "correctly_identified": 153,
                                    "found_per_100": 57.1, "false_alarms": 58,
                                    "false_alarms_per_100_others": 11.6}]}

        text = " ".join(practical_behaviour(behaviour, "Outcome"))

        assert "268" in text and "153" in text
        assert "57.1 in every 100" in text

    def test_it_states_what_was_missed(self):
        """The misses are the half a score hides.

        A model reported at 0.77 accuracy on this data misses 43% of the
        patients who actually have the condition. Both are true; only the
        second tells the reader what they are deciding about.
        """
        from orchestrator.plain_language import practical_behaviour

        behaviour = {"kind": "classification", "n_predictions": 768,
                     "per_class": [{"class": "1", "actual_count": 268,
                                    "correctly_identified": 153,
                                    "found_per_100": 57.1, "false_alarms": 58,
                                    "false_alarms_per_100_others": 11.6}]}

        text = " ".join(practical_behaviour(behaviour, "Outcome"))

        assert "115" in text  # 268 - 153, the missed cases
        assert "missed" in text

    def test_it_refuses_to_judge_the_tradeoff(self):
        """Whether the balance is acceptable is not a fact about the data.

        It depends on what each kind of mistake costs, which is context the
        framework does not have. Choosing for the reader would be exactly the
        overstatement this project exists to prevent.
        """
        from orchestrator.plain_language import practical_behaviour

        behaviour = {"kind": "classification", "n_predictions": 100,
                     "per_class": [{"class": "1", "actual_count": 50,
                                    "correctly_identified": 25,
                                    "found_per_100": 50.0, "false_alarms": 5,
                                    "false_alarms_per_100_others": 10.0}]}

        text = " ".join(practical_behaviour(behaviour, "y")).lower()

        assert "depends on what each kind of mistake costs" in text

    def test_regression_error_is_given_against_the_target_range(self):
        """An error is only large or small relative to what is predicted.

        Being out by 5 is excellent for house prices and useless for shoe
        sizes, so the spread travels with the error.
        """
        from orchestrator.plain_language import practical_behaviour

        behaviour = {"kind": "regression", "n_predictions": 3000,
                     "median_absolute_error": 0.2358,
                     "error_90th_percentile": 0.7964,
                     "central_spread_of_target": 4.2797,
                     "error_as_share_of_spread": 0.0551}

        text = " ".join(practical_behaviour(behaviour, "value"))

        assert "0.2358" in text
        assert "4.28" in text  # the spread it is measured against

    def test_no_behaviour_produces_no_section(self):
        """A run that could not measure this must not invent a section."""
        from orchestrator.plain_language import practical_behaviour

        assert practical_behaviour(None, "y") == []

    def test_both_renderers_carry_it(self):
        from orchestrator.export import to_html, to_markdown
        from state import AgentOutput

        state = WorkflowState(df=pd.DataFrame({"a": range(40), "b": [0, 1] * 20}),
                              dataset_name="t.csv")
        state.target_column, state.task_type = "b", "classification"
        state.outputs["profiling"] = AgentOutput(
            agent="profiling", findings={"usable_features": ["a"]})
        state.outputs["modelling"] = AgentOutput(
            agent="modelling",
            findings={"best_model": "Logistic Regression", "best_score": 0.73,
                      "primary_metric": "f1_macro", "beats_baseline": True,
                      "operating_behaviour": {
                          "kind": "classification", "n_predictions": 40,
                          "per_class": [{"class": "1", "actual_count": 20,
                                         "correctly_identified": 14,
                                         "found_per_100": 70.0,
                                         "false_alarms": 3,
                                         "false_alarms_per_100_others": 15.0}]}})

        for rendered in (to_markdown(state), to_html(state)):
            assert "What This Means in Practice" in rendered
            assert "14" in rendered


class TestEvidenceTables:
    """The computed comparisons must reach the page, in both renderers.

    The modelling stage scores every candidate on five measures with a standard
    deviation each and ranks every feature by permutation importance. All of it
    was reaching the report as one line of prose naming the winner, so a reader
    could not see whether the winner was clear of the field or half a standard
    deviation ahead of four rivals -- which are different findings.
    """

    @staticmethod
    def _state():
        from state import AgentOutput

        df = pd.DataFrame({"a": range(40), "b": [0, 1] * 20})
        state = WorkflowState(df=df, dataset_name="t.csv")
        state.target_column, state.task_type = "b", "classification"
        state.outputs["profiling"] = AgentOutput(
            agent="profiling", findings={"usable_features": ["a"]})
        state.outputs["modelling"] = AgentOutput(
            agent="modelling",
            findings={"best_model": "Logistic Regression", "best_score": 0.7374,
                      "primary_metric": "f1_macro", "beats_baseline": True},
            evidence={"evaluation": {
                "primary_metric": "f1_macro",
                "results": [
                    {"model": "Logistic Regression", "status": "ok",
                     "scores": {"f1_macro": 0.7374, "f1_macro_std": 0.0111,
                                "accuracy": 0.7747, "accuracy_std": 0.0147}},
                    {"model": "Majority baseline", "status": "ok",
                     "is_baseline": True,
                     "scores": {"f1_macro": 0.3943, "f1_macro_std": 0.0008,
                                "accuracy": 0.6510, "accuracy_std": 0.0021}},
                ]},
                "feature_importance": {
                    "test_rows": 192,
                    "importances": [
                        {"feature": "Glucose", "importance": 0.1015,
                         "std": 0.0455, "informative": True},
                        {"feature": "BloodPressure", "importance": 0.0,
                         "std": 0.0102, "informative": False},
                    ]}})
        return state

    def test_both_renderers_show_the_model_comparison(self):
        from orchestrator.export import to_html, to_markdown

        state = self._state()

        for rendered in (to_markdown(state), to_html(state)):
            assert "Model Comparison" in rendered
            # Every candidate, not only the winner.
            assert "Logistic Regression" in rendered
            assert "Majority baseline" in rendered

    def test_the_comparison_carries_the_spread_not_just_the_score(self):
        """Without the standard deviation a reader cannot tell a close race.

        Two models within one standard deviation of each other are not
        distinguished by the data, and reporting only the point estimates hides
        that.
        """
        from orchestrator.export import to_markdown

        rendered = to_markdown(self._state())

        assert "0.7374" in rendered
        assert "0.0111" in rendered

    def test_both_renderers_show_what_the_model_relies_on(self):
        from orchestrator.export import to_html, to_markdown

        state = self._state()

        for rendered in (to_markdown(state), to_html(state)):
            assert "What the Model Relies On" in rendered
            assert "Glucose" in rendered
            # Uninformative features are listed rather than hidden: a reader
            # deciding what to collect next needs to know what did not help.
            assert "BloodPressure" in rendered

    def test_a_run_without_modelling_renders_no_tables(self):
        """An EDA-only run has nothing to compare, and must not claim to."""
        from orchestrator.export import to_markdown
        from state import AgentOutput

        state = WorkflowState(df=pd.DataFrame({"a": range(10)}),
                              dataset_name="t.csv")
        state.outputs["profiling"] = AgentOutput(
            agent="profiling", findings={"usable_features": ["a"]})

        rendered = to_markdown(state)

        assert "Model Comparison" not in rendered


class TestValidatedRevision:
    """A rewrite must be told what was wrong with the draft.

    The orchestrator routes a report that fails validation back to the report
    agent, and the architecture describes this as the report being re-composed
    "with the validator's objections in hand". It was not: the report agent read
    nothing from the validation stage, so the rewrite was built from an
    essentially identical prompt and could only converge by coincidence -- while
    spending two further report and validation round trips to get there.
    """

    @staticmethod
    def _state_with(validation_findings):
        from state import AgentOutput

        df = pd.DataFrame({"a": range(30), "b": [0, 1] * 15})
        state = WorkflowState(df=df, dataset_name="t.csv")
        state.target_column, state.task_type = "b", "classification"
        state.outputs["profiling"] = AgentOutput(
            agent="profiling", findings={"usable_features": ["a"]})
        if validation_findings is not None:
            state.outputs["validation"] = AgentOutput(
                agent="validation", findings=validation_findings)
        return state

    @staticmethod
    def _prompt_for(state, monkeypatch):
        import agents.report_agent as report_agent

        captured = {}
        original = report_agent.ask_json

        def spy(agent, system, user, **kwargs):
            captured["user"] = user
            return original(agent, system, user, **kwargs)

        monkeypatch.setattr(report_agent, "ask_json", spy)
        output = report_agent.ReportAgent().run(state)
        assert output.ok, output.error
        return captured.get("user", "")

    def test_a_rewrite_is_given_the_objections_to_address(self, monkeypatch):
        state = self._state_with({"verdict": "needs_revision", "issues": [
            {"severity": "critical", "stage": "report",
             "problem": "Cited 0.99 accuracy where 0.83 was computed.",
             "suggested_fix": "State the computed 0.83."}]})

        prompt = self._prompt_for(state, monkeypatch)

        assert "Cited 0.99 accuracy" in prompt
        assert "State the computed 0.83." in prompt

    def test_a_first_draft_is_not_told_about_a_revision(self, monkeypatch):
        """Only a rewrite carries objections.

        The first report has nothing to answer for, and an instruction to
        address objections that do not exist would be an invitation to invent
        them.
        """
        prompt = self._prompt_for(self._state_with(None), monkeypatch)

        assert "revision_required" not in prompt

    def test_a_clean_validation_adds_no_objections(self, monkeypatch):
        state = self._state_with({"verdict": "sound", "issues": []})

        prompt = self._prompt_for(state, monkeypatch)

        assert "revision_required" not in prompt


class TestRendererParity:
    """The Markdown and HTML reports must say the same things.

    They are built by two separate functions walking the same state, and the
    drift is always in one direction: something is added to the Markdown and
    the HTML silently lacks it -- in the file people actually open. It has
    happened four times on this project (the at-a-glance card, chart guidance,
    the review heading, and provider provenance), each time caught by reading
    the output rather than by a test. These are those tests.
    """

    @staticmethod
    def _rendered(monkeypatch):
        from orchestrator.export import to_html, to_markdown
        from state import AgentOutput

        df = pd.DataFrame({
            "age": [22, 38, 26, 35, 35, 54, 2, 27, 14, 4] * 6,
            "fare": [7.2, 71.3, 7.9, 53.1, 8.0, 51.9, 21.1, 11.1, 30.1, 16.7] * 6,
            "survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 1] * 6,
        })
        state = WorkflowState(df=df, dataset_name="parity.csv")
        state.target_column, state.task_type = "survived", "classification"
        state.plan = ["profiling", "modelling", "report", "validation"]
        state.outputs["profiling"] = AgentOutput(
            agent="profiling", findings={"usable_features": ["age", "fare"]})
        state.outputs["modelling"] = AgentOutput(
            agent="modelling",
            findings={"best_model": "Random Forest", "best_score": 0.8311,
                      "primary_metric": "f1_macro", "beats_baseline": True,
                      "permutation_p_value": 0.0196})
        state.outputs["validation"] = AgentOutput(
            agent="validation",
            findings={"verdict": "needs_revision",
                      "issue_counts": {"critical": 0, "warning": 2, "info": 1},
                      "issues": [{"severity": "warning", "stage": "eda",
                                  "problem": "A claim could not be reconciled.",
                                  "suggested_fix": "State the measured value."}]})
        # A report section, because the glossary is driven by the prose the
        # interpreting model actually wrote. Terms reach the report through
        # this text, which is exactly why they cannot be glossed at the point
        # of use and are defined once at the end instead.
        state.outputs["report"] = AgentOutput(
            agent="report",
            findings={"n_sections": 1, "sections": [
                {"heading": "What the Model Found",
                 "body": ("The macro F1 of 0.8311 was reached by a Random "
                          "Forest, well clear of the baseline.")}]})
        state.usage = {"calls": 9, "total_tokens": 12_000, "latency_s": 31.2,
                       "by_provider": {"groq": 7, "gemini": 2}}
        return to_markdown(state), to_html(state)

    def test_both_renderers_open_with_the_plain_language_card(self, monkeypatch):
        markdown, html = self._rendered(monkeypatch)

        assert "at a glance" in markdown.lower()
        assert "at a glance" in html.lower()
        # The answer must be stated in words, not left to the reader to infer
        # from a metric, in both.
        for rendered in (markdown, html):
            assert "predicts it reliably" in rendered

    def test_both_renderers_name_the_provider_that_answered(self, monkeypatch):
        """A failover run's configured backend is "auto", which names no model.

        The report knows which provider actually wrote each narrative, so
        stating only the setting would be vaguer about its own provenance than
        the run was.
        """
        markdown, html = self._rendered(monkeypatch)

        for rendered in (markdown, html):
            assert "groq" in rendered
            assert "gemini" in rendered

    def test_both_renderers_frame_the_verdict_for_a_non_specialist(self, monkeypatch):
        """"NEEDS REVISION" reads to a non-specialist as "this is broken".

        It means the automated reviewer found points it could not reconcile,
        which is a reason to read those points, not to discard the analysis.
        """
        markdown, html = self._rendered(monkeypatch)

        assert "Validation: NEEDS REVISION" not in markdown
        for rendered in (markdown, html):
            assert "confirm" in rendered.lower()

    def test_both_renderers_define_only_the_terms_used(self, monkeypatch):
        markdown, html = self._rendered(monkeypatch)

        for rendered in (markdown, html):
            assert "macro F1" in rendered
            # Not used by this report, so not defined in it.
            assert "multicollinearity" not in rendered


class TestFailoverBackend:
    """Several free tiers used as one budget.

    The framework's wall-clock on a real provider was measured to be almost
    entirely spent asleep: stage durations of 62, 150 and 450 seconds on stages
    whose computation takes under a second, because the retry logic was waiting
    out a rate limit. Moving to a different provider's allowance is the cheaper
    answer, and these tests pin the decisions that make that safe.
    """

    @staticmethod
    def _rate_limited(name):
        class RateLimited:
            def __init__(self):
                self.name = name
                self.calls = 0

            def complete(self, system, user):
                self.calls += 1
                error = LLMError(f"{name} returned 429: rate limit reached")
                error.rate_limited = True
                raise error
        return RateLimited()

    @staticmethod
    def _working(name):
        from agents.llm import LLMResponse

        class Working:
            def __init__(self):
                self.name = name
                self.calls = 0

            def complete(self, system, user):
                self.calls += 1
                return LLMResponse(text="{}", input_tokens=1, output_tokens=1,
                                   latency_s=0.0, backend=name)
        return Working()

    def test_a_rate_limited_provider_hands_over_to_the_next(self):
        from agents.llm import FailoverBackend

        spent, spare = self._rate_limited("spent"), self._working("spare")
        chain = FailoverBackend([spent, spare])

        assert chain.complete("s", "u").backend == "spare"

    def test_a_rate_limited_provider_is_not_tried_again_immediately(self):
        """The point of failover is to stop paying the cost repeatedly.

        Handing over once but re-attempting the exhausted provider on every
        subsequent call would reintroduce the latency the failover exists to
        avoid, one request at a time.
        """
        from agents.llm import FailoverBackend

        spent, spare = self._rate_limited("spent"), self._working("spare")
        chain = FailoverBackend([spent, spare], cooldown_s=300)

        for _ in range(3):
            chain.complete("s", "u")

        assert spent.calls == 1
        assert spare.calls == 3

    def test_a_non_rate_limit_failure_does_not_rest_the_provider(self):
        """A refused request says nothing about the account's allowance.

        A malformed prompt or an unexpected response shape is a property of
        that one request. Setting the provider aside for it would discard a
        working provider over a transient, request-specific fault.
        """
        from agents.llm import FailoverBackend

        class Refuses:
            name = "refuses"

            def complete(self, system, user):
                raise LLMError("returned 400: malformed request")

        refuses, spare = Refuses(), self._working("spare")
        chain = FailoverBackend([refuses, spare])
        chain.complete("s", "u")

        assert "refuses" not in chain._resting

    def test_every_provider_exhausted_still_attempts_one(self):
        """Resting is a guess about when an allowance returns, not a fact.

        If the guess is wrong for every provider, refusing to send anything is
        a worse outcome than trying the one that has rested longest.
        """
        from agents.llm import FailoverBackend

        spent = self._rate_limited("spent")
        chain = FailoverBackend([spent], cooldown_s=300)

        with pytest.raises(LLMError):
            chain.complete("s", "u")
        assert [b.name for b in chain._available()] == ["spent"]

    def test_no_configured_provider_says_what_to_set(self):
        from agents.llm import FailoverBackend

        with pytest.raises(LLMError, match="GROQ_API_KEY"):
            FailoverBackend([])

    def test_the_chain_retries_less_than_a_lone_provider(self, monkeypatch):
        """Within a chain, waiting out a limit is the slower of two options.

        Six attempts is right when a provider is the only one there is. With
        another allowance available, spending minutes on the first is exactly
        the cost this feature removes.
        """
        from agents import llm

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test-key-not-used")
        monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
        chain = llm.build_failover(["groq", "gemini"])

        assert len(chain.backends) == 2
        assert all(b.max_attempts == 2 for b in chain.backends)
        assert llm.GroqBackend.max_attempts == 6

    def test_an_unconfigured_provider_does_not_break_the_chain(self, monkeypatch):
        """One absent optional key must not take the working providers with it.

        This failed in practice. Backends report a missing key as `LLMError`,
        except Anthropic's, which raises a plain `RuntimeError` from the config
        object -- so building a chain on a machine with a Groq key but no
        Anthropic key raised instead of returning the Groq backend. A provider
        that cannot be constructed is simply one this machine has not set up,
        whatever exception says so.
        """
        from agents import llm

        monkeypatch.setenv("GROQ_API_KEY", "gsk_test-key-not-used")
        for absent in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROK_API_KEY",
                       "ANTHROPIC_API_KEY"):
            monkeypatch.delenv(absent, raising=False)
        monkeypatch.setattr(CONFIG.llm, "api_key", None, raising=False)

        chain = llm.build_failover()

        assert [b.name for b in chain.backends] == ["groq"]
        assert "anthropic" in chain.skipped

    def test_the_backend_names_are_one_list(self):
        """The registry is the single source of the valid names.

        This drifted once already: `--backend groq` was refused by the
        evaluation CLI's hand-written argparse choices while the backend itself
        existed and worked.
        """
        from agents import llm

        assert set(llm.BACKEND_NAMES) == set(llm.BACKEND_CONSTRUCTORS)
        assert "groq" in llm.BACKEND_NAMES
        assert "auto" in llm.BACKEND_NAMES


class TestPromptBudget:
    """Prompts that accumulate every stage's findings must be made to fit.

    An oversized request is refused outright rather than rate-limited, so no
    amount of waiting recovers it. The stages affected are the report and the
    validator, which are the two the framework can least afford to lose.
    """

    def test_a_small_payload_is_untouched(self):
        from agents.llm import bound_payload

        payload = {"a": "short", "b": [1, 2, 3]}
        result, trimmed = bound_payload(payload, 10_000)
        assert result == payload
        assert trimmed is False

    def test_an_oversized_payload_is_brought_under_budget(self):
        from agents.llm import bound_payload
        import json as _json

        payload = {f"agent_{i}": {"narrative": "x" * 5_000,
                                  "findings": [f"item {j}" for j in range(200)]}
                   for i in range(8)}
        assert len(_json.dumps(payload)) > 20_000

        result, trimmed = bound_payload(payload, 20_000)
        assert trimmed is True
        assert len(_json.dumps(result, default=str)) <= 20_000

    def test_trimming_keeps_every_agent_rather_than_dropping_some(self):
        """Losing a stage entirely would hide it; every key must survive."""
        from agents.llm import bound_payload

        payload = {f"agent_{i}": {"narrative": "y" * 4_000} for i in range(8)}
        result, trimmed = bound_payload(payload, 5_000)
        assert trimmed is True
        assert set(result.keys()) == set(payload.keys())

    def test_omitted_list_items_are_declared_not_silently_dropped(self):
        from agents.llm import bound_payload

        payload = {"findings": [f"finding number {i}" for i in range(400)]}
        result, _ = bound_payload(payload, 500)
        assert any("more omitted" in str(x) for x in result["findings"])


class TestNumberCheckPrecision:
    """The number check must not manufacture its own false positives.

    Found by a real-model run: computed values were stored rounded to four
    decimal places, so a report citing five decimals was compared against a
    value the checker had itself flattened. The difference then exceeded the
    tolerance and correct numbers were reported as unsupported. A check that
    cries wolf is worse than no check, because the reader learns to ignore it.
    """

    def test_a_five_decimal_citation_of_a_computed_valueis_supported(self):

        computed = as_floats({normalise(v) for v in
                                   (0.008014, 0.019941, 0.021053)})
        for written, decimals in (("0.00801", 5), ("0.01994", 5), ("0.02105", 5)):
            value = float(written)
            tolerance = 0.5 * (10 ** -decimals)
            assert is_supported(value, tolerance, computed), written

    def test_an_invented_number_is_still_rejected(self):
        """Loosening precision must not blunt the check it exists to perform."""

        computed = as_floats({normalise(v) for v in (0.008014, 0.5224)})
        assert is_supported(0.9999, 0.5 * (10 ** -4), computed) is False
        assert is_supported(0.7311, 0.5 * (10 ** -4), computed) is False

    def test_precision_survives_the_round_trip(self):

        values = as_floats({normalise(0.008014)})
        assert values[0] == pytest.approx(0.008014, abs=1e-9)


class TestPromptBudgetGuarantee:
    """Trimming must actually meet the budget, and lose the right things.

    Approaching the budget is not enough: an oversized request is refused
    outright, so a payload that is merely smaller still loses the stage. These
    tests pin the guarantee and the priority order it uses to get there.
    """

    def _material(self):
        return {
            f"stage_{i}": {
                "narrative": "Interpretation sentence. " * 60,
                "findings": {
                    "headline_score": 0.8324,
                    "columns": [f"column_{j}" for j in range(120)],
                    "pairs": [{"a": f"f{j}", "b": f"g{j}", "r": 0.5}
                              for j in range(80)],
                },
            }
            for i in range(6)
        }

    def test_the_budget_is_met_not_merely_approached(self):
        from agents.llm import bound_payload
        import json as _json

        result, trimmed = bound_payload(self._material(), 8_000)
        assert trimmed is True
        assert len(_json.dumps(result, default=str)) <= 8_000

    def test_every_stage_survives_with_its_narrative(self):
        """A stage reduced to nothing is invisible; that must never happen."""
        from agents.llm import bound_payload

        material = self._material()
        result, _ = bound_payload(material, 8_000)
        assert set(result) == set(material)
        for stage in result.values():
            assert stage.get("narrative")

    def test_bulk_detail_goes_before_headline_values(self):
        from agents.llm import bound_payload

        result, _ = bound_payload(self._material(), 8_000)
        kept = [s for s in result.values()
                if isinstance(s.get("findings"), dict)
                and "headline_score" in s["findings"]]
        assert kept, "the headline value was dropped before the bulk lists"

    def test_the_callers_material_is_not_mutated(self):
        """Summarising for a prompt must not alter the findings of record."""
        from agents.llm import bound_payload
        import copy

        material = self._material()
        before = copy.deepcopy(material)
        bound_payload(material, 8_000)
        assert material == before


class TestFeatureEngineering:
    """Derived features must earn their place, not be assumed to help.

    This is where a data science pipeline most easily flatters itself: new
    columns always look like progress, and a wider frame feels richer whether or
    not it carries more signal.
    """

    def test_a_log_transform_is_applied_only_to_a_right_tail(self):
        """log1p lengthens a left tail; applying it there makes skew worse."""
        from tools.features import engineer_features

        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "right_skewed": rng.lognormal(1.0, 1.2, 400),
            "left_skewed": 100 - rng.lognormal(1.0, 1.2, 400),
            "y": rng.integers(0, 2, 400),
        })
        applied = engineer_features(df, "y")["applied"]
        logged = {a["source"] for a in applied if a["kind"] == "log_transform"}
        assert "right_skewed" in logged
        assert "left_skewed" not in logged

    def test_an_identifier_is_not_grouped_into_a_constant(self):
        """Every value distinct means every value rare; grouping yields nothing."""
        from tools.features import engineer_features

        df = pd.DataFrame({
            "row_id": [f"id-{i}" for i in range(300)],
            "value": np.linspace(0, 1, 300),
            "y": [i % 2 for i in range(300)],
        })
        applied = engineer_features(df, "y")["applied"]
        grouped = [a for a in applied
                   if a["kind"] == "rare_category_grouping" and a["source"] == "row_id"]
        assert not grouped

    def test_missingness_is_preserved_before_imputation_erases_it(self):
        from tools.features import engineer_features

        rng = np.random.default_rng(1)
        values = rng.normal(size=400)
        values[:120] = np.nan
        df = pd.DataFrame({"partly_missing": values, "y": rng.integers(0, 2, 400)})
        applied = engineer_features(df, "y")["applied"]
        assert any(a["kind"] == "missing_indicator" and a["source"] == "partly_missing"
                   for a in applied)

    def test_target_encoding_is_declined_and_the_reason_recorded(self):
        """A leaking feature must be absent *and* explained, not silently missing."""
        from tools.features import engineer_features

        df = pd.DataFrame({"cat": [f"c{i % 40}" for i in range(400)],
                           "y": [i % 2 for i in range(400)]})
        declined = engineer_features(df, "y")["declined"]
        target_encoding = [d for d in declined if d["kind"] == "target_encoding"]
        assert target_encoding
        assert "leak" in target_encoding[0]["reason"].lower()

    def test_the_target_is_never_transformed_into_a_feature(self):
        """Deriving from the target would leak the answer into the inputs."""
        from tools.features import engineer_features

        rng = np.random.default_rng(2)
        df = pd.DataFrame({"x": rng.normal(size=300),
                           "skewed_target": rng.lognormal(1.0, 1.5, 300)})
        applied = engineer_features(df, "skewed_target")["applied"]
        assert all(a["source"] != "skewed_target" for a in applied)

    def test_a_negligible_gain_does_not_adopt_the_features(self, titanic, mock):
        from agents.feature_agent import FeatureAgent

        state = profiled(titanic)
        before = state.data.shape[1]
        output = FeatureAgent().run(state)
        state.record(output)
        if output.findings.get("skipped"):
            pytest.skip("features stage skipped for this fixture")
        if not output.findings["adopted"]:
            assert state.data.shape[1] == before

    def test_claiming_a_benefit_for_rejected_features_is_flagged(self, titanic, mock):
        """The guard against dressing up a rejected result."""

        assert overstated_feature_benefit("The derived features improved accuracy markedly.")
        # The honest statement uses the same words and must not be flagged.
        assert not overstated_feature_benefit(
            "The derived features did not improve accuracy and were discarded.")
        assert not overstated_feature_benefit(
            "Engineering does not boost performance on this dataset.")


class TestDistributedCorrelations:
    """The Spark path must return what the pandas path returns, key for key.

    An earlier version returned a shape of its own. Nothing failed loudly: the
    EDA agent would simply have received keys it did not understand, and the
    identical-contract claim the design rests on would have been false in the
    place it matters most.
    """

    def test_the_contract_matches_the_pandas_tool(self, tmp_path):
        from tools import spark_engine
        from tools.eda import correlations as pandas_correlations

        ok, _ = spark_engine.spark_available()
        if not ok:
            pytest.skip("Spark is not available in this environment")

        rng = np.random.default_rng(7)
        base = rng.normal(size=400)
        df = pd.DataFrame({
            "a": base,
            "b": base * 0.9 + rng.normal(scale=0.3, size=400),   # strongly related
            "c": rng.normal(size=400),                            # unrelated
        })
        path = tmp_path / "corr.csv"
        df.to_csv(path, index=False)

        expected = pandas_correlations(df)
        actual = spark_engine.correlations(
            spark_engine.load(str(path)), ["a", "b", "c"])

        assert set(expected).issubset(set(actual))
        assert len(actual["pairs"]) == len(expected["pairs"])
        for pair in actual["pairs"]:
            assert set(pair) >= {"column_a", "column_b", "pearson", "spearman",
                                 "strength", "direction", "non_linear", "n"}

    def test_both_engines_agree_on_the_coefficients(self, tmp_path):
        from tools import spark_engine
        from tools.eda import correlations as pandas_correlations

        ok, _ = spark_engine.spark_available()
        if not ok:
            pytest.skip("Spark is not available in this environment")

        rng = np.random.default_rng(11)
        base = rng.normal(size=500)
        df = pd.DataFrame({
            "x": base,
            "y": base * 1.2 + rng.normal(scale=0.4, size=500),
            "z": -base * 0.8 + rng.normal(scale=0.5, size=500),
        })
        path = tmp_path / "agree.csv"
        df.to_csv(path, index=False)

        expected = {frozenset((p["column_a"], p["column_b"])): p
                    for p in pandas_correlations(df)["pairs"]}
        actual = spark_engine.correlations(
            spark_engine.load(str(path)), ["x", "y", "z"])["pairs"]

        assert actual, "no pairs were returned to compare"
        for pair in actual:
            other = expected[frozenset((pair["column_a"], pair["column_b"]))]
            assert pair["pearson"] == pytest.approx(other["pearson"], abs=1e-4)
            assert pair["spearman"] == pytest.approx(other["spearman"], abs=1e-4)


class TestModelSelection:
    """Selecting on a difference smaller than its own noise is selecting on noise.

    Observed on wine quality: the macro-F1 leader won by 0.0138 with a standard
    deviation of 0.0314 -- indistinguishable from four other models -- while
    scoring 5.7 accuracy points worse than one of them. Reporting it understated
    the framework for no gain on the metric it was chosen for.
    """

    @staticmethod
    def _result(model, f1, std, accuracy, baseline=False):
        return {
            "model": model, "status": "ok", "is_baseline": baseline,
            "primary_metric": "f1_macro", "primary_score": f1,
            "scores": {"f1_macro": f1, "f1_macro_std": std, "accuracy": accuracy},
        }

    def test_a_tie_on_the_primary_metric_is_broken_on_accuracy(self):
        from tools.modelling import _select_best

        results = [
            self._result("Decision Tree", 0.2893, 0.0314, 0.5379),
            self._result("Extra Trees", 0.2755, 0.0136, 0.5953),
        ]
        best, selection = _select_best(results, "f1_macro", "classification")
        assert best["model"] == "Extra Trees"
        assert selection["changed_selection"] is True
        assert selection["leader_on_primary"] == "Decision Tree"

    def test_a_real_lead_on_the_primary_metric_is_respected(self):
        """A gap wider than the noise must not be overridden by accuracy."""
        from tools.modelling import _select_best

        results = [
            self._result("Strong", 0.8000, 0.0100, 0.8100),
            self._result("Weak but accurate", 0.6000, 0.0100, 0.9500),
        ]
        best, selection = _select_best(results, "f1_macro", "classification")
        assert best["model"] == "Strong"
        assert selection.get("changed_selection") in (False, None)

    def test_a_baseline_is_never_selected_over_a_real_model(self):
        from tools.modelling import _select_best

        results = [
            self._result("Majority baseline", 0.9000, 0.0100, 0.9900, baseline=True),
            self._result("Real model", 0.5000, 0.0100, 0.6000),
        ]
        best, _ = _select_best(results, "f1_macro", "classification")
        assert best["model"] == "Real model"

    def test_regression_ties_prefer_the_more_stable_model(self):
        """With no accuracy to compare, the steadier model across folds wins."""
        from tools.modelling import _select_best

        results = [
            {"model": "Volatile", "status": "ok", "is_baseline": False,
             "primary_metric": "r2", "primary_score": 0.900,
             "scores": {"r2": 0.900, "r2_std": 0.080}},
            {"model": "Steady", "status": "ok", "is_baseline": False,
             "primary_metric": "r2", "primary_score": 0.880,
             "scores": {"r2": 0.880, "r2_std": 0.005}},
        ]
        best, _ = _select_best(results, "r2", "regression")
        assert best["model"] == "Steady"

    def test_fold_spread_does_not_tie_a_clearly_worse_model(self):
        """The tolerance is a standard error, not a fold standard deviation.

        Kaggle House Prices, 5-fold CV: one fold holds the known outlier houses
        and R-squared varies by 0.12 across folds. Under a one-sd tolerance a
        Decision Tree 11.5 points behind the leader counted as tied, and won
        on its lower variance.
        """
        from tools.modelling import _select_best

        def reg(model, r2, std):
            return {"model": model, "status": "ok", "is_baseline": False,
                    "primary_metric": "r2", "primary_score": r2,
                    "scores": {"r2": r2, "r2_std": std}}

        results = [reg("Gradient Boosting", 0.8530, 0.1165),
                   reg("LightGBM", 0.8485, 0.0867),
                   reg("Decision Tree", 0.7378, 0.0697)]
        best, selection = _select_best(results, "r2", "regression", folds=5)
        assert best["model"] != "Decision Tree"
        assert "Decision Tree" not in selection["tied_within_1_se"]

    def test_the_selection_rule_is_recorded_for_the_reader(self):
        """A changed pick must be explainable, not an unexplained model name."""
        from tools.modelling import _select_best

        results = [
            self._result("Decision Tree", 0.2893, 0.0314, 0.5379),
            self._result("Extra Trees", 0.2755, 0.0136, 0.5953),
        ]
        _, selection = _select_best(results, "f1_macro", "classification")
        assert "accuracy" in selection["rule"]
        assert set(selection["tied_within_1_se"]) == {"Decision Tree", "Extra Trees"}


class TestLogTargetCandidates:
    """Log-scale regressors are offered for positive, right-skewed targets.

    A price-like target spans orders of magnitude, and its errors are
    proportional rather than additive. Fitting on log1p and inverting with
    expm1 matches that. The variants are added to the field and must win on
    cross-validation like any other candidate; they never replace the raw ones.
    """

    def test_warranted_only_for_positive_skewed_regression(self):
        from tools.modelling import _log_target_warranted

        rng = np.random.default_rng(0)
        skewed = pd.Series(rng.lognormal(12, 0.6, 500))
        assert _log_target_warranted(skewed, "regression")
        assert not _log_target_warranted(skewed, "classification")
        assert not _log_target_warranted(pd.Series(rng.normal(0, 1, 500)), "regression")
        assert not _log_target_warranted(skewed - skewed.median(), "regression")

    def test_variants_are_added_not_substituted(self):
        from tools.modelling import _candidate_models

        plain = _candidate_models("regression", 1000)
        logged = _candidate_models("regression", 1000, log_target=True)
        assert set(plain) < set(logged)
        added = set(logged) - set(plain)
        assert added and all(name.endswith("(log target)") for name in added)

    def test_a_log_target_model_predicts_on_the_original_scale(self):
        from tools.modelling import _candidate_models

        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 3))
        y = np.exp(11 + X[:, 0] * 0.5 + rng.normal(0, 0.1, 300))
        model = _candidate_models("regression", 300, log_target=True)[
            "Ridge Regression (log target)"]
        pred = model.fit(X, y).predict(X)
        assert 0.5 < np.median(pred) / np.median(y) < 2.0


class TestVerificationLayer:
    """The verification layer must be enumerable, not merely asserted.

    Before this module existed the checks were scattered across nine agent files
    as `warnings.append` calls. The documentation claimed ten mechanisms and no
    reader could confirm it -- the real count turned out to be twelve. A
    contribution described as "verification as an architectural layer" has to be
    a layer you can list.
    """

    def test_every_mechanism_declares_what_it_compares_against(self):
        """A check with no computed ground truth is a prompt, not a check."""
        import verification as V

        assert V.MECHANISMS, "no mechanisms registered"
        for mechanism in V.MECHANISMS:
            assert mechanism.name
            assert mechanism.stage
            assert mechanism.checks
            assert mechanism.compares_against, (
                f"{mechanism.name} does not say what it verifies against")

    def test_mechanism_names_are_unique(self):
        import verification as V

        names = [m.name for m in V.MECHANISMS]
        assert len(names) == len(set(names)), f"duplicate mechanism names: {names}"

    def test_the_documented_count_matches_the_code(self):
        """The count in the docs is derived from the registry, so it cannot drift."""
        import io
        from pathlib import Path

        import verification as V

        count = len(V.MECHANISMS)
        readme = Path(__file__).resolve().parents[1] / "README.md"
        text = io.open(readme, encoding="utf-8").read()
        assert f"{count} verification mechanisms" in text or f"**{count}**" in text, (
            f"README does not state the actual mechanism count of {count}")

    def test_negation_is_respected_across_every_claim_vocabulary(self):
        """The shared matcher must not punish an agent for reporting honestly."""
        import verification as V

        for terms, affirmative, negated in (
            (V.SEGMENT_CLAIMS,
             "The rows fall into three distinct groups.",
             "The rows do not fall into distinct groups."),
            (V.BENEFIT_CLAIMS,
             "The derived columns improved accuracy.",
             "The derived columns did not improve accuracy."),
        ):
            assert V.asserts_any(affirmative, terms), affirmative
            assert not V.asserts_any(negated, terms), negated

    def test_a_negation_in_a_previous_sentence_does_not_excuse_a_claim(self):
        """The window must not reach back across a sentence boundary."""
        import verification as V

        text = ("The data is not missing any values. "
                "It separates into three clear segments.")
        assert V.asserts_any(text, V.SEGMENT_CLAIMS), (
            "an unrelated earlier negation wrongly suppressed the claim")

    def test_a_suppressed_negative_result_is_detected(self):
        import verification as V

        silent = "The model achieved an accuracy of 0.52 across five folds."
        honest = "The model was no better than chance and found no reliable signal."
        assert V.suppressed_negative_result(silent, beats_baseline=False) is True
        assert V.suppressed_negative_result(honest, beats_baseline=False) is False
        # When signal was found there is nothing to suppress.
        assert V.suppressed_negative_result(silent, beats_baseline=True) is False


class TestResponseCache:
    """Replaying model responses must be sound, honest, and invalidated correctly.

    The cache exists because free API tiers meter tokens per minute, which makes
    iteration slow rather than expensive. It is only defensible because the
    framework fixes temperature at 0.0: the provider is being asked for a
    deterministic answer, so a stored one is the answer it would give.
    """

    def test_an_identical_request_is_replayed(self, tmp_path, monkeypatch):
        from agents import cache

        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "c")
        assert cache.lookup("groq", "m", "sys", "user") is None
        cache.store("groq", "m", "sys", "user", '{"ok": true}', 10, 5)

        stored = cache.lookup("groq", "m", "sys", "user")
        assert stored is not None
        assert stored["text"] == '{"ok": true}'
        assert stored["input_tokens"] == 10

    def test_an_edited_prompt_does_not_reuse_the_old_answer(self, tmp_path, monkeypatch):
        """The key covers both prompts, so changing an agent invalidates its cache."""
        from agents import cache

        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "c")
        cache.store("groq", "m", "system v1", "user", '{"v": 1}', 1, 1)

        assert cache.lookup("groq", "m", "system v2", "user") is None
        assert cache.lookup("groq", "m", "system v1", "user")["text"] == '{"v": 1}'

    def test_a_different_model_does_not_share_entries(self, tmp_path, monkeypatch):
        """Two models given the same prompt are two different questions."""
        from agents import cache

        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "c")
        cache.store("groq", "model-a", "sys", "user", '{"from": "a"}', 1, 1)
        assert cache.lookup("groq", "model-b", "sys", "user") is None

    def test_disabling_the_cache_stops_both_reads_and_writes(self, tmp_path, monkeypatch):
        from agents import cache

        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "c")
        monkeypatch.setattr(CONFIG, "llm_cache_enabled", False)
        cache.store("groq", "m", "sys", "user", '{"ok": true}', 1, 1)
        assert cache.lookup("groq", "m", "sys", "user") is None

    def test_a_corrupt_entry_is_a_miss_rather_than_a_crash(self, tmp_path, monkeypatch):
        """An interrupted write must cost one API call, not a broken run."""
        from agents import cache

        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "c")
        cache.store("groq", "m", "sys", "user", '{"ok": true}', 1, 1)
        path = cache._path(cache._key("groq", "m", "sys", "user"))
        path.write_text("{ truncated", encoding="utf-8")
        assert cache.lookup("groq", "m", "sys", "user") is None

    def test_the_mock_backend_is_never_cached(self, titanic, mock):
        """A test that registers a new mock must receive it, not a stored reply."""
        from agents import llm

        llm.USAGE.reset()
        mock.register("profiling", lambda _: {
            "dataset_description": "first", "domain": "d", "usable_features": [],
            "excluded_columns": [], "target_column": None, "task_type": "none",
            "target_reasoning": "r", "concerns": [], "narrative": "n"})
        first = llm.ask_json("profiling", "sys", "user")

        mock.register("profiling", lambda _: {
            "dataset_description": "second", "domain": "d", "usable_features": [],
            "excluded_columns": [], "target_column": None, "task_type": "none",
            "target_reasoning": "r", "concerns": [], "narrative": "n"})
        second = llm.ask_json("profiling", "sys", "user")

        assert first["dataset_description"] == "first"
        assert second["dataset_description"] == "second"
        assert llm.USAGE.cache_hits == 0

    def test_cache_hits_are_not_counted_as_api_calls(self):
        """Reported cost must describe a first run, not a warm cache."""
        from agents.llm import UsageTracker

        usage = UsageTracker()
        usage.record_cache_hit("eda")
        assert usage.cache_hits == 1
        assert usage.calls == 0
        assert usage.summary()["cache_hits"] == 1


class TestConsoleEncoding:
    """A non-ASCII character in a report must not crash the command line.

    Found on a real-model run: Windows consoles default to cp1252, which cannot
    encode the non-breaking hyphens, em-dashes and curly quotes language models
    emit routinely. The report files are UTF-8 and were written correctly, but
    printing a summary of one raised UnicodeEncodeError *after* the whole
    analysis had succeeded -- six minutes of work appearing to fail on the last
    line, with the output sitting correctly on disk.
    """

    @staticmethod
    def _cp1252_stream():
        import io as _io
        return _io.TextIOWrapper(_io.BytesIO(), encoding="cp1252", errors="strict")

    def test_a_cp1252_console_would_reject_model_punctuation(self):
        """Establishes the hazard the fix addresses, so it is not hypothetical."""
        stream = self._cp1252_stream()
        with pytest.raises(UnicodeEncodeError):
            stream.write("cross‑validation")   # non-breaking hyphen
            stream.flush()

    def test_every_entry_point_makes_the_console_safe(self):
        """run, demo and doctor all print report text, so all three need it."""
        import importlib

        for name in ("run", "demo", "doctor"):
            module = importlib.import_module(name)
            assert hasattr(module, "_make_console_unicode_safe"), (
                f"{name}.py can print report text but does not guard the console")

    def test_the_guard_survives_a_stream_that_cannot_reconfigure(self, monkeypatch):
        """A redirected or wrapped stream must not turn into a crash of its own."""
        import io as _io
        import run

        class Unreconfigurable(_io.StringIO):
            def reconfigure(self, **kwargs):
                raise AttributeError("not supported here")

        monkeypatch.setattr(sys, "stdout", Unreconfigurable())
        monkeypatch.setattr(sys, "stderr", Unreconfigurable())
        run._make_console_unicode_safe()   # must not raise
