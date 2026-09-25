"""Regression tests for defects found by code review.

Each class here pins one defect: what failed, and the behaviour that replaced
it. They are kept apart from `test_framework.py` so the set of review findings
and their fixes can be read in one place.
"""
from __future__ import annotations

import http.client
import json
import time
import urllib.request

import numpy as np
import pandas as pd
import pytest

from agents import cache, llm
from agents.llm import LLMError, bound_payload
from config import CONFIG


# --------------------------------------------------------------------------- #
# The model-call layer
# --------------------------------------------------------------------------- #

class _Reply:
    """A stand-in for the object `urllib.request.urlopen` returns."""

    def __init__(self, body: bytes = b"", fail: Exception | None = None):
        self.body, self.fail = body, fail

    def read(self) -> bytes:
        if self.fail:
            raise self.fail
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestRequestFailuresReachTheFailoverChain:
    """Every transport failure must leave `_request_json` as `LLMError`.

    The failover chain and every agent's fallback catch that type and nothing
    else. A gateway page served with status 200, a timeout while the body was
    being read, or a body cut short escaped as other exception types: the chain
    did not move to the next provider, and a failure at the planning call took
    down the whole run instead of falling back to the full pipeline.
    """

    @pytest.mark.parametrize("reply", [
        _Reply(b"<html>Bad gateway</html>"),
        _Reply(fail=TimeoutError("read timed out")),
        _Reply(fail=ConnectionResetError("reset by peer")),
        _Reply(fail=http.client.IncompleteRead(b"{", 40)),
    ], ids=["html-200", "read-timeout", "connection-reset", "truncated-body"])
    def test_the_failure_is_an_llm_error(self, reply, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: reply)
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        with pytest.raises(LLMError):
            llm._request_json(lambda: None, "Groq", max_attempts=2)

    def test_the_chain_moves_on_after_such_a_failure(self, monkeypatch):
        class Broken:
            name, model = "first", "m"

            def complete(self, system, user):
                return llm.GroqBackend.complete(self, system, user)

        class Working:
            name, model = "second", "m"

            def complete(self, system, user):
                return llm.LLMResponse(text='{"ok": true}', backend="second")

        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda req, timeout: _Reply(b"<html>down</html>"))
        monkeypatch.setattr(time, "sleep", lambda seconds: None)
        broken = Broken()
        broken.max_attempts, broken.base_url, broken.api_key = 1, "http://x", "k"
        chain = llm.FailoverBackend([broken, Working()])

        assert chain.complete("sys", "user").backend == "second"

    def test_a_backend_that_cannot_start_degrades_the_stage(self, monkeypatch):
        """Anthropic's constructor raises RuntimeError for a missing key."""
        def unbuildable():
            raise RuntimeError("ANTHROPIC_API_KEY is not set.")

        monkeypatch.setattr(llm, "get_backend", unbuildable)
        with pytest.raises(LLMError, match="could not be started"):
            llm.ask_json("profiling", "sys", "user")


class _Scripted:
    """A backend that returns a fixed sequence of reply texts."""

    name, model = "groq", "m"

    def __init__(self, *texts: str):
        self.texts, self.calls = list(texts), 0

    def complete(self, system, user):
        self.calls += 1
        return llm.LLMResponse(text=self.texts.pop(0), backend="groq")


class TestOnlyValidatedRepliesAreCached:
    """A reply is written to the cache only after it has been validated.

    Storing on arrival made one bad reply permanent: a truncated body and every
    retry's reply were all saved, and every later run replayed them with no
    network call, so the stage degraded on every run until the cache was
    cleared by hand.
    """

    @pytest.fixture(autouse=True)
    def isolated_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(CONFIG, "llm_cache_enabled", True)

    def test_an_invalid_reply_is_not_stored(self, monkeypatch):
        backend = _Scripted('{"trunc', '{"trunc', '{"trunc')
        monkeypatch.setattr(llm, "get_backend", lambda: backend)
        with pytest.raises(LLMError):
            llm.ask_json("report", "sys", "user", required_keys=("a",))
        assert cache.statistics()["entries"] == 0

    def test_a_valid_reply_is_stored_and_replayed(self, monkeypatch):
        backend = _Scripted('{"a": 1}')
        monkeypatch.setattr(llm, "get_backend", lambda: backend)
        llm.ask_json("report", "sys", "user", required_keys=("a",))

        replay = _Scripted()
        monkeypatch.setattr(llm, "get_backend", lambda: replay)
        assert llm.ask_json("report", "sys", "user", required_keys=("a",)) == {"a": 1}
        assert replay.calls == 0

    def test_a_bad_entry_left_by_the_old_behaviour_is_evicted(self, monkeypatch):
        tagged = f"[agent:eda]\nsys\n\n{llm.UNTRUSTED_DATA_NOTICE}"
        cache.store("groq", "m", tagged, "user", '{"trunc', 1, 1)

        backend = _Scripted('{"a": 2}')
        monkeypatch.setattr(llm, "get_backend", lambda: backend)
        assert llm.ask_json("eda", "sys", "user", required_keys=("a",)) == {"a": 2}
        assert cache.lookup("groq", "m", tagged, "user") is None


class TestTrimmingAlwaysMakesProgress:
    """The last-resort trimmer must remove something on every pass.

    A nested dict reduced to prose keys and `_omitted` was descended into,
    found to have nothing droppable, and abandoned; the outer loop re-selected
    it on every remaining pass and returned the payload still over budget.
    """

    def test_prose_only_findings_do_not_stall_the_trimmer(self):
        payload = {
            f"stage{i}": {"findings": {"summary": "s" * 900,
                                       "detail": ["x" * 50] * 30},
                          "narrative": "n" * 900}
            for i in range(14)
        }
        result, trimmed = bound_payload(payload, 6_000)
        assert trimmed
        assert len(json.dumps(result)) <= 6_000
        # Every stage is still represented by its narrative.
        assert all("narrative" in result[f"stage{i}"] for i in range(14))


# --------------------------------------------------------------------------- #
# Model output of the wrong shape
# --------------------------------------------------------------------------- #

def _titanic_like(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sex = rng.choice(["male", "female"], n, p=[0.65, 0.35])
    return pd.DataFrame({
        "Survived": (rng.random(n) < np.where(sex == "female", 0.74, 0.19)).astype(int),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": sex,
        "Age": rng.normal(29, 14, n).round(1),
        "Fare": rng.gamma(2, 20, n).round(2),
    })


class TestModelValuesOfTheWrongShape:
    """Names returned as objects or lists must be read, not crash the stage.

    `c in real` raised `TypeError: unhashable type` for a dict or list element,
    which failed the mandatory profiling stage and everything after it.
    """

    @pytest.fixture
    def mock(self):
        backend = llm.get_backend("mock")
        original = dict(backend._handlers)
        yield backend
        backend._handlers = original

    def test_profiling_reads_names_given_as_objects_and_lists(self, mock):
        from agents.profiling_agent import ProfilingAgent
        from state import WorkflowState

        mock.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x",
            "usable_features": [{"name": "Sex"}, {"column": "Age"}, ["Fare"], 7],
            "excluded_columns": [], "target_column": ["Survived"],
            "task_type": ["classification"], "target_reasoning": "x",
            "concerns": [], "narrative": "x"})
        output = ProfilingAgent().run(WorkflowState(df=_titanic_like(), dataset_name="t"))

        assert output.status == "ok"
        assert output.findings["target_column"] == "Survived"
        assert output.findings["task_type"] == "classification"
        assert {"Sex", "Age", "Fare"} <= set(output.findings["usable_features"])

    def test_a_stated_absence_of_target_is_respected(self, mock):
        """"none" as a string means no target, not a column to fall back from."""
        from agents.profiling_agent import ProfilingAgent
        from state import WorkflowState

        mock.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x", "usable_features": ["Age"],
            "excluded_columns": [], "target_column": "none", "task_type": "none",
            "target_reasoning": "x", "concerns": [], "narrative": "x"})
        output = ProfilingAgent().run(WorkflowState(df=_titanic_like(), dataset_name="t"))

        assert output.findings["target_column"] is None
        assert output.findings["task_type"] == "none"

    def test_modelling_reads_features_given_as_objects(self, mock):
        from agents.modelling_agent import ModellingAgent
        from agents.profiling_agent import ProfilingAgent
        from state import WorkflowState

        mock.register("modelling", lambda _: {
            "recommended_model": {"name": "Random Forest"},
            "recommendation_reason": "x", "expected_performance": "x",
            "evaluation_metrics": [], "improvements": [], "caveats": [],
            "important_features": [{"feature": "Sex"}, ["Ghost"]],
            "signal_assessment": ["Moderate"], "narrative": "x"})
        state = WorkflowState(df=_titanic_like(), dataset_name="t")
        state.record(ProfilingAgent().run(state))
        output = ModellingAgent().run(state)

        assert output.status != "failed", output.error
        assert output.findings["signal_assessment"] == "moderate"
        assert "Sex" in output.findings["important_features"]
        assert "Ghost" not in output.findings["important_features"]


# --------------------------------------------------------------------------- #
# Engine selection
# --------------------------------------------------------------------------- #

class TestEngineChoiceRespectsTheFileFormat:
    """Spark is chosen only for files its reader can parse.

    Size alone decided, and `load` parsed everything as CSV: a large Excel or
    JSON file crossed the threshold and was read by Spark as comma-separated
    text.
    """

    @pytest.fixture(autouse=True)
    def spark_everywhere(self, monkeypatch):
        from tools import spark_engine

        monkeypatch.setattr(spark_engine, "spark_available", lambda: (True, "available"))
        monkeypatch.setattr(CONFIG, "spark_size_threshold_mb", 0.0)
        monkeypatch.setattr(CONFIG, "spark_enabled", True)

    @pytest.mark.parametrize("name", ["big.xlsx", "big.xls", "big.json"])
    def test_formats_spark_cannot_read_stay_in_memory(self, tmp_path, name):
        from tools import spark_engine

        path = tmp_path / name
        path.write_bytes(b"\x00" * 64)
        decision = spark_engine.choose_engine(str(path))
        assert decision["engine"] == "pandas"
        assert "CSV and Parquet" in decision["reason"]

    @pytest.mark.parametrize("name", ["big.csv", "big.parquet", "big.txt"])
    def test_formats_spark_reads_are_distributed(self, tmp_path, name):
        from tools import spark_engine

        path = tmp_path / name
        path.write_bytes(b"a,b\n1,2\n")
        assert spark_engine.choose_engine(str(path))["engine"] == "spark"

    def test_load_refuses_a_format_it_would_misparse(self, tmp_path):
        from tools import spark_engine

        with pytest.raises(ValueError, match="CSV and Parquet"):
            spark_engine.load(str(tmp_path / "data.xlsx"))


# --------------------------------------------------------------------------- #
# The verification layer: every mechanism registered, called, and switchable
# --------------------------------------------------------------------------- #

import verification as V  # noqa: E402

#: One input per mechanism that it must flag. Keyed by registered name, so a
#: mechanism added without an example fails `test_every_mechanism_has_an_example`
#: -- which is what forces it through the ablation checks below.
MECHANISM_EXAMPLES = {
    "unsupported_numbers": (V.unsupported_numbers, ("Accuracy was 0.99.", {"0.5"})),
    "causal_language": (V.causal_language, ("Age causes survival.",)),
    "hallucinated_columns": (V.hallucinated_columns, (["Ghost"], ["Age"])),
    "overstated_segmentation": (V.overstated_segmentation,
                                ("The rows form three distinct groups.",)),
    "overstated_feature_benefit": (V.overstated_feature_benefit,
                                   ("The derived columns improved accuracy.",)),
    "suppressed_negative_result": (V.suppressed_negative_result,
                                   ("Accuracy was 0.52.", False)),
    "unsupported_non_linear_signal": (V.unsupported_non_linear_signal, (["a"], {"b"})),
    "invented_features": (V.invented_features, (["x_log"], {"y_log"})),
    "overridden_signal_claim": (V.overridden_signal_claim, ("weak", False, "Random Forest")),
    "unsupported_importance_claim": (V.unsupported_importance_claim, (["a"], {"b"})),
    "overstated_effect": (V.overstated_effect, ({"a": 0.01},)),
    "refused_cleaning_action": (V.refused_cleaning_action,
                                ("delete_everything", {"drop_column"})),
    "failed_chart_request": (V.failed_chart_request, ("Unknown chart kind",)),
}


class TestTheAblationSwitchesOffEveryMechanism:
    """`--no-verification` must silence every registered mechanism.

    `invented_features` and four others had no guard, and the modelling
    override, the quality refusals and the validator's modelling checks ran
    inline where the switch never reached them. The ablation therefore
    measured a framework that was still largely verifying itself.
    """

    def test_every_mechanism_has_an_example(self):
        assert set(MECHANISM_EXAMPLES) == {m.name for m in V.MECHANISMS}

    @pytest.mark.parametrize("name", sorted(MECHANISM_EXAMPLES))
    def test_the_mechanism_fires_when_enabled(self, name, monkeypatch):
        monkeypatch.setattr(CONFIG, "verification_enabled", True)
        check, args = MECHANISM_EXAMPLES[name]
        assert check(*args), f"{name} did not flag its example"

    @pytest.mark.parametrize("name", sorted(MECHANISM_EXAMPLES))
    def test_the_mechanism_is_silent_when_disabled(self, name, monkeypatch):
        monkeypatch.setattr(CONFIG, "verification_enabled", False)
        check, args = MECHANISM_EXAMPLES[name]
        assert not check(*args), f"{name} ignores the ablation switch"

    def test_an_audit_measures_despite_the_switch_and_counts_nothing(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "verification_enabled", False)
        V.reset_firings()
        with V.auditing():
            assert V.causal_language("Age causes survival.")
        assert V.firings()["causal_language"] == 0

    def test_every_registered_mechanism_is_called_by_an_agent(self):
        """A registered check nothing calls is decoration, not verification."""
        from pathlib import Path

        agents = Path(__file__).resolve().parents[1] / "agents"
        source = "\n".join(p.read_text(encoding="utf-8") for p in agents.glob("*.py"))
        uncalled = [m.name for m in V.MECHANISMS if f"{m.name}(" not in source]
        assert not uncalled, f"registered but never called: {uncalled}"


class TestFiringsAreCountedByTheMechanisms:
    def test_a_run_records_how_often_each_mechanism_fired(self, monkeypatch):
        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        mock = llm.get_backend("mock")
        original = dict(mock._handlers)
        try:
            mock.register("report", lambda _: {
                "title": "R", "executive_summary": "Accuracy reached 97.3%.",
                "sections": [], "key_takeaways": [], "limitations": [],
                "recommended_next_steps": []})
            state = run_workflow(_titanic_like(), "t.csv")
        finally:
            mock._handlers = original

        assert set(state.verification) == {m.name for m in V.MECHANISMS}
        assert state.verification["unsupported_numbers"] >= 1
        assert state.summary()["verification"] == state.verification


class TestNumbersInText:
    """A hyphen is not a minus sign."""

    @pytest.mark.parametrize("text,expected", [
        ("ages 30-40", [30.0, 40.0]),
        ("on 2024-01-15", [2024.0, 1.0, 15.0]),
        ("a correlation of -0.45", [-0.45]),
        ("r=-0.45", [-0.45]),
        ("a correlation of −0.45", [-0.45]),
        ("between 0.5-0.7", [0.5, 0.7]),
    ])
    def test_values_are_read_as_written(self, text, expected):
        assert [value for _, value, _ in V.numbers_in(text)] == expected

    def test_a_range_is_not_reported_as_an_invented_negative(self):
        allowed: set[str] = set()
        V.collect_numbers({"min": 30, "max": 40}, allowed)
        assert V.unsupported_numbers("Most passengers were aged 30-40.", allowed) == []


# --------------------------------------------------------------------------- #
# The report's permitted numbers
# --------------------------------------------------------------------------- #

class TestOnlyComputedNumbersArePermitted:
    """A report may cite what a tool computed, not what a model wrote.

    The permitted set was collected from the prompt material, which includes
    every stage narrative and the validator objections. An invented 0.99
    quoted back in an objection became permitted in the next draft.
    """

    @pytest.fixture
    def mock(self):
        backend = llm.get_backend("mock")
        original = dict(backend._handlers)
        yield backend
        backend._handlers = original

    @staticmethod
    def _state(df=None):
        from agents.profiling_agent import ProfilingAgent
        from state import WorkflowState

        state = WorkflowState(df=_titanic_like() if df is None else df,
                              dataset_name="t.csv")
        state.record(ProfilingAgent().run(state))
        return state

    @staticmethod
    def _report(summary):
        return lambda _: {
            "title": "R", "executive_summary": summary, "sections": [],
            "key_takeaways": [], "limitations": [], "recommended_next_steps": []}

    def test_a_figure_quoted_in_an_objection_is_not_laundered(self, mock):
        from agents.report_agent import ReportAgent
        from state import AgentOutput

        state = self._state()
        state.record(AgentOutput(agent="validation", findings={
            "verdict": "unreliable", "issues": [{
                "severity": "critical", "location": "report",
                "problem": "The report claims an accuracy of 0.991 that nothing computed.",
                "suggested_fix": "Remove 0.991."}]}))
        mock.register("report", self._report("The model reached 0.991 accuracy."))

        output = ReportAgent().run(state)
        assert "0.991" in output.findings["unsupported_numbers"]

    def test_a_figure_only_an_upstream_narrative_states_is_not_permitted(self, mock):
        from agents.report_agent import ReportAgent

        state = self._state()
        state.outputs["profiling"].narrative = "Roughly 73.9% of the rows are complete."
        mock.register("report", self._report("About 73.9% of the rows are complete."))

        output = ReportAgent().run(state)
        assert "73.9" in output.findings["unsupported_numbers"]

    def test_counts_computed_by_the_quality_stage_remain_permitted(self, mock):
        from agents.quality_agent import QualityAgent
        from agents.report_agent import ReportAgent

        df = _titanic_like()
        state = self._state(pd.concat([df, df.head(37)], ignore_index=True))
        state.record(QualityAgent().run(state))
        after = state.findings("quality")["rows_after"]
        mock.register("report", self._report(f"After cleaning, {after} rows remained."))

        assert ReportAgent().run(state).findings["unsupported_numbers"] == []


# --------------------------------------------------------------------------- #
# A null modelling result
# --------------------------------------------------------------------------- #

class TestANullResultIsNeitherRecommendedNorSaved:
    """When nothing beats the baseline, no model is recommended or saved.

    A model assessed as "weak" stayed recommended, because the override only
    fired for "strong" or "moderate"; the validator objection to it was marked
    revisable, so the workflow re-ran the report twice to no effect. And the
    winner on pure noise was saved, so `predict.py --latest` scored new data
    with a model the report had rejected.
    """

    @pytest.fixture
    def noise_state(self):
        from agents.profiling_agent import ProfilingAgent
        from state import WorkflowState

        backend = llm.get_backend("mock")
        original = dict(backend._handlers)
        backend.register("profiling", lambda _: {
            "dataset_description": "x", "domain": "x",
            "usable_features": ["a", "b", "c"], "excluded_columns": [],
            "target_column": "target", "task_type": "classification",
            "target_reasoning": "x", "concerns": [], "narrative": "x"})
        backend.register("modelling", lambda _: {
            "recommended_model": "Random Forest", "recommendation_reason": "x",
            "expected_performance": "x", "evaluation_metrics": [],
            "important_features": [], "signal_assessment": "weak",
            "improvements": [], "caveats": [], "narrative": "x"})
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400),
                           "c": rng.choice(list("xyz"), 400),
                           "target": rng.integers(0, 2, 400)})
        state = WorkflowState(df=df, dataset_name="noise.csv")
        state.record(ProfilingAgent().run(state))
        yield state
        backend._handlers = original

    def test_a_weak_recommendation_is_withdrawn(self, noise_state):
        from agents.modelling_agent import ModellingAgent

        output = ModellingAgent().run(noise_state)
        assert output.findings["beats_baseline"] is False
        assert output.findings["recommended_model"] is None
        assert any("recommendation was withdrawn" in w for w in output.warnings)

    def test_no_model_is_saved(self, noise_state):
        from agents.modelling_agent import ModellingAgent

        output = ModellingAgent().run(noise_state)
        assert output.findings["saved_model"] is None
        assert "No model was saved" in (output.findings["saved_model_note"] or "")

    def test_the_validator_does_not_loop_on_a_modelling_finding(self, noise_state):
        from agents.validation_agent import ValidationAgent
        from state import AgentOutput

        noise_state.record(AgentOutput(agent="modelling", findings={
            "beats_baseline": False, "recommended_model": "Random Forest",
            "signal_assessment": "weak"}))
        issues = ValidationAgent()._mechanical_checks(noise_state)
        modelling_issues = [i for i in issues if i["location"] == "modelling"]
        assert modelling_issues
        assert not any(i["revisable"] for i in modelling_issues)

    def test_under_the_ablation_the_recommendation_stands_and_the_audit_sees_it(
            self, noise_state, monkeypatch):
        from agents.modelling_agent import ModellingAgent

        monkeypatch.setattr(CONFIG, "verification_enabled", False)
        noise_state.record(ModellingAgent().run(noise_state))
        assert noise_state.findings("modelling")["recommended_model"] == "Random Forest"
        assert V.audit(noise_state)["overridden_signal_claim"] == 1


class TestQualityRefusalsFollowTheSwitch:
    def test_the_screen_is_off_under_the_ablation(self, monkeypatch):
        from agents.quality_agent import QualityAgent
        from state import WorkflowState

        state = WorkflowState(df=_titanic_like(), dataset_name="t.csv")
        state.target_column = "Survived"
        request = [{"action": "drop_column", "column": "Survived"}]

        monkeypatch.setattr(CONFIG, "verification_enabled", True)
        approved, refused = QualityAgent()._screen(request, state.data, state)
        assert not approved and "target column" in refused[0]

        monkeypatch.setattr(CONFIG, "verification_enabled", False)
        approved, refused = QualityAgent()._screen(request, state.data, state)
        assert approved and not refused


# --------------------------------------------------------------------------- #
# Imputation happens inside the folds
# --------------------------------------------------------------------------- #

def _messy_titanic() -> pd.DataFrame:
    """Missing ages, fares stored as text, and sexes in inconsistent case."""
    raw = _titanic_like()
    raw.loc[::5, "Age"] = np.nan
    # Heavily right-skewed, as real fares are, so a log feature is derived.
    fares = np.random.default_rng(3).lognormal(3, 1, len(raw)).round(2)
    raw["Fare"] = fares.astype(str)
    raw.loc[::7, "Sex"] = raw.loc[::7, "Sex"].str.upper() + " "
    return raw


class TestImputationHappensInsideTheFolds:
    """An authorised imputation is performed by the model pipeline, per fold.

    The quality stage filled columns with a median computed over every row
    before modelling cross-validated on them, so held-out rows informed their
    own fill value; and the feature stage, running afterwards, could never
    build a missingness indicator for a column that had been filled.
    """

    @pytest.fixture
    def mock(self):
        backend = llm.get_backend("mock")
        original = dict(backend._handlers)
        yield backend
        backend._handlers = original

    def _cleaned(self, mock):
        from agents.profiling_agent import ProfilingAgent
        from agents.quality_agent import QualityAgent
        from state import WorkflowState

        mock.register("quality", lambda _: {
            "actions": [{"action": "impute_median", "column": "Age", "reason": "x"}],
            "declined": [], "limitations": [], "narrative": "x"})
        df = _titanic_like()
        df.loc[:59, "Age"] = np.nan
        state = WorkflowState(df=df, dataset_name="t.csv")
        state.record(ProfilingAgent().run(state))
        output = QualityAgent().run(state)
        return state, output

    def test_the_working_frame_is_not_filled(self, mock):
        state, output = self._cleaned(mock)
        assert state.data["Age"].isna().sum() == 60
        assert [a["column"] for a in output.findings["actions_deferred"]] == ["Age"]
        assert not any(a["action"] == "impute_median"
                       for a in output.findings["actions_applied"])

    def test_the_missingness_indicator_can_then_be_built(self, mock):
        from tools.features import engineer_features

        state, _ = self._cleaned(mock)
        built = {a["feature"] for a in
                 engineer_features(state.data, "Survived", "classification")["applied"]}
        assert "Age_was_missing" in built


# --------------------------------------------------------------------------- #
# The saved model takes raw rows
# --------------------------------------------------------------------------- #

class TestTheSavedModelTakesRawRows:
    """A saved model must score the original file, not the working frame.

    Its signature listed derived columns such as `Fare_log` that no raw file
    contains, so `predict.py` on the original data failed; and a repair such
    as a text-to-number conversion meant the pipeline received strings where
    it had been fitted on floats.
    """

    @staticmethod
    def _fitted():
        from tools.features import engineer_features
        from tools.modelling import fit_final_model
        from tools.quality import replay_cleaning

        raw = _messy_titanic()
        cleaning = [{"stage": "quality", "action": "to_numeric", "column": "Fare"},
                    {"stage": "quality", "action": "normalise_categories", "column": "Sex"}]
        engineered = engineer_features(replay_cleaning(raw, cleaning),
                                       "Survived", "classification")
        working = engineered["frame"]
        features = [c for c in working.columns if c != "Survived"]
        pipeline, signature = fit_final_model(
            working, "Survived", features, "classification", "Logistic Regression",
            preparation=cleaning + engineered["recipe"])
        return raw, working, features, pipeline, signature

    def test_the_signature_asks_for_raw_columns_only(self):
        raw, _, _, _, signature = self._fitted()
        derived = {d["feature"] for d in signature["derived_features"]}

        assert {"Age_was_missing", "Fare_log"} <= derived
        assert set(signature["expects_columns"]) <= set(raw.columns)
        assert not derived & set(signature["expects_columns"])

    def test_raw_rows_score_exactly_as_the_prepared_rows_did(self):
        from tools.modelling import fit_final_model

        raw, working, features, pipeline, signature = self._fitted()
        # The same model fitted directly on the prepared frame, with no
        # preparation step: the reference the saved pipeline must reproduce.
        plain, plain_signature = fit_final_model(
            working, "Survived", features, "classification", "Logistic Regression")

        from_raw = pipeline.predict_proba(raw[signature["expects_columns"]])
        reference = plain.predict_proba(working[plain_signature["expects_columns"]])
        np.testing.assert_allclose(from_raw, reference)

    def test_predict_scores_the_original_file(self, tmp_path, monkeypatch):
        import sys

        import joblib

        import predict

        raw, _, _, pipeline, signature = self._fitted()
        model = tmp_path / "titanic_model.joblib"
        joblib.dump(pipeline, model)
        (tmp_path / "titanic_model_signature.json").write_text(
            json.dumps(signature, default=str), encoding="utf-8")
        rows = tmp_path / "new_rows.csv"
        raw.drop(columns="Survived").to_csv(rows, index=False)
        scored = tmp_path / "scored.csv"

        monkeypatch.setattr(sys, "argv", ["predict.py", str(model), str(rows),
                                          "-o", str(scored)])
        assert predict.main() == 0
        assert len(pd.read_csv(scored)) == len(raw)

    def test_a_workflow_saves_a_model_that_scores_the_original_rows(self, monkeypatch):
        import joblib

        from orchestrator.graph import run_workflow

        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        raw = _messy_titanic()
        state = run_workflow(raw, "messy.csv")
        saved = state.findings("modelling").get("saved_model")
        assert saved, state.findings("modelling").get("saved_model_note")

        pipeline = joblib.load(saved["path"])
        columns = saved["expects_columns"]
        assert set(columns) <= set(raw.columns)
        # Rows the analysis kept, scored from the original file and from the
        # prepared frame: identical only if every repair and derived column
        # is rebuilt by the saved pipeline itself.
        index = state.data.index
        np.testing.assert_allclose(
            pipeline.predict_proba(raw.loc[index, columns]),
            pipeline.predict_proba(state.data.loc[index, columns]))


class TestDuplicateHeadersAreMadeUnique:
    @pytest.mark.parametrize("header,expected", [
        (["x", "x", "y"], ["x", "x.1", "y"]),
        (["a", "a", "a.1"], ["a", "a.2", "a.1"]),
        (["a", "a", "a", "a.1", "a.2"], ["a", "a.3", "a.4", "a.1", "a.2"]),
    ])
    def test_renamed_columns_never_collide(self, header, expected):
        from orchestrator.graph import _ensure_unique_columns

        frame = pd.DataFrame([list(range(len(header)))], columns=header)
        out, renamed = _ensure_unique_columns(frame)
        assert list(out.columns) == expected
        assert out.columns.is_unique and renamed


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #

class TestTheInterface:
    """Results belong to their dataset, and untrusted text is never markup.

    After another dataset was loaded, the previous run's results stayed on the
    page under the new dataset's name. And column names, file names and model
    output were placed into HTML unescaped, so a column called
    "<img src=x onerror=...>" became an element.
    """

    APP = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "app.py")

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        monkeypatch.setattr(CONFIG, "permutation_tests", 0)

    def test_results_from_another_dataset_are_hidden(self):
        from streamlit.testing.v1 import AppTest

        from orchestrator.graph import run_workflow

        app = AppTest.from_file(self.APP, default_timeout=120)
        app.session_state["state"] = run_workflow(_titanic_like().head(120), "titanic.csv")
        app.session_state["state_key"] = "titanic.csv|120x5|0"
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        assert len(app.tabs) == 6          # shown while nothing else is loaded

        app.sidebar.selectbox[0].select("iris.csv").run()
        assert not app.exception, [e.value for e in app.exception]
        assert len(app.tabs) == 0
        assert any("Earlier results are hidden" in m.value for m in app.markdown)

    def test_untrusted_names_are_escaped_in_markup(self):
        from streamlit.testing.v1 import AppTest

        from orchestrator.graph import run_workflow

        hostile = '<img src=x onerror=alert(1)>.csv'
        app = AppTest.from_file(self.APP, default_timeout=120)
        app.session_state["state"] = run_workflow(_titanic_like().head(120), hostile)
        app.run()
        assert not app.exception, [e.value for e in app.exception]

        markup = [m.value for m in app.markdown if m.proto.allow_html]
        assert any("&lt;img src=x" in block for block in markup)
        assert not any("<img src=x" in block for block in markup)


class TestProgressIsReportedAsTheRunAdvances:
    """The interface is told as each stage finishes, and cannot break the run."""

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        monkeypatch.setattr(CONFIG.llm, "backend", "mock")
        monkeypatch.setattr(CONFIG, "permutation_tests", 0)

    def test_every_stage_is_reported_once_the_plan_exists(self):
        from orchestrator.graph import run_workflow

        seen: list[tuple[int, list[str]]] = []
        state = run_workflow(_titanic_like().head(120), "t.csv",
                             on_progress=lambda s: seen.append(
                                 (len(s.plan), list(s.outputs))))

        assert seen and seen[0][0] > 0          # the plan exists before any stage
        assert seen[-1][1] == list(state.outputs)
        # Completed stages only accumulate: the observer sees the run advance.
        counts = [len(outputs) for _, outputs in seen]
        assert counts == sorted(counts)

    def test_a_failing_observer_does_not_fail_the_analysis(self):
        from orchestrator.graph import run_workflow

        def broken(state):
            raise RuntimeError("the interface went away")

        state = run_workflow(_titanic_like().head(120), "t.csv", on_progress=broken)
        assert state.findings("validation").get("verdict")


class TestQuickRun:
    """A demonstration shortcut that cannot silently change the answer."""

    def test_it_switches_off_only_the_two_costliest_options(self, monkeypatch):
        from config import FrameworkConfig, apply_quick_run

        monkeypatch.setattr(CONFIG, "stacking_max_rows", 20_000)
        monkeypatch.setattr(CONFIG, "tune_models", True)
        permutations = CONFIG.permutation_tests

        apply_quick_run(True)
        assert CONFIG.stacking_max_rows == 0 and CONFIG.tune_models is False
        # Fewer permutations would put p = 1/(n+1) above the threshold and
        # make every dataset report no signal.
        assert CONFIG.permutation_tests == permutations
        assert 1 / (CONFIG.permutation_tests + 1) < CONFIG.permutation_alpha

        # Turning it off restores the real defaults, not whatever was last set.
        apply_quick_run(False)
        defaults = FrameworkConfig()
        assert CONFIG.stacking_max_rows == defaults.stacking_max_rows
        assert CONFIG.tune_models == defaults.tune_models


class TestTheReportGetsALargerOutputAllowance:
    """The report was cut off mid-JSON at the shared 2,560-token cap."""

    def test_the_allowance_applies_to_the_report_call_only(self, monkeypatch):
        seen = {}

        class Recorder:
            name, model = "groq", "m"

            def complete(self, system, user):
                seen[system.split("]")[0]] = CONFIG.llm.max_tokens
                return llm.LLMResponse(text='{"a": 1}', backend="groq")

        monkeypatch.setattr(llm, "get_backend", lambda: Recorder())
        monkeypatch.setattr(CONFIG, "llm_cache_enabled", False)
        before = CONFIG.llm.max_tokens
        llm.ask_json("report", "s", "u")
        llm.ask_json("eda", "s", "u")

        assert seen["[agent:report"] == CONFIG.llm.max_tokens_for["report"] > before
        assert seen["[agent:eda"] == before
        assert CONFIG.llm.max_tokens == before
