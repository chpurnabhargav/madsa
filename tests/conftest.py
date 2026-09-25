"""Test configuration: keep the suite fast enough that it actually gets run.

Adding the stacked ensemble and the hyperparameter search took this suite from
about seven minutes to over thirty, and a suite that costs half an hour is one
that stops being run before a commit -- which is worse than a slightly less
thorough suite that runs every time.

The cost compounds rather than adds. Stacking fits three base models inside its
own inner cross-validation, and the permutation test then refits *the winner*
fifty times against shuffled targets. Once stacking starts winning, those fifty
refits each become ten. Disabling stacking in tests therefore removes far more
than stacking's own share.

So the expensive options are off by default here, and tests that exist
specifically to exercise them turn them back on. The framework's real defaults
are untouched: this changes what the tests cost, not what the framework does.
"""
from __future__ import annotations

import pytest

from config import CONFIG


@pytest.fixture(autouse=True)
def fast_defaults(request, monkeypatch):
    """Make model fitting cheap unless a test opts out.

    A test marked `slow` gets the framework's real settings, because the thing
    it is checking is the behaviour those settings produce. Everything else gets
    the cheap configuration, since it is testing orchestration, verification or
    contracts rather than the ensemble itself.
    """
    if request.node.get_closest_marker("slow"):
        yield
        return

    # Zero means no dataset qualifies, so the stacked ensemble is never built.
    monkeypatch.setattr(CONFIG, "stacking_max_rows", 0)

    # The search is the second-largest cost and is verified directly by
    # TestHyperparameterTuning, which marks itself slow.
    monkeypatch.setattr(CONFIG, "tune_models", False)

    yield


@pytest.fixture(autouse=True)
def isolated_model_directory(tmp_path, monkeypatch):
    """Keep models fitted by tests out of the real `models/` folder.

    Every test that ran a workflow on data with signal saved a model there --
    over a hundred files named after test fixtures had accumulated -- and
    `predict.py --latest` picks whichever is newest, so running the suite
    silently changed which model a user's next prediction would use.
    """
    import agents.modelling_agent as modelling_agent

    monkeypatch.setattr(modelling_agent, "MODELS_DIR", tmp_path)


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: exercises the full model set, ensembles and tuning included; "
        "run with `pytest -m slow` or as part of the unmarked full suite",
    )
