"""Central configuration for the MADSA framework.

Everything tunable lives here so experiments in the evaluation harness can vary
one setting at a time without touching agent code.
"""
from __future__ import annotations

import os

import warnings

# Recent Windows builds ship without `wmic`, which joblib runs to count physical
# cores. The probe fails, joblib falls back to the logical count -- entirely
# correct behaviour -- and warns. The warning's message text, however, contains a
# formatted traceback, so every run of this framework prints what looks exactly
# like a crash into the middle of its output while nothing is wrong. On a project
# whose whole point is that a reader can trust what it reports, output that looks
# like a failure and is not is worse than a real error.
#
# Filtering the warning is not enough: joblib also calls `traceback.print_tb`
# unconditionally, straight to stderr, which no warnings filter can reach. The
# only way to stop it is to not take that branch at all. joblib skips the
# physical-core probe entirely when LOKY_MAX_CPU_COUNT is *strictly less than*
# the logical core count, so the value matters -- setting it to the core count,
# as this previously did, falls through to the probe and prints the traceback
# anyway.
#
# Reserving one core is the right value independently of that: this runs on a
# machine someone is using, and a sweep that pins every core makes it unusable.
#
# This must run before scikit-learn is imported anywhere, and config is imported
# first by every module in the project.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 4) - 1)))
warnings.filterwarnings(
    "ignore", message="Could not find the number of physical cores",
    category=UserWarning)

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATASETS_DIR = ROOT / "datasets"
REPORTS_DIR = ROOT / "reports"
RUNS_DIR = ROOT / "runs"
#: Fitted pipelines, saved so the model a report recommends can actually be
#: used. Without this the framework fits, scores, reports and then discards the
#: model, leaving a reader told that something predicts well and unable to
#: obtain it.
MODELS_DIR = ROOT / "models"

for _d in (DATASETS_DIR, REPORTS_DIR, RUNS_DIR, MODELS_DIR):
    _d.mkdir(exist_ok=True)


@dataclass
class LLMConfig:
    """Model settings shared by every agent.

    A single model is used throughout so that token cost reported in the
    evaluation is attributable to orchestration behaviour rather than to a mix
    of differently priced models.
    """

    # Which reasoning engine the agents use:
    #   "mock"      - deterministic canned decisions, free, used while developing
    #   "anthropic" - Claude via the Anthropic API, used for reported results
    #   "ollama"    - a local open-weight model, free and reproducible offline
    backend: str = os.getenv("MADSA_BACKEND", "mock")

    model: str = "claude-sonnet-5"
    # Providers count the reserved output allowance against a per-request or
    # per-minute token limit, not just the tokens actually generated. A 4096
    # reservation therefore consumed half of an 8,000-token budget before the
    # prompt was even counted, and requests were refused as too large while the
    # real usage was far smaller. Observed agent outputs peak near 2,100 tokens.
    max_tokens: int = 2560
    temperature: float = 0.0  # deterministic: the paper needs reproducible runs
    api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        return self.api_key


@dataclass
class FrameworkConfig:
    """Behavioural switches for the orchestrator."""

    llm: LLMConfig = field(default_factory=LLMConfig)

    # Adaptive planning: when False the orchestrator runs a fixed pipeline.
    # Kept as a switch so the evaluation can report on planning behaviour
    # without turning the project into a comparative study.
    adaptive_planning: bool = True

    # How many times the validator may send an agent back for a redo.
    max_validation_retries: int = 2

    # Rows sampled for expensive statistics on large datasets.
    profile_sample_rows: int = 100_000

    # A column with more than this fraction missing is flagged as unusable.
    unusable_missing_fraction: float = 0.6

    # Cardinality at or below this makes a numeric column a candidate target
    # for classification rather than regression.
    max_classification_classes: int = 20

    # Permutations used to test whether the winning model learned real signal.
    # More permutations give a finer p-value at proportionally more compute; 50
    # is the smallest count that can resolve the corrected threshold below.
    permutation_tests: int = 50

    # Character budget for the prompts that receive every earlier stage's
    # findings (the report and validation agents). Roughly four characters per
    # token, so this targets about 5,000 tokens and leaves headroom under the
    # 8,000-token per-request limit of the free tiers the project runs on.
    # Prompts above the budget are trimmed rather than sent and refused.
    report_prompt_char_budget: int = 12_000

    # A stacked ensemble refits every base model inside its own inner
    # cross-validation, so its cost is several times that of any single
    # candidate. Above this many rows that cost outweighs the fraction of a
    # percentage point it typically adds.
    stacking_max_rows: int = 20_000

    # --- Hyperparameter tuning -------------------------------------------
    # The candidate models are compared untuned, which is the honest way to
    # establish what the data supports. The winner is then searched, and the
    # gain is measured on data the search never saw, so "tuning helped" is a
    # result rather than an assumption.
    tune_models: bool = True
    tuning_iterations: int = 15

    # Above this many rows the search costs more than it returns, and the run
    # reports that it was skipped instead of silently taking a long time.
    tuning_max_rows: int = 30_000

    # Replay identical model requests from disk instead of re-sending them.
    # Sound because temperature is fixed at 0.0: the provider is being asked for
    # a deterministic answer, so a stored one is the answer it would give. This
    # is what makes iteration affordable on a rate-limited free tier, and what
    # lets the run reported in the paper be replayed exactly, later, without
    # depending on the provider still serving that model version.
    llm_cache_enabled: bool = os.getenv("MADSA_CACHE", "1") != "0"

    # --- Distributed execution -------------------------------------------
    # The framework chooses between in-memory (pandas) and distributed (Spark)
    # execution per dataset. Spark is not the faster engine below these
    # thresholds: JVM startup and shuffle costs exceed the work being done, so
    # switching early would make the framework slower while looking scalable.
    spark_enabled: bool = os.getenv("MADSA_SPARK", "1") != "0"
    spark_size_threshold_mb: float = 50.0
    spark_row_threshold: int = 500_000
    spark_master: str = os.getenv("MADSA_SPARK_MASTER", "local[*]")
    spark_shuffle_partitions: int = 8  # cluster default of 200 is wrong on one machine
    spark_driver_memory: str = "2g"

    # Rows drawn back into memory when a distributed run reaches a stage that
    # cannot be distributed (charting, scikit-learn modelling). The driver never
    # holds more than this many rows.
    #
    # The binding constraint is not memory but the permutation test, which
    # refits the winning model 50 times: at 200,000 rows that alone runs for
    # hours. 50,000 rows keeps a cross-validated fit to seconds while leaving
    # the sample far larger than any dataset the in-memory engine would handle
    # differently, so the statistics it produces are not the weaker ones.
    spark_collect_rows: int = 50_000

    # Significance threshold for that test. It is stricter than the conventional
    # 0.05 because the model being tested was chosen as the best of several
    # candidates, and selecting a winner before testing it inflates significance.
    permutation_alpha: float = 0.02

    #: Switches every verification mechanism off. Only an ablation should
    #: ever set this: it exists so the framework can be measured against itself
    #: with one variable changed, which is the only controlled comparison
    #: available to us. The surveyed papers evaluate on their own benchmarks
    #: with their own metrics -- AutoML-Agent reports SR, NPS and CS over seven
    #: tasks -- so their figures cannot be placed beside ours. Running our own
    #: system with the layer disabled can.
    verification_enabled: bool = True

    #: Majority-to-minority ratio above which a target counts as imbalanced.
    #: Read by the EDA tool, which reports it, and by the modeller, which acts
    #: on it. It was written out in only one of those places, so the framework
    #: described an imbalance it then did nothing about.
    imbalance_threshold: float = 3.0


CONFIG = FrameworkConfig()
