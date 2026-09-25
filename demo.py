"""A guided demonstration of the framework, for someone seeing it for the first time.

`run.py` analyses one dataset and `app.py` opens an interface, but neither shows
what the framework is actually claiming. This does: it runs the pipeline on a
dataset that contains real signal and on one that is pure noise, and prints the
two verdicts side by side.

That pairing is the whole argument. Reporting high accuracy on real data proves
nothing on its own -- a pipeline that always reports high accuracy would do the
same. Reporting *no signal* on random data is what makes the first number
trustworthy, and only the two together say anything worth saying.

    python demo.py

Runs offline on the mock backend. Takes a few minutes: the permutation
test refits the winning model fifty times against shuffled targets, and
that cost is the point rather than overhead. No API key, no network.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from config import CONFIG, DATASETS_DIR
from orchestrator.graph import run_workflow

LINE = "=" * 74


def _noise(n_rows: int = 500, n_features: int = 8, seed: int = 0) -> pd.DataFrame:
    """Random features and a random target: nothing to find, by construction."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=[f"sensor_{i}" for i in range(n_features)],
    )
    df["outcome"] = rng.integers(0, 2, n_rows)
    return df


def _describe(state, label: str) -> dict:
    """Pull out the few numbers that matter for the comparison."""
    modelling = state.findings("modelling")
    validation = state.findings("validation")
    evaluation = (state.outputs["modelling"].evidence or {}).get("evaluation", {}) \
        if "modelling" in state.outputs else {}

    accuracy = None
    best = modelling.get("best_model")
    for row in evaluation.get("results", []):
        if row.get("model") == best and not row.get("is_baseline"):
            accuracy = row.get("scores", {}).get("accuracy")

    permutation = evaluation.get("permutation_test") or {}
    return {
        "label": label,
        "stages": len(state.completed),
        "model": best or "not run",
        "accuracy": accuracy,
        "primary": modelling.get("best_score"),
        "metric": modelling.get("primary_metric"),
        "claims_signal": modelling.get("beats_baseline"),
        "p_value": permutation.get("p_value"),
        "verdict": validation.get("verdict"),
    }


def _print(result: dict) -> None:
    accuracy = f"{result['accuracy']:.4f}" if result["accuracy"] is not None else "n/a"
    primary = (f"{result['primary']:.4f} ({result['metric']})"
               if result["primary"] is not None else "n/a")
    print(f"  stages run       : {result['stages']}")
    print(f"  winning model    : {result['model']}")
    print(f"  accuracy         : {accuracy}")
    print(f"  primary metric   : {primary}")
    print(f"  permutation p    : {result['p_value']}")
    print(f"  claims signal    : {result['claims_signal']}")
    print(f"  validation       : {result['verdict']}")


def _make_console_unicode_safe() -> None:
    """Stop a non-ASCII character in the report from crashing the CLI.

    Windows consoles default to cp1252, which cannot encode the non-breaking
    hyphens, em-dashes and curly quotes that language models emit routinely. The
    report files are written as UTF-8 and are unaffected, but printing a summary
    of one would raise UnicodeEncodeError *after* the entire analysis had
    succeeded -- six minutes of work appearing to fail at the last line, with
    the output sitting correctly on disk.

    Reconfiguring to UTF-8 fixes it where the console supports it; `errors` is
    set so that a console which does not degrades to a replacement character
    rather than an exception. Losing the shape of a dash is a fair trade for not
    losing the run.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # Not a reconfigurable stream (redirected to a pipe in some
            # environments). Nothing to do, and nothing worth failing over.
            pass


def main() -> int:
    _make_console_unicode_safe()
    CONFIG.llm.backend = "mock"  # offline, deterministic, free

    print(LINE)
    print("  Multi-Agent Data Science Framework - demonstration")
    print(LINE)
    print("\nThe same pipeline is run twice: once on a real medical dataset, and")
    print("once on random numbers. Watch what it refuses to claim.\n")

    started = time.time()

    print(LINE)
    print("  1. REAL DATA - heart disease, 303 patients")
    print(LINE)
    real = pd.read_csv(DATASETS_DIR / "heart_disease.csv")
    real_result = _describe(run_workflow(real, "heart_disease.csv"), "real")
    _print(real_result)

    print()
    print(LINE)
    print("  2. PURE NOISE - random features, random target")
    print(LINE)
    noise_result = _describe(run_workflow(_noise(), "pure_noise.csv"), "noise")
    _print(noise_result)

    print()
    print(LINE)
    print("  WHAT THIS SHOWS")
    print(LINE)

    honest = real_result["claims_signal"] and not noise_result["claims_signal"]
    if honest:
        print("\n  On real data the framework found signal and said so.")
        print("  On random data it found none and said that too.\n")
        print("  The second result is what makes the first one worth believing.")
        print("  A pipeline that reported success on both would be telling you")
        print("  nothing at all -- and most would, because a flexible model can")
        print("  always fit noise if nobody checks whether the fit is real.\n")
        print("  The check is a permutation test: the winning model is refitted")
        print("  against repeatedly shuffled targets to measure what 'no")
        print("  relationship' scores. That is the honest reference.")
    else:
        print("\n  UNEXPECTED: the framework did not separate signal from noise.")
        print(f"  real claims signal : {real_result['claims_signal']}")
        print(f"  noise claims signal: {noise_result['claims_signal']}")
        print("\n  This is a defect. The framework's central claim is that it")
        print("  distinguishes the two.")

    print(f"\n  Completed in {time.time() - started:.0f}s on the offline mock backend.")
    print("  To write a full report for a dataset, run:")
    print("      python run.py datasets/heart_disease.csv")
    print()
    return 0 if honest else 1


if __name__ == "__main__":
    sys.exit(main())
