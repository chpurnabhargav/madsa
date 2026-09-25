"""Health check: tells you what is broken and how to fix it, without guesswork.

Run this first whenever something misbehaves, before reading a traceback:

    python doctor.py

Every check reports one of three things. **OK** means verified working, not
merely present. **WARN** means a capability is unavailable but the framework
degrades around it -- the analysis still runs, with less in it. **FAIL** means
something the framework needs is missing, and the run will not work until it is
fixed.

Each failure carries the exact command that fixes it. The intent is that this
file answers "why is it not working?" on its own, so the answer does not have to
come from somewhere else.

Exit code 0 means nothing failed, so this can gate a script or a CI step.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
import time
from pathlib import Path

OK, WARN, FAIL = "OK", "WARN", "FAIL"
_SYMBOL = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}

results: list[tuple[str, str, str, str]] = []


def check(name: str, status: str, detail: str, fix: str = "") -> None:
    results.append((name, status, detail, fix))
    print(f"  {_SYMBOL[status]} {name:<34} {detail}")
    if fix and status != OK:
        print(f"         fix: {fix}")


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------------- #
# 1. Interpreter and required packages
# --------------------------------------------------------------------------- #

def check_python() -> None:
    section("1. Python")
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version >= (3, 10):
        check("Python version", OK, text)
    else:
        check("Python version", FAIL, f"{text} is too old",
              "Install Python 3.10 or newer; the code uses `X | None` type syntax.")


REQUIRED = {
    "pandas": "data handling",
    "numpy": "numerics",
    "sklearn": "models, PCA, clustering",
    "scipy": "statistical tests",
    "statsmodels": "variance inflation factors",
    "plotly": "charts",
    "langgraph": "agent orchestration",
    "dotenv": "reading .env",
}

OPTIONAL = {
    "pyspark": ("distributed execution on large files",
                "pip install pyspark   (also needs Java; without it, large files "
                "still run in memory)"),
    "streamlit": ("the web interface", "pip install streamlit"),
    "kaleido": ("embedding chart images in reports",
                "pip install kaleido   (without it, reports reference charts "
                "without pictures)"),
    "pytest": ("running the test suite", "pip install pytest"),
}


def check_packages() -> None:
    section("2. Packages")
    for module, purpose in REQUIRED.items():
        try:
            importlib.import_module(module)
            check(module, OK, purpose)
        except ImportError:
            check(module, FAIL, f"missing - needed for {purpose}",
                  "pip install -r requirements.txt")

    for module, (purpose, fix) in OPTIONAL.items():
        try:
            importlib.import_module(module)
            check(module, OK, f"{purpose} (optional)")
        except ImportError:
            check(module, WARN, f"missing - {purpose} unavailable", fix)


# --------------------------------------------------------------------------- #
# 2. Distributed execution
# --------------------------------------------------------------------------- #

def check_spark() -> None:
    section("3. Distributed execution (optional)")
    java = os.environ.get("JAVA_HOME") or shutil.which("java")
    if not java:
        check("Java runtime", WARN, "not found on PATH",
              "Install a JDK (17 or 21) if you want Spark. Without it, files "
              "above the size threshold still run in memory.")
        return
    check("Java runtime", OK, "found")

    try:
        from tools import spark_engine
        available, reason = spark_engine.spark_available()
        if available:
            check("Spark", OK, "available for files above the size threshold")
        else:
            check("Spark", WARN, reason,
                  "Not required. The framework falls back to pandas.")
    except Exception as exc:  # pragma: no cover - environment dependent
        check("Spark", WARN, f"could not be checked: {type(exc).__name__}",
              "Not required; the framework falls back to pandas.")


# --------------------------------------------------------------------------- #
# 3. Language-model access
# --------------------------------------------------------------------------- #

PROVIDERS = {
    "groq": ("GROQ_API_KEY", "gsk_", "console.groq.com - free tier"),
    "gemini": ("GEMINI_API_KEY", "", "aistudio.google.com/apikey - free tier"),
    "anthropic": ("ANTHROPIC_API_KEY", "sk-ant-", "console.anthropic.com - paid"),
    "grok": ("GROK_API_KEY", "xai-", "console.x.ai - paid"),
}


def check_backends() -> None:
    section("4. Language-model backends")
    check("mock backend", OK, "always available - runs the whole pipeline offline")

    configured = []
    for name, (variable, prefix, where) in PROVIDERS.items():
        value = os.getenv(variable, "")
        if not value or value.endswith("..."):
            check(name, WARN, "no key set",
                  f"Optional. Add {variable} to .env - {where}")
            continue
        if prefix and not value.startswith(prefix):
            # The commonest real mistake: Groq and Grok are different companies
            # with near-identical names, and their keys are easy to swap.
            check(name, FAIL, f"key does not start with '{prefix}'",
                  f"This looks like another provider's key. {variable} should "
                  f"begin with '{prefix}'. Groq (gsk_) and Grok (xai-) are "
                  f"different services.")
            continue
        check(name, OK, f"key present ({value[:6]}...)")
        configured.append(name)

    if not configured:
        check("real-model runs", WARN, "no provider configured",
              "Optional. Everything except written narrative works on the mock "
              "backend. For a real model, a free Groq or Gemini key is enough.")
    elif len(configured) == 1:
        check("failover", WARN, f"only {configured[0]} configured",
              "Optional but worth doing. Free tiers rate-limit hard, and the "
              "framework then spends most of its wall-clock waiting for the "
              "allowance to replenish - measured at 450s on stages whose "
              "computation takes under a second. A second free key from a "
              "different provider lets '--backend auto' move on instead of "
              "waiting.")
    else:
        check("failover", OK,
              f"{len(configured)} providers ({', '.join(configured)})",
              "Run with '--backend auto' to use them as one budget.")


# --------------------------------------------------------------------------- #
# 4. Files and directories
# --------------------------------------------------------------------------- #

def check_files() -> None:
    section("5. Files and directories")
    root = Path(__file__).resolve().parent

    datasets = sorted((root / "datasets").glob("*.csv"))
    if datasets:
        check("datasets", OK, f"{len(datasets)} CSV file(s) available")
    else:
        check("datasets", FAIL, "no datasets found",
              "The datasets/ directory is empty. Restore it, or pass your own "
              "CSV path to run.py.")

    for name in ("reports", "runs"):
        directory = root / name
        try:
            directory.mkdir(exist_ok=True)
            probe = directory / ".write_probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            check(f"{name}/ writable", OK, "yes")
        except OSError as exc:
            check(f"{name}/ writable", FAIL, str(exc)[:60],
                  f"The framework cannot save output. Check permissions on "
                  f"{directory}.")

    env = root / ".env"
    example = root / ".env.example"
    if env.exists():
        check(".env", OK, "present (gitignored)")
    elif example.exists():
        check(".env", WARN, "not created",
              "Only needed for real-model runs: copy .env.example to .env.")

    # A key left in the committed template is the one mistake here with a real
    # consequence, so it is checked explicitly rather than assumed absent.
    if example.exists():
        text = example.read_text(encoding="utf-8")
        leaked = [line.split("=")[0] for line in text.splitlines()
                  if "API_KEY=" in line and len(line.split("=", 1)[1].strip()) > 8
                  and not line.strip().endswith("...")]
        if leaked:
            check("keys in template", FAIL, f"real key(s) in .env.example: {leaked}",
                  "Move them to .env and blank them in .env.example. That file "
                  "is committed to version control.")
        else:
            check("keys in template", OK, "placeholders only")


# --------------------------------------------------------------------------- #
# 5. Does it actually run?
# --------------------------------------------------------------------------- #

def check_pipeline() -> None:
    section("6. End-to-end smoke test")
    try:
        import numpy as np
        import pandas as pd

        from config import CONFIG
        from orchestrator.graph import run_workflow
    except Exception as exc:
        check("imports", FAIL, f"{type(exc).__name__}: {exc}",
              "The package could not be imported. Run this from inside madsa/.")
        return

    CONFIG.llm.backend = "mock"
    rng = np.random.default_rng(0)
    size = 160
    signal = rng.normal(size=size)
    frame = pd.DataFrame({
        "feature_a": signal,
        "feature_b": signal * 0.8 + rng.normal(scale=0.4, size=size),
        "feature_c": rng.normal(size=size),
        "outcome": (signal > 0).astype(int),
    })

    started = time.time()
    try:
        state = run_workflow(frame, "smoke_test.csv")
    except Exception as exc:
        check("workflow", FAIL, f"{type(exc).__name__}: {exc}",
              "The pipeline raised. Run `python -m pytest tests/test_framework.py "
              "-m \"not slow\"` to locate the stage.")
        return

    elapsed = time.time() - started
    failed = [n for n, o in state.outputs.items() if o.status == "failed"]
    if failed:
        check("agent stages", FAIL, f"failed: {failed}",
              "Inspect the named stage's warnings in the run record.")
    else:
        degraded = [n for n, o in state.outputs.items() if o.status == "degraded"]
        detail = f"{len(state.completed)} stage(s) ran in {elapsed:.0f}s"
        if degraded:
            check("agent stages", WARN, f"{detail}; degraded: {degraded}",
                  "Degraded means the stage fell back to computed results. "
                  "Usually the model was unreachable; harmless on the mock backend.")
        else:
            check("agent stages", OK, detail)

    verdict = state.findings("validation").get("verdict")
    if verdict in ("sound", "needs_revision"):
        check("validation", OK, f"verdict '{verdict}'")
    else:
        check("validation", WARN, f"verdict '{verdict}'",
              "'unreliable' means the framework does not stand behind its own "
              "output. On this synthetic smoke test that would be unexpected.")

    modelling = state.findings("modelling")
    if modelling.get("beats_baseline"):
        check("signal detection", OK, "found the planted signal")
    else:
        check("signal detection", WARN, "did not find the planted signal",
              "The smoke test data has a real relationship. Not finding it may "
              "indicate a modelling problem, though small samples are noisy.")


# --------------------------------------------------------------------------- #

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
    print("=" * 72)
    print("  Multi-Agent Data Science Framework - health check")
    print("=" * 72)

    check_python()
    check_packages()
    check_spark()
    check_backends()
    check_files()
    check_pipeline()

    failures = [r for r in results if r[1] == FAIL]
    warnings = [r for r in results if r[1] == WARN]

    print("\n" + "=" * 72)
    print(f"  {len(results) - len(failures) - len(warnings)} OK, "
          f"{len(warnings)} warnings, {len(failures)} failures")
    print("=" * 72)

    if failures:
        print("\nThese must be fixed before the framework will run:\n")
        for name, _, detail, fix in failures:
            print(f"  * {name}: {detail}")
            if fix:
                print(f"      {fix}")
        print()
        return 1

    if warnings:
        print("\nOptional capabilities unavailable (the framework works without "
              "them):\n")
        for name, _, detail, _ in warnings:
            print(f"  * {name}: {detail}")

    print("\nEverything required is working. Try:\n"
          "    python demo.py\n"
          "    python run.py datasets/heart_disease.csv\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
