"""Command-line entry point.

    python run.py datasets/titanic.csv
    python run.py data.csv --goal "Which customers are likely to churn?"
    python run.py data.csv --backend anthropic
    python run.py data.csv --fixed-pipeline      # disable adaptive planning
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from agents import llm
from config import CONFIG, apply_quick_run
from orchestrator.export import save_report
from orchestrator.graph import run_workflow
from tools import spark_engine


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset, choosing the reader from the file extension."""
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use CSV, Excel, JSON or Parquet.")


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
    parser = argparse.ArgumentParser(
        description="Run the multi-agent data science workflow over a dataset.")
    parser.add_argument("dataset", type=Path, help="Path to the dataset file.")
    parser.add_argument("--goal", help="An optional analysis objective in plain language.")
    # The choices come from the backend registry so this list cannot fall
    # behind the backends that actually exist, which it has done before.
    parser.add_argument("--backend", choices=llm.BACKEND_NAMES,
                        help="Reasoning backend. 'auto' uses whichever "
                             "providers have keys, moving to the next when one "
                             "hits its rate limit. Defaults to the configured "
                             "one.")
    parser.add_argument("--fixed-pipeline", action="store_true",
                        help="Disable adaptive planning and run every stage.")
    parser.add_argument("--quick", action="store_true",
                        help="Skip the stacked ensemble and the hyperparameter "
                             "search, for a demonstration. Every other model "
                             "and check still runs.")
    parser.add_argument("--engine", choices=("pandas", "spark"),
                        help="Force an execution engine instead of choosing by data size.")
    parser.add_argument("--output", type=Path, help="Directory for the report files.")
    parser.add_argument("--quiet", action="store_true", help="Print only the result paths.")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"error: no such file: {args.dataset}", file=sys.stderr)
        return 1

    if args.backend:
        CONFIG.llm.backend = args.backend
    if args.fixed_pipeline:
        CONFIG.adaptive_planning = False
    if args.quick:
        apply_quick_run(True)

    # The execution engine is chosen from the file before anything is read, so
    # a dataset too large for memory is never loaded into it. Reading first and
    # deciding afterwards would fail exactly in the case the distributed engine
    # exists to handle.
    engine = spark_engine.choose_engine(str(args.dataset))
    if args.engine:
        engine = {"engine": args.engine, "size_mb": engine["size_mb"],
                  "reason": f"Engine forced to {args.engine} on the command line."}

    sdf = None
    try:
        if engine["engine"] == "spark":
            sdf = spark_engine.load(str(args.dataset))
            # Only a bounded sample is brought into the driver, for the stages
            # that cannot be distributed.
            df = spark_engine.to_pandas_sample(sdf, CONFIG.spark_collect_rows)
        else:
            df = load_dataset(args.dataset)
    except Exception as exc:
        print(f"error: could not read the dataset: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("error: the dataset is empty.", file=sys.stderr)
        return 1

    if not args.quiet:
        if sdf is not None:
            print(f"Dataset : {args.dataset.name}  "
                  f"({sdf.count():,} rows x {len(sdf.columns)} columns, {engine['size_mb']} MB)")
            print(f"Engine  : Spark  ({df.shape[0]:,}-row sample held in memory)")
        else:
            print(f"Dataset : {args.dataset.name}  "
                  f"({df.shape[0]:,} rows x {df.shape[1]} columns)")
            print("Engine  : pandas")
        print(f"Backend : {CONFIG.llm.backend}"
              f"{'' if CONFIG.adaptive_planning else '  (fixed pipeline)'}")
        print("Running the workflow...\n")

    try:
        state = run_workflow(df, args.dataset.name, user_goal=args.goal,
                             spark_df=sdf, engine=engine)
    except Exception as exc:
        print(f"error: the workflow failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    paths = save_report(state, directory=args.output)
    if sdf is not None:
        spark_engine.stop_session()

    if not args.quiet:
        validation = state.findings("validation")
        print(f"Plan      : {' -> '.join(state.plan)}")
        print(f"Completed : {' -> '.join(state.completed)}")
        if skipped := (state.planning or {}).get("skipped"):
            for s in skipped:
                if isinstance(s, dict):
                    print(f"  skipped {s.get('stage')}: {s.get('reason')}")
        print(f"\nValidation: {validation.get('verdict', 'unknown')}  "
              f"({validation.get('issue_counts', {})})")
        if state.revisions:
            print(f"Revisions : {state.revisions}")
        usage = state.usage or {}
        print(f"Cost      : {usage.get('calls', 0)} model calls, "
              f"{usage.get('total_tokens', 0):,} tokens")
        print(f"Elapsed   : {state.summary()['elapsed_s']:.1f}s\n")

        report = state.findings("report")
        if title := report.get("title"):
            print(title)
            print("-" * min(len(title), 78))
        if summary := report.get("executive_summary"):
            print(summary + "\n")

    for kind, path in paths.items():
        print(f"{kind:9}: {path}")

    # A failed validation is reported through the exit code so the framework can
    # be used in a pipeline where an unreliable analysis should stop the build.
    return 2 if state.findings("validation").get("verdict") == "unreliable" else 0


if __name__ == "__main__":
    raise SystemExit(main())
