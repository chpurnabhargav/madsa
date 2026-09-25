"""Score new rows with a model the framework produced.

    python predict.py models/diabetes_20260920_191154.joblib new_patients.csv
    python predict.py --latest new_patients.csv -o scored.csv

The analysis pipeline fits models on cross-validation folds and discards them,
which meant a report could recommend a model that no longer existed anywhere. The
modelling stage now refits the winner on every row and saves it; this is the
other half of that, and the reason the framework can be said to run end to end.

Preprocessing lives inside the saved pipeline, so the input here is raw rows in
the shape the original dataset had -- no imputation or encoding to reproduce.
That includes what the analysis did before modelling: numbers stored as text
are converted, category case is normalised, outliers are clipped to the fences
fitted during the run, and derived columns such as a missingness flag or a
logarithm are rebuilt, all by the pipeline's first step. The signature written
beside the model records which raw columns that means, and this script checks
the input against it before scoring rather than after, since a mismatch
discovered inside scikit-learn surfaces as an opaque error about feature counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from config import MODELS_DIR


def _load(model_path: Path) -> tuple[object, dict]:
    """The pipeline and its signature, which are written as a pair."""
    pipeline = joblib.load(model_path)
    signature_path = model_path.with_name(model_path.stem + "_signature.json")
    signature = {}
    if signature_path.exists():
        signature = json.loads(signature_path.read_text(encoding="utf-8"))
    return pipeline, signature


def _latest_model() -> Path | None:
    """The most recently saved model, for the common case of scoring after a run."""
    models = sorted(MODELS_DIR.glob("*.joblib"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0] if models else None


def _check_columns(df: pd.DataFrame, signature: dict) -> list[str]:
    """Differences between what the model expects and what it was given.

    Reported all at once rather than one at a time: someone fixing a spreadsheet
    wants the whole list, not to rediscover the next problem on each attempt.
    """
    expected = signature.get("expects_columns")
    if not expected:
        return []

    problems: list[str] = []
    missing = [c for c in expected if c not in df.columns]
    if missing:
        problems.append(f"Missing column(s) the model needs: {', '.join(missing)}")

    extra = [c for c in df.columns if c not in expected
             and c != signature.get("target")]
    if extra:
        # Not fatal. Extra columns are ignored, but silently dropping something
        # the caller thought was being used would be worse than saying so.
        problems.append(
            f"Ignoring column(s) the model was not fitted on: {', '.join(extra)}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score new rows with a model this framework produced.")
    parser.add_argument("model", nargs="?", type=Path,
                        help="Path to a .joblib saved by a run.")
    parser.add_argument("data", type=Path, help="CSV of rows to score.")
    parser.add_argument("--latest", action="store_true",
                        help="Use the most recently saved model.")
    parser.add_argument("-o", "--output", type=Path,
                        help="Write the scored rows here instead of printing.")
    args = parser.parse_args()

    model_path = args.model
    if args.latest or model_path is None:
        model_path = _latest_model()
        if model_path is None:
            print("No saved model found. Run an analysis first:\n"
                  "    python run.py datasets/heart_disease.csv", file=sys.stderr)
            return 1
    if not model_path.exists():
        print(f"No such model: {model_path}", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"No such file: {args.data}", file=sys.stderr)
        return 1

    pipeline, signature = _load(model_path)
    df = pd.read_csv(args.data)

    print(f"Model   : {signature.get('model', model_path.stem)}")
    # `--latest` picks the newest model from any run, so which analysis it
    # came from is stated before it scores anything.
    if dataset := signature.get("dataset"):
        print(f"Fitted  : on {dataset}, {signature.get('fitted_at', 'date unknown')}")
    print(f"Predicts: {signature.get('target', 'unknown')} "
          f"({signature.get('task', 'unknown')})")
    print(f"Rows    : {len(df):,}")
    if derived := signature.get("derived_features"):
        print(f"Rebuilds: {', '.join(d['feature'] for d in derived)}")

    problems = _check_columns(df, signature)
    fatal = [p for p in problems if p.startswith("Missing")]
    for problem in problems:
        print(f"  ! {problem}")
    if fatal:
        print("\nThe model cannot score these rows.", file=sys.stderr)
        return 1

    expected = signature.get("expects_columns")
    frame = df[expected] if expected else df

    try:
        predictions = pipeline.predict(frame)
    except Exception as exc:
        print(f"\nScoring failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    out = df.copy()
    out["prediction"] = predictions

    # Probabilities when the model has them: a hard label answers "which class"
    # and not "how sure", and the second is what a threshold decision needs.
    if hasattr(pipeline, "predict_proba"):
        try:
            probabilities = pipeline.predict_proba(frame)
            for i, label in enumerate(pipeline.classes_):
                out[f"probability_{label}"] = probabilities[:, i]
        except Exception:
            pass  # not every classifier supports it; the label still stands

    if args.output:
        out.to_csv(args.output, index=False)
        print(f"\nWrote {len(out):,} scored row(s) to {args.output}")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print()
            print(out.head(20).to_string(index=False))
            if len(out) > 20:
                print(f"\n... {len(out) - 20:,} more row(s). Use -o to write them all.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
