"""The full workflow at scale: the large datasets, end to end.

    python -m evaluation.benchmark                 # every dataset below
    python -m evaluation.benchmark covertype telco_churn

The head-to-head with published systems is evaluation/benchmark_comparison.py,
run under each paper's own protocol. This answers a different question: does
the complete framework -- quality repairs, feature engineering, model
selection, tuning, verification -- still work, and still choose the right
target, on datasets of tens to hundreds of thousands of rows, where files above
the configured size are read by Apache Spark.

No published agent result exists on these datasets, so none is shown. Model
scores are computed by scikit-learn and not by the language model, so the
offline backend gives the same numbers as a live one and costs no API quota.

Run `python datasets/fetch_benchmark_data.py` first; these files are downloaded
rather than committed.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from orchestrator.graph import run_workflow  # noqa: E402
from tools import spark_engine  # noqa: E402

OUT = ROOT / "benchmarks" / "scale"

# name -> (file, stated goal, expected target). The goal is what a user would
# type; the expected target is checked, not supplied, so a wrong choice by the
# framework is reported rather than hidden.
DATASETS: dict[str, tuple[str, str | None, str]] = {
    "adult_income": ("adult_income.csv", None, "income"),
    "bank_marketing": ("bank_marketing.csv",
                       "Predict whether the client subscribed.", "subscribed"),
    "telco_churn": ("telco_churn.csv",
                    "Predict whether the customer will churn.", "CustomerChurned"),
    "miniboone": ("miniboone.csv",
                  "Identify the particle signal from the detector readings.", "signal"),
    "covertype": ("covertype.csv", "Predict the forest Cover_Type.", "Cover_Type"),
    "credit_card_fraud": ("credit_card_fraud.csv",
                          "Identify which transactions are fraudulent.", "Class"),
}


def run_one(name: str) -> dict[str, Any]:
    file, goal, expected = DATASETS[name]
    path = ROOT / "datasets" / file
    decision = spark_engine.choose_engine(str(path))
    sdf = None
    if decision["engine"] == "spark":
        sdf = spark_engine.load(str(path))
        df = spark_engine.to_pandas_sample(sdf, CONFIG.spark_collect_rows)
    else:
        df = pd.read_csv(path)

    started = time.time()
    state = run_workflow(df, file, user_goal=goal, spark_df=sdf, engine=decision)
    elapsed = round(time.time() - started, 1)

    modelling = state.outputs.get("modelling")
    evaluation = ((modelling.evidence or {}).get("evaluation") or {}) if modelling else {}
    row: dict[str, Any] = {"dataset": name, "target": state.target_column,
                           "expected_target": expected,
                           "target_correct": state.target_column == expected,
                           "engine": decision["engine"], "elapsed_s": elapsed}
    if evaluation.get("status") != "ok":
        return {**row, "status": "no_model"}

    best = next(r for r in evaluation["results"] if r["model"] == evaluation["best_model"])
    perm = evaluation.get("permutation_test") or {}
    return {
        **row,
        "status": "ok",
        "rows_total": int(state.full_row_count or len(df)),
        "rows_modelled": int(evaluation["n_rows"]),
        "task": evaluation["task"],
        "best_model": evaluation["best_model"],
        "tuned": evaluation.get("best_model_tuned"),
        "scores": best["scores"],
        "baseline_score": evaluation.get("baseline_score"),
        "lift_over_baseline": evaluation.get("lift_over_baseline"),
        "p_value": perm.get("p_value"),
        "permutation_test_error": evaluation.get("permutation_test_error"),
        "beats_baseline": evaluation.get("beats_baseline"),
        "suspected_leakage": state.findings("modelling").get("suspected_leakage"),
        "full_data_mllib": state.findings("modelling").get("distributed_validation"),
        "n_models_compared": sum(1 for r in evaluation["results"] if not r["is_baseline"]),
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Dataset | Rows | Engine | Target (correct?) | Best model | Accuracy "
             "| Macro F1 | ROC-AUC | Baseline | Full data, Spark MLlib | Time (s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        mark = "yes" if r.get("target_correct") else "**no**"
        if r.get("status") != "ok":
            lines.append(f"| {r['dataset']} | | | {r.get('target')} ({mark}) "
                         f"| {r.get('status')}: {r.get('error', '')} | | | | | | |")
            continue
        s = r["scores"]
        rows_txt = f"{r['rows_total']:,}"
        if r["rows_modelled"] < r["rows_total"]:
            rows_txt += f" ({r['rows_modelled']:,} modelled)"
        full = r.get("full_data_mllib") or {}
        full_txt = (f"{full['mllib_model']}: acc {full.get('accuracy')}, F1 {full.get('f1_macro')} "
                    f"({full['rows_train']:,} train rows)" if full.get("status") == "ok" else "–")
        lines.append(f"| {r['dataset']} | {rows_txt} | {r['engine']} | {r['target']} ({mark}) "
                     f"| {r['best_model']} | {s.get('accuracy', '–')} | {s.get('f1_macro', '–')} "
                     f"| {s.get('roc_auc', '–')} | {r['baseline_score']} | {full_txt} "
                     f"| {r['elapsed_s']} |")
    return "\n".join(lines)


def main(names: list[str]) -> None:
    # Narrative only; no figure depends on the language model.
    CONFIG.llm.backend = "mock"
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for name in names or list(DATASETS):
            done = OUT / f"{name}.json"
            # Resume: a finished dataset is kept unless named explicitly.
            if not names and done.exists() and "expected_target" in done.read_text(encoding="utf-8"):
                continue
            print(f"running {name}...", flush=True)
            try:
                row = run_one(name)
            except Exception as exc:  # one dataset failing must not lose the rest
                row = {"dataset": name, "status": "failed",
                       "error": f"{type(exc).__name__}: {exc}"}
            (OUT / f"{name}.json").write_text(json.dumps(row, indent=2, default=str),
                                              encoding="utf-8")
            print(f"  {row.get('status')} {row.get('target')} {row.get('best_model')} "
                  f"{(row.get('scores') or {}).get('accuracy')} {row.get('elapsed_s')}s",
                  flush=True)
    finally:
        spark_engine.stop_session()

    rows = [json.loads((OUT / f"{n}.json").read_text(encoding="utf-8"))
            for n in DATASETS if (OUT / f"{n}.json").exists()]
    table = format_table(rows)
    (OUT.parent / "scale.md").write_text(
        "# The full workflow at scale\n\nFive-fold cross-validated scores of the model "
        "the framework selected. Files above the size threshold are read by Apache "
        "Spark; in-memory stages use a bounded sample.\n\n" + table + "\n",
        encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main(sys.argv[1:])
    sys.stdout.flush()
    # Results are on disk; do not wait on lingering worker or browser processes.
    os._exit(0)
