"""Head-to-head comparison with published LLM data-science agents.

    python -m evaluation.benchmark_comparison            # both protocols
    python -m evaluation.benchmark_comparison --protocol holdout --seeds 10

Why this exists
---------------
Most of the papers in our survey evaluate on their own benchmarks with their
own metrics, so their figures cannot be set beside ours. One paper is the
exception: Data Interpreter (Hong et al., ACL Findings 2025, arXiv 2402.18679)
reports per-dataset results on four public datasets we also use -- Wine
Recognition, Breast Cancer Wisconsin, Titanic and Kaggle House Prices -- for
itself and five other agent frameworks (AutoGen, OpenInterpreter, TaskWeaver,
XAgent, OpenDevin). On those four, a like-for-like comparison is possible.

Like-for-like means following their protocol rather than ours:

* Their protocol is a single hold-out: 20% of the rows are set aside, the agent
  works on the other 80%, and accuracy (or RMSLE for House Prices) is measured
  once on the 20%. Our usual figures are 5-fold cross-validation means, which
  are not the same quantity. So this script splits 80/20, runs the complete
  framework on the 80% only -- profiling, cleaning, feature engineering,
  model selection, tuning -- and scores the saved model on the untouched 20%.
  Nothing about the hold-out reaches the framework before it is scored.

* A single 20% split of Wine Recognition is 36 rows; one row is 2.8 points of
  accuracy. A comparison decided by which rows happened to land in the
  hold-out would prove nothing, so the split is repeated over several seeds and
  the mean and standard deviation are reported alongside the per-split values.

* Their figures are Normalized Performance Scores: for accuracy NPS is the
  accuracy itself; for RMSLE it is 1 / (1 + RMSLE). Both are reported here in
  the same units as their Table 6.

A second paper, LAMBDA (Sun et al., 2025, arXiv 2407.17535, Table 2), reports
5-fold cross-validated accuracy on Wine and Breast Cancer Wisconsin -- the
protocol our own figures already use. Its "best accuracy" is the highest of
nine models it fitted, so ours is reported two ways: the model the framework
actually selected, and the best of its candidates (their definition).

What it does not control: their exact split rows (their split files are not
reproduced here), and the language model (they used gpt-4-1106-preview; our
numbers do not depend on the language model at all, because no agent here
computes a figure -- see the agent contract). Both are stated in the output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from orchestrator.graph import run_workflow  # noqa: E402

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Published figures. Data Interpreter, Table 6 ("Normalized performance
# score"), arXiv 2402.18679v4 / ACL Findings 2025. Read from the paper's HTML
# table, not from a summary.
# --------------------------------------------------------------------------- #

PUBLISHED_NPS = {
    "wine":          {"AutoGen": 1.00, "OpenInterpreter": 1.00, "TaskWeaver": 1.00,
                      "XAgent": 1.00, "OpenDevin": 0.96, "Data Interpreter": 0.96},
    "breast_cancer": {"AutoGen": 0.97, "OpenInterpreter": 0.96, "TaskWeaver": 0.96,
                      "XAgent": 0.94, "OpenDevin": 0.96, "Data Interpreter": 0.99},
    "titanic":       {"AutoGen": 0.82, "OpenInterpreter": 0.81, "TaskWeaver": 0.43,
                      "XAgent": 0.00, "OpenDevin": 0.81, "Data Interpreter": 0.82},
    "house_prices":  {"AutoGen": 0.88, "OpenInterpreter": 0.87, "TaskWeaver": 0.49,
                      "XAgent": 0.00, "OpenDevin": 0.87, "Data Interpreter": 0.91},
}

DATASETS = {
    # name: (file, target, metric)
    "wine":          ("sklearn_wine.csv", "wine_class", "accuracy"),
    "breast_cancer": ("breast_cancer.csv", "diagnosis", "accuracy"),
    "titanic":       ("titanic.csv", "Survived", "accuracy"),
    "house_prices":  ("kaggle_house_prices.csv", "SalePrice", "rmsle"),
}

# LAMBDA, Table 2: best 5-fold cross-validated accuracy of nine models, as a
# fraction. Read from the paper's HTML table.
PUBLISHED_CV = {"wine": {"LAMBDA": 0.9889}, "breast_cancer": {"LAMBDA": 0.9807}}

# Tracked in git, unlike runs/: these are the figures the README, the report and
# the slides cite, so they travel with the code that produced them.
OUT_DIR = ROOT / "benchmarks"
AGENTS = ("AutoGen", "OpenInterpreter", "TaskWeaver", "XAgent", "OpenDevin",
          "Data Interpreter")


def _score(metric: str, y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    """The raw metric and its Normalized Performance Score, as their Table 6."""
    if metric == "accuracy":
        acc = float(np.mean(np.asarray(y_true).astype(str) == np.asarray(y_pred).astype(str)))
        return acc, acc
    # RMSE between the logarithms of predicted and observed prices -- the
    # Kaggle House Prices metric, and the one their benchmark specifies.
    pred = np.clip(np.asarray(y_pred, dtype=float), 1.0, None)
    rmsle = float(np.sqrt(np.mean((np.log(pred) - np.log(np.asarray(y_true, dtype=float))) ** 2)))
    return rmsle, 1.0 / (1.0 + rmsle)


def run_one(name: str, seed: int) -> dict:
    """One split: run the whole framework on 80%, score the saved model on 20%."""
    file, target, metric = DATASETS[name]
    df = pd.read_csv(ROOT / "datasets" / file)
    stratify = df[target] if metric == "accuracy" else None
    train, hold = train_test_split(df, test_size=0.2, random_state=seed,
                                   stratify=stratify)
    train, hold = train.reset_index(drop=True), hold.reset_index(drop=True)

    started = time.time()
    state = run_workflow(train, f"{name}_split{seed}.csv")
    modelling = state.findings("modelling")
    saved = modelling.get("saved_model") or {}

    record = {"dataset": name, "seed": seed, "metric": metric,
              "train_rows": len(train), "holdout_rows": len(hold),
              "target_chosen": state.target_column,
              "best_model": modelling.get("best_model"),
              "elapsed_s": round(time.time() - started, 1)}

    if state.target_column != target:
        record["error"] = f"framework chose target {state.target_column!r}, expected {target!r}"
        return record
    if not saved.get("path"):
        record["error"] = "no model was saved (baseline won or modelling did not run)"
        return record

    pipeline = joblib.load(saved["path"])
    columns = saved.get("expects_columns") or [c for c in hold.columns if c != target]
    predictions = pipeline.predict(hold[columns])
    raw, nps = _score(metric, hold[target], predictions)
    record.update({"raw": round(raw, 4), "nps": round(nps, 4)})

    # Benchmark models are artefacts of the comparison, not deliverables.
    for p in (saved.get("path"), saved.get("signature")):
        if p:
            Path(p).unlink(missing_ok=True)
    return record


def run_cv(name: str) -> dict:
    """The whole framework on the full dataset; its own 5-fold CV figures."""
    file, target, _ = DATASETS[name]
    df = pd.read_csv(ROOT / "datasets" / file)
    started = time.time()
    state = run_workflow(df, file)
    modelling = state.outputs.get("modelling")
    evaluation = ((modelling.evidence or {}).get("evaluation") or {}) if modelling else {}
    record = {"dataset": name, "target_chosen": state.target_column,
              "elapsed_s": round(time.time() - started, 1)}
    if state.target_column != target or evaluation.get("status") != "ok":
        record["error"] = f"no evaluation for target {target!r}"
        return record
    models = [r for r in evaluation["results"] if not r["is_baseline"]]
    chosen = next(r for r in models if r["model"] == evaluation["best_model"])
    best = max(models, key=lambda r: r["scores"]["accuracy"])
    record.update({
        "cv_folds": evaluation["cv_folds"],
        "selected_model": chosen["model"],
        "selected_accuracy": chosen["scores"]["accuracy"],
        "selected_accuracy_std": chosen["scores"].get("accuracy_std"),
        "best_candidate": best["model"],
        "best_candidate_accuracy": best["scores"]["accuracy"],
        "candidates": len(models),
    })
    return record


def summarise(records: list[dict], cv_records: list[dict] | None = None) -> str:
    """Their Table 6 beside our hold-out results, in their units."""
    lines = [
        "# Head-to-head with published LLM data-science agents",
        "",
        "Published figures: Data Interpreter (Hong et al., ACL Findings 2025, "
        "arXiv 2402.18679), Table 6, Normalized Performance Score. Accuracy for "
        "the three classification sets; 1/(1+RMSLE) for House Prices.",
        "",
        "Ours: the complete framework run on an 80% split, scored once on the "
        "untouched 20% -- their protocol -- repeated over several random splits.",
        "",
        "| Dataset | AutoGen | OpenInterpreter | TaskWeaver | XAgent | OpenDevin "
        "| Data Interpreter | **Ours (mean ± sd)** | Ours, per split |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in DATASETS:
        ok = [r for r in records if r["dataset"] == name and "nps" in r]
        pub = PUBLISHED_NPS[name]
        if ok:
            vals = np.array([r["nps"] for r in ok])
            ours = f"**{vals.mean():.3f} ± {vals.std(ddof=1) if len(vals) > 1 else 0:.3f}**"
            per = ", ".join(f"{v:.3f}" for v in vals)
        else:
            ours, per = "—", "—"
        lines.append(
            f"| {name} | " + " | ".join(f"{pub[k]:.2f}" for k in
            ("AutoGen", "OpenInterpreter", "TaskWeaver", "XAgent", "OpenDevin",
             "Data Interpreter")) + f" | {ours} | {per} |")

    means = {n: np.mean([r["nps"] for r in records if r["dataset"] == n and "nps" in r])
             for n in DATASETS if any(r["dataset"] == n and "nps" in r for r in records)}
    if len(means) == len(DATASETS):
        lines.append("| **Average of the four** | " + " | ".join(
            f"{np.mean([PUBLISHED_NPS[n][k] for n in DATASETS]):.3f}" for k in AGENTS)
            + f" | **{np.mean(list(means.values())):.3f}** | |")

    hp = [r for r in records if r["dataset"] == "house_prices" and "raw" in r]
    if hp:
        rm = np.array([r["raw"] for r in hp])
        lines += ["", f"House Prices raw RMSLE (ours): {rm.mean():.4f} ± "
                      f"{rm.std(ddof=1) if len(rm) > 1 else 0:.4f}. "
                      f"Data Interpreter's NPS of 0.91 corresponds to RMSLE "
                      f"{1/0.91 - 1:.4f}."]

    if cv_records:
        lines += ["", "## Cross-validation protocol: LAMBDA", "",
                  "LAMBDA (Sun et al., 2025, arXiv 2407.17535), Table 2: 5-fold "
                  "cross-validated accuracy, best of the nine models it fitted.", "",
                  "| Dataset | LAMBDA (best of 9) | **Ours, selected model** "
                  "| Ours, best candidate (their definition) |",
                  "|---|---|---|---|"]
        for r in cv_records:
            if "error" in r:
                lines.append(f"| {r['dataset']} | — | {r['error']} | |")
                continue
            lines.append(
                f"| {r['dataset']} | {PUBLISHED_CV[r['dataset']]['LAMBDA']:.4f} "
                f"| **{r['selected_accuracy']:.4f} ± {r['selected_accuracy_std']:.4f}** "
                f"({r['selected_model']}) | {r['best_candidate_accuracy']:.4f} "
                f"({r['best_candidate']}, of {r['candidates']}) |")

    errors = [r for r in records if "error" in r]
    lines += ["", "## What is and is not controlled", "",
              "- Same datasets, same metric, same 80/20 hold-out protocol.",
              "- Not the same split rows: their split files are not reproduced, "
              "so several seeded splits are reported instead of one.",
              "- Not the same language model: they used gpt-4-1106-preview. Our "
              "figures do not depend on the language model -- no agent here "
              "computes a number -- so this run used the offline backend.",
              f"- Runs with an error: {len(errors)}"
              + ("" if not errors else " -- " + "; ".join(
                  f"{e['dataset']}/seed {e['seed']}: {e['error']}" for e in errors))]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--dataset", action="append", choices=list(DATASETS))
    parser.add_argument("--protocol", choices=("holdout", "cv", "all"), default="all")
    args = parser.parse_args()

    # No agent computes a figure, so the narrative backend cannot change a
    # result; the offline one is used so the comparison costs nothing to rerun.
    CONFIG.llm.backend = "mock"
    # The permutation test decides whether the report claims signal; it plays
    # no part in which model is chosen or what it predicts. Skipped for speed.
    CONFIG.permutation_tests = 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = OUT_DIR / "holdout_results.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[(r["dataset"], r["seed"])] = r

    names = args.dataset or list(DATASETS)
    for name in names if args.protocol in ("holdout", "all") else []:
        for seed in range(args.seeds):
            if (name, seed) in done:
                continue
            try:
                record = run_one(name, seed)
            except Exception as exc:  # recorded, not fatal: the sweep continues
                record = {"dataset": name, "seed": seed,
                          "error": f"{type(exc).__name__}: {exc}"}
            # Appended per split, so an interrupted sweep resumes rather than
            # starting over.
            with log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            done[(name, seed)] = record
            shown = record.get("nps", record.get("error"))
            print(f"  {name} seed {seed}: {shown}  ({record.get('best_model')})",
                  flush=True)

    cv_log = OUT_DIR / "cv_results.jsonl"
    cv_done = {}
    if cv_log.exists():
        for line in cv_log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            cv_done[r["dataset"]] = r
    for name in [n for n in names if n in PUBLISHED_CV] if args.protocol != "holdout" else []:
        if name in cv_done:
            continue
        try:
            record = run_cv(name)
        except Exception as exc:  # recorded, not fatal
            record = {"dataset": name, "error": f"{type(exc).__name__}: {exc}"}
        with cv_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        cv_done[name] = record
        print(f"  {name} 5-fold CV: {record.get('selected_accuracy', record.get('error'))}",
              flush=True)

    report = summarise(list(done.values()), list(cv_done.values()))
    (OUT_DIR / "head_to_head.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Every result is on disk by now. The chart exporter's browser and the
    # joblib worker pool can keep the interpreter alive after main returns --
    # observed holding a finished sweep open for minutes -- so exit directly.
    os._exit(code)
