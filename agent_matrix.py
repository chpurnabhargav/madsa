"""Per-agent health across every real dataset.

Running the pipeline and seeing "sound" at the end says the workflow finished.
It does not say each of the nine agents did useful work: a stage can return
`ok` having produced almost nothing, and the run still completes. This checks
what each agent actually yielded, dataset by dataset, and writes a line per
dataset as it finishes so an interruption costs one run rather than all of them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import pandas as pd

from config import CONFIG
from orchestrator.graph import CANONICAL_ORDER, run_workflow

DATASETS = [
    ("titanic", None),
    ("heart_disease", None),
    ("diabetes", None),
    ("breast_cancer", None),
    ("iris", None),
    ("sklearn_wine", None),
    ("wine_quality", None),
    ("california_housing", "Explain what drives house values across California."),
]

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("agent_matrix")
OUT.mkdir(parents=True, exist_ok=True)


def substance(stage: str, findings: dict) -> str:
    """A short description of what this agent actually produced.

    Deliberately specific per agent: 'ok' is not evidence that a stage did its
    job, and the only way to tell is to look at the thing it was supposed to
    make.
    """
    if findings.get("skipped"):
        return "skipped"
    if stage == "profiling":
        return f"target={findings.get('target_column')} feats={len(findings.get('usable_features') or [])}"
    if stage == "quality":
        return f"{len(findings.get('actions') or [])} action(s), score {findings.get('quality_score')}"
    if stage == "features":
        return (f"+{findings.get('n_added')} adopted={findings.get('adopted')} "
                f"gain={findings.get('gain')}")
    if stage == "eda":
        return f"{len(findings.get('key_insights') or [])} insight(s)"
    if stage == "multivariate":
        return (f"PCA {findings.get('components_for_90pct_variance')}/"
                f"{findings.get('n_features_analysed')} sil={findings.get('silhouette')}")
    if stage == "visualization":
        return f"{len(findings.get('charts') or [])} chart(s)"
    if stage == "modelling":
        return (f"{findings.get('best_model')} {findings.get('best_score')} "
                f"signal={findings.get('beats_baseline')}")
    if stage == "report":
        return (f"{findings.get('n_sections')} section(s), "
                f"{len(findings.get('unsupported_numbers') or [])} bad number(s)")
    if stage == "validation":
        return f"{findings.get('verdict')}, {len(findings.get('issues') or [])} issue(s)"
    return "-"


def main() -> int:
    CONFIG.llm.backend = "mock"
    for name, goal in DATASETS:
        marker = OUT / f"{name}.json"
        if marker.exists():
            continue
        path = Path("datasets") / f"{name}.csv"
        if not path.exists():
            continue

        started = time.time()
        try:
            state = run_workflow(pd.read_csv(path), f"{name}.csv", user_goal=goal)
        except Exception as exc:
            marker.write_text(json.dumps(
                {"dataset": name, "crashed": f"{type(exc).__name__}: {exc}"}),
                encoding="utf-8")
            print(f"{name}: CRASHED {type(exc).__name__}: {exc}", flush=True)
            continue

        record = {
            "dataset": name,
            "rows": int(state.data.shape[0]),
            "elapsed_s": round(time.time() - started, 1),
            "plan": state.plan,
            "stages": {},
        }
        for stage in CANONICAL_ORDER:
            output = state.outputs.get(stage)
            if output is None:
                record["stages"][stage] = {"status": "not planned"}
                continue
            record["stages"][stage] = {
                "status": output.status,
                "warnings": len(output.warnings),
                "first_warning": (output.warnings[0][:110] if output.warnings else None),
                "produced": substance(stage, output.findings),
                "narrative_chars": len(output.narrative or ""),
            }
        marker.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
        print(f"{name}: done in {record['elapsed_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
