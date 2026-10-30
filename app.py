"""Streamlit interface for the multi-agent framework.

    streamlit run app.py

The interface deliberately shows the workflow, not only its result. Which stages
the coordinator chose, what each agent decided, where an agent's output was
corrected, and what the validator objected to are all visible, because the
framework's claim is about how the analysis was produced and that claim is only
inspectable if the process is shown.

Two devices carry that. The pipeline rail draws the nine agents in the order
they run, with the verification layer beneath them; before a run it explains
the system, and after one it becomes the record of what happened. The ledger in
the Verification tab lists every registered check with what it compares
against and what it caught -- the layer the project contributes, made visible.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

import verification
from agents import llm
from config import CONFIG, apply_quick_run
from orchestrator import plain_language as pl
from orchestrator.export import save_report, to_html, to_markdown
from orchestrator.graph import CANONICAL_ORDER, run_workflow
from tools import spark_engine

st.set_page_config(page_title="MADSA · Verified multi-agent analysis",
                   page_icon="📊", layout="wide")

HERE = Path(__file__).resolve().parent
DATASETS = HERE / "datasets"

#: Each stage's name as a reader knows it, and what it does, in run order.
STAGES = {
    "profiling": ("Profiling", "Reads structure and types, and finds the likely target"),
    "quality": ("Data quality", "Finds problems and authorises proportionate repairs"),
    "features": ("Features", "Derives predictors, and keeps them only if they measurably help"),
    "eda": ("Exploration", "Correlations, group differences and distributions"),
    "multivariate": ("Structure", "PCA, clustering and redundancy across all features"),
    "visualization": ("Charts", "Chooses the charts that carry the findings"),
    "modelling": ("Modelling", "Cross-validates models and tests them against chance"),
    "report": ("Report", "Writes the report from computed findings only"),
    "validation": ("Validation", "Reviews every claim against the calculations"),
}

#: A reader's name for each registered check.
CHECK_LABELS = {
    "unsupported_numbers": "Number provenance",
    "causal_language": "Causal language",
    "hallucinated_columns": "Invented columns",
    "overstated_segmentation": "Overstated segments",
    "overstated_feature_benefit": "Overstated feature benefit",
    "suppressed_negative_result": "Buried null result",
    "unsupported_non_linear_signal": "Unsupported non-linear signal",
    "invented_features": "Invented features",
    "overridden_signal_claim": "Signal and recommendation",
    "unsupported_importance_claim": "Unsupported importance",
    "overstated_effect": "Overstated effect",
    "refused_cleaning_action": "Repair screening",
    "failed_chart_request": "Chart validation",
}

METRIC_LABELS = {"f1_macro": "Macro F1", "r2": "R-squared",
                 "roc_auc": "ROC AUC", "accuracy": "Accuracy"}

#: Provider keys, checked up front so a missing one is reported here rather
#: than surfacing as an authentication failure part-way through a run.
KEY_FOR = {"anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY",
           "groq": "GROQ_API_KEY", "grok": "GROK_API_KEY"}


# --------------------------------------------------------------------------- #
# Markup helpers
#
# Every value placed in markup is escaped. Column names, file names and model
# output are all untrusted: a dataset column called "<img src=x onerror=...>"
# was inserted verbatim by the previous interface. Dollar signs are escaped as
# well, because Streamlit renders text between two of them as LaTeX.
# --------------------------------------------------------------------------- #

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=IBM+Plex+Mono:wght@400;500&'
    'family=IBM+Plex+Sans+Condensed:wght@500;600&'
    'family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&'
    'family=STIX+Two+Text:ital,wght@0,400;0,600;1,400;1,600&display=swap" '
    'rel="stylesheet">'
)


def esc(value: Any) -> str:
    """A value made safe to place in markup."""
    text = "" if value is None else str(value)
    return html.escape(text, quote=True).replace("$", "&#36;")


def md(text: Any) -> str:
    """Text made safe to render as Markdown: no accidental LaTeX."""
    return ("" if text is None else str(text)).replace("$", "\\$")


def show(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def _stylesheet() -> str:
    """The stylesheet as a single block with no comments or blank lines.

    Streamlit renders through a Markdown parser, and CommonMark ends an HTML
    block at the first blank line -- a readable stylesheet would close its own
    <style> tag partway through and print the remainder as text.
    """
    raw = (HERE / "assets" / "app.css").read_text(encoding="utf-8")
    without_comments = re.sub(r"/\*.*?\*/", "", raw, flags=re.S)
    return "\n".join(line for line in without_comments.splitlines() if line.strip())


def callout(title: str, detail: str, tone: str = "") -> str:
    return (f'<div class="callout {tone}"><div class="t">{esc(title)}</div>'
            f'<div class="d">{esc(detail)}</div></div>')


def readout(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return (f'<div class="readout t-{tone}"><div class="k">{esc(label)}</div>'
            f'<div class="v">{esc(value)}</div>'
            + (f'<div class="s">{esc(sub)}</div>' if sub else "") + "</div>")


def _station_of(mechanism: verification.Mechanism) -> str:
    """The stage a check is drawn under. "any" is shown where it first applies."""
    return "profiling" if mechanism.stage == "any" else mechanism.stage


def _checks_by_stage() -> dict[str, list[str]]:
    guards: dict[str, list[str]] = {}
    for mechanism in verification.MECHANISMS:
        guards.setdefault(_station_of(mechanism), []).append(mechanism.name)
    return guards


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #

def masthead() -> str:
    return (
        '<div class="masthead">'
        '<div class="eyebrow">MADSA · Multi-agent data science</div>'
        '<div class="headline">Every figure in the report is traced to a calculation.</div>'
        '<div class="lede">Nine agents analyse a dataset in sequence. Tools compute '
        'every statistic; the agents only interpret them, and '
        f'{len(verification.MECHANISMS)} checks hold each interpretation against the '
        'numbers before anything reaches the report.</div></div>')


def rail(state: Any = None, live: bool = False) -> str:
    """The nine agents in run order, with the verification layer beneath.

    Numbered because the order is real: each stage reads what the earlier ones
    established. Without a run it describes each agent; with one it records
    what each did, how long it took, and what the checks it carries caught.
    `live` draws a run still in progress: the stage at work, and those waiting.
    """
    guards = _checks_by_stage()
    fired = ((state.verification or verification.firings()) if live
             else state.verification or {}) if state is not None else {}
    skipped_now = ({e.get("agent") for e in state.trace if e.get("event") == "stage_skipped"}
                   if state is not None else set())
    working = None
    if live and state is not None:
        working = next((s for s in CANONICAL_ORDER
                        if s not in state.outputs and s not in skipped_now
                        and (not state.plan or s in state.plan)), None)
    stations = []
    for number, stage in enumerate(CANONICAL_ORDER, 1):
        name, duty = STAGES[stage]
        checks = guards.get(stage, [])
        if state is None:
            status, meta = "idle", duty
            guard = (f'<span class="guard">{_plural(len(checks), "check")}</span>'
                     if checks else "")
        elif live and stage == working:
            status, meta, guard = "running", "working…", ""
        elif live and stage not in state.outputs and stage not in skipped_now and (
                not state.plan or stage in state.plan):
            status, meta, guard = "pending", "waiting", ""
        else:
            output = state.outputs.get(stage)
            if output is None:
                status = "skipped"
                meta = ("left out by the plan" if state.plan and stage not in state.plan
                        else "not needed for this data")
                guard = ""
            else:
                status = output.status if output.status in ("ok", "degraded", "failed") else "ok"
                word = {"ok": "done", "degraded": "degraded", "failed": "failed"}[status]
                meta = f"{output.duration_s:.1f}s · {word}"
                caught = sum(fired.get(n, 0) for n in checks)
                if not checks:
                    guard = ""
                elif not CONFIG.verification_enabled:
                    guard = '<span class="guard none">checks off</span>'
                elif caught:
                    guard = f'<span class="guard caught">caught {caught}</span>'
                else:
                    guard = '<span class="guard">checks clear</span>'
        stations.append(
            f'<li class="station s-{status}"><span class="num">{number:02d}</span>'
            f'<span class="node"></span><span class="name">{esc(name)}</span>'
            f'<span class="meta">{esc(meta)}</span>{guard}</li>')

    total = len(verification.MECHANISMS)
    if state is None:
        head = ('<span class="t">The workflow</span>'
                '<span class="n">nine agents · shown in the order they run</span>')
        foot = (f'<span>Verification layer · {total} checks at the stage boundaries</span>'
                '<span>A number comes from a tool, never from the language model</span>')
    elif live:
        planned = [s for s in CANONICAL_ORDER if not state.plan or s in state.plan]
        done = sum(1 for s in planned if s in state.outputs or s in skipped_now)
        head = ('<span class="t">Running</span>'
                f'<span class="n">{done} of {len(planned)} planned stages finished</span>')
        foot = (f'<span>Verification layer · {total} checks</span>'
                f'<span>{sum(fired.values())} caught so far</span>')
    else:
        ran = sum(1 for s in CANONICAL_ORDER if s in state.outputs)
        caught_total = sum(fired.values())
        head = ('<span class="t">This run</span>'
                f'<span class="n">{ran} of 9 stages ran'
                + (f' · report revised {_plural(state.revisions, "time")}'
                   if state.revisions else "") + '</span>')
        foot = (f'<span>Verification layer · {total} checks</span><span>'
                + ("switched off for this run" if not CONFIG.verification_enabled
                   else f"caught {caught_total} along the way" if caught_total
                   else "nothing needed correcting")
                + '</span>')
    return (f'<div class="figure"><div class="figure-head">{head}</div>'
            f'<ol class="rail">{"".join(stations)}</ol>'
            f'<div class="layer">{foot}</div></div>')


def dataset_strip(df: pd.DataFrame, name: str, engine: dict | None = None) -> str:
    try:
        duplicates = f"{int(df.duplicated().sum()):,}"
    except TypeError:  # cells holding lists or dicts cannot be compared
        duplicates = "—"
    rows = f"{df.shape[0]:,}"
    if (engine or {}).get("full_rows"):
        # The frame on screen is the bounded sample; say so, and how large the
        # file really is, rather than presenting the sample as the dataset.
        rows = f"{engine['full_rows']:,} (Spark; {df.shape[0]:,}-row sample in memory)"
    facts = [("Rows", rows), ("Columns", f"{df.shape[1]:,}"),
             ("Missing cells", f"{int(df.isna().sum().sum()):,}"),
             ("Duplicate rows", duplicates)]
    return ('<div class="dataset"><div class="file"><span class="k">Dataset</span>'
            f'<div class="v">{esc(name)}</div></div>'
            + "".join(f'<div class="fact"><span class="k">{esc(k)}</span>'
                      f'<span class="v">{esc(v)}</span></div>' for k, v in facts)
            + "</div>")


SEALS = {
    "sound": ("Sound", "every check passed"),
    "needs_revision": ("Confirm", "points to confirm"),
    "unreliable": ("Unreliable", "read Verification first"),
}


def result_head(state: Any) -> str:
    """The answer, in words, beside how far to trust it."""
    modelling = state.findings("modelling")
    validation = state.findings("validation")
    verdict = validation.get("verdict") or "unknown"
    target = state.target_column
    rows = int(state.full_row_count or state.data.shape[0])

    if modelling and not modelling.get("skipped"):
        metric = modelling.get("primary_metric")
        if modelling.get("beats_baseline"):
            headline = (f"Yes: {modelling.get('best_model')} predicts {target} "
                        f"better than guessing.")
            meaning = pl.plain_score(metric, modelling.get("best_score"))
            meaning = (meaning[:1].upper() + meaning[1:] + ".") if meaning else ""
        else:
            headline = f"No: nothing in this data predicts {target} better than guessing."
            meaning = ("That is a finding about the data, not a failure of the "
                       "analysis. Every model was held against deliberately "
                       "unintelligent baselines and against shuffled labels, and "
                       "none cleared them.")
    else:
        headline = f"{rows:,} rows explored across {state.data.shape[1]} columns."
        meaning = ("No outcome to predict was identified, so this analysis "
                   "describes the data rather than modelling it.")

    eyebrow = f"Analysis of {state.dataset_name}"
    if state.engine == "spark":
        eyebrow += " · Apache Spark"
    label, note = SEALS.get(verdict, ("Unchecked", "no verdict recorded"))
    if verdict == "needs_revision":
        note = _plural(int(validation.get("n_issues") or 0), "point") + " to confirm"
    return ('<div class="result-head"><div class="answer">'
            f'<div class="eyebrow">{esc(eyebrow)}</div>'
            f'<div class="headline">{esc(headline)}</div>'
            + (f'<div class="meaning">{esc(meaning)}</div>' if meaning else "")
            + f'</div><div class="seal {esc(verdict)}"><div class="k">Verdict</div>'
            f'<div class="v">{esc(label)}</div><div class="s">{esc(note)}</div></div></div>')


def readouts(state: Any) -> str:
    modelling = state.findings("modelling")
    cards = [readout("Predicting", state.target_column or "Nothing",
                     (state.task_type or "exploratory").replace("none", "exploratory").title(),
                     "focus")]
    if modelling and not modelling.get("skipped"):
        metric = modelling.get("primary_metric") or "score"
        score = modelling.get("best_score")
        baseline = modelling.get("baseline_score")
        compared = len(modelling.get("model_scores") or {})
        cards.append(readout(
            METRIC_LABELS.get(metric, metric),
            f"{score:.4f}" if isinstance(score, (int, float)) else "—",
            (f"best of {compared} models" if compared else "")
            + (f" · baseline {baseline:.4f}" if isinstance(baseline, (int, float)) else "")))
        beats = modelling.get("beats_baseline")
        p_value = modelling.get("permutation_p_value")
        cards.append(readout(
            "Signal", "Found" if beats else "None",
            (f"permutation p = {p_value}" if p_value is not None else "")
            or ("clears the baseline" if beats else "does not beat guessing"),
            "good" if beats else "bad"))
    else:
        cards.append(readout("Rows", f"{int(state.full_row_count or state.data.shape[0]):,}"))
        cards.append(readout("Columns", f"{state.data.shape[1]:,}"))

    fired = state.verification or {}
    caught = sum(fired.values())
    checks = len(verification.MECHANISMS)
    cards.append(readout(
        "Verification", f"{caught} caught" if caught else "Nothing caught",
        f"by {sum(1 for v in fired.values() if v)} of {checks} checks"
        if caught else f"{checks} checks ran", "focus"))
    return f'<div class="readouts">{"".join(cards)}</div>'


def ledger(state: Any) -> str:
    """Every registered check, what it is held against, and what it caught."""
    fired = state.verification or {}
    order = {stage: i for i, stage in enumerate(CANONICAL_ORDER)}
    rows = []
    for mechanism in sorted(verification.MECHANISMS,
                            key=lambda m: order.get(_station_of(m), 99)):
        count = fired.get(mechanism.name, 0)
        if not CONFIG.verification_enabled:
            tally = '<span class="tally off">switched off</span>'
        elif count:
            tally = f'<span class="tally caught">caught {count}</span>'
        else:
            tally = '<span class="tally clear">clear</span>'
        stage = ("Every stage" if mechanism.stage == "any"
                 else STAGES.get(mechanism.stage, (mechanism.stage,))[0])
        checks = mechanism.checks[:1].upper() + mechanism.checks[1:]
        rows.append(
            f'<tr><td>{esc(stage)}</td><td class="check">'
            f'<b>{esc(CHECK_LABELS.get(mechanism.name, mechanism.name))}</b>'
            f'<span>{esc(checks)}.</span></td>'
            f'<td class="against">{esc(mechanism.compares_against)}</td><td>{tally}</td></tr>')
    caught_by = sum(1 for v in fired.values() if v)
    return ('<div class="ledger-wrap"><table class="ledger"><thead><tr>'
            '<th>Stage</th><th>Check</th><th>Held against</th><th>This run</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            f'<div class="ledger-foot"><span>{len(verification.MECHANISMS)} checks · '
            f'{caught_by} caught something in this run</span>'
            '<span>Each compares what a model wrote with what a tool computed</span>'
            '</div></div>')


def objection(issue: dict[str, Any]) -> str:
    severity = str(issue.get("severity") or "info").lower()
    if severity not in ("critical", "warning", "info"):
        severity = "info"
    source = ("automated check" if issue.get("source") == "automated_check"
              else "model review")
    parts = [f'<div class="objection"><div class="head">'
             f'<span class="sev {severity}">{esc(severity)}</span>'
             f'<span>{esc(llm.as_text(issue.get("location")) or "analysis")}</span>'
             f'<span>· {esc(source)}</span></div>'
             f'<div class="body">{esc(llm.as_text(issue.get("problem")))}</div>']
    if fix := llm.as_text(issue.get("suggested_fix")):
        parts.append(f'<div class="fix">Suggested fix: {esc(fix)}</div>')
    if evidence := llm.as_text(issue.get("evidence")):
        parts.append(f'<div class="ev">{esc(evidence)}</div>')
    parts.append("</div>")
    return "".join(parts)


def fingerprint(df: pd.DataFrame, name: str) -> str:
    """Enough to tell one loaded dataset from another."""
    try:
        digest = int(pd.util.hash_pandas_object(df.head(500), index=False).sum())
    except Exception:  # unhashable cells; the name and shape still distinguish
        digest = 0
    return f"{name}|{df.shape[0]}x{df.shape[1]}|{digest}"


@st.cache_resource(show_spinner="Loading the dataset...")
def load_included(path: str) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
    """An included dataset, read by the engine its size calls for.

    The same decision `run.py` makes. Reading Covertype's 581,012 rows straight
    into pandas would skip the distributed path the framework exists to
    demonstrate: profiling, quality and EDA run on Spark across every row, and
    the in-memory stages read a bounded sample. Cached, so a rerun of the page
    does not reload it; the run receives a copy.
    """
    engine = spark_engine.choose_engine(path)
    if engine["engine"] == "spark":
        sdf = spark_engine.load(path)
        # Counted once here, not on every redraw of the page.
        engine = {**engine, "full_rows": sdf.count()}
        return spark_engine.to_pandas_sample(sdf, CONFIG.spark_collect_rows), sdf, engine
    return pd.read_csv(path), None, engine


def read_upload(uploaded: Any) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith((".csv", ".txt")):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded)
    raise ValueError("use a CSV, Excel or Parquet file")


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #

show(FONTS + "<style>" + _stylesheet() + "</style>")

with st.sidebar:
    show('<div class="brand"><div class="mark">MADSA</div>'
         '<div class="tag">Verified multi-agent analysis</div></div>')

    show('<div class="side-head">Dataset</div>')
    uploaded = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "parquet"],
                                help="CSV, Excel or Parquet. Nothing leaves this "
                                     "machine except the summaries sent to the "
                                     "language model you choose below.")
    samples = sorted(DATASETS.glob("*.csv")) if DATASETS.exists() else []
    sample_choice = None
    if samples:
        sample_choice = st.selectbox("Or choose an included dataset",
                                     ["—"] + [p.name for p in samples])
    goal = st.text_area(
        "Question to answer (optional)",
        placeholder="e.g. Which customers are likely to churn?",
        help="A stated question steers which column is treated as the outcome "
             "and what the report emphasises.")

    show('<div class="side-head">Reasoning model</div>')
    # From the registry, not retyped: a hand-written copy of this list had
    # already fallen behind once, and omitted the failover chain.
    backends = list(llm.BACKEND_NAMES)
    backend = st.selectbox(
        "Provider", backends,
        index=backends.index(CONFIG.llm.backend) if CONFIG.llm.backend in backends else 0,
        help="Which model writes the explanations. Every statistic, model and "
             "check is computed the same way whichever you choose; only the "
             "wording changes. 'mock' runs offline at no cost; 'auto' uses "
             "whichever providers you have keys for, moving to the next when one "
             "reaches its rate limit.")
    if (needed := KEY_FOR.get(backend)) and not os.getenv(needed):
        st.warning(f"{needed} is not set. Add it to .env, or choose 'mock' to "
                   f"run offline.")

    show('<div class="side-head">Workflow</div>')
    adaptive = st.toggle(
        "Adaptive planning", value=CONFIG.adaptive_planning,
        help="On: the coordinator chooses which stages this dataset needs. "
             "Off: every stage runs, in a fixed order.")
    retries = st.slider(
        "Report revisions allowed", 0, 3, CONFIG.max_validation_retries,
        help="How many times validation may send the report back to be rewritten "
             "with its objections in hand.")
    quick = st.toggle(
        "Quick run", value=False,
        help="For live demonstrations. Skips the stacked ensemble and the "
             "hyperparameter search, which take most of the time on a laptop. "
             "Every remaining model is still cross-validated and tested against "
             "shuffled labels, and every check still runs.")

    show('<div class="side-note">Every statistic is computed by a tool. The '
         'language model interprets the results, and a number it writes is '
         'checked against the computed ones before the report is delivered.</div>')

# -- Load ----------------------------------------------------------------------

df, dataset_name, spark_df, engine = None, None, None, None
if uploaded is not None:
    try:
        df, dataset_name = read_upload(uploaded), uploaded.name
    except Exception as exc:
        st.error(f"That file could not be read ({type(exc).__name__}: {exc}). "
                 f"Check that it is a CSV, Excel or Parquet file with a header row.")
elif sample_choice and sample_choice != "—":
    df, spark_df, engine = load_included(str(DATASETS / sample_choice))
    dataset_name = sample_choice

state = st.session_state.get("state")
state_key = st.session_state.get("state_key")
current_key = fingerprint(df, dataset_name) if df is not None else None

# Results belong to the dataset they were computed on. They previously stayed
# on screen after another dataset was loaded, under the new dataset's name.
stale = (state is not None and df is not None
         and (state_key != current_key if state_key
              else state.dataset_name != dataset_name))
showing_results = state is not None and not stale

if df is None and not showing_results:
    show(masthead())
    show(rail())
    show(callout(
        "Choose a dataset to begin",
        f"Pick one of the {len(samples)} included datasets in the sidebar, or "
        "upload your own CSV, Excel or Parquet file. A question is optional. With "
        "the offline 'mock' model a run takes seconds; with a hosted model, a few "
        "minutes."))
    st.stop()

if df is not None:
    rail_slot = None
    if not showing_results:
        show(masthead())
        # A placeholder, so the run below can redraw this same rail as each
        # agent finishes rather than stacking a second one beneath it.
        rail_slot = st.empty()
        rail_slot.markdown(rail(), unsafe_allow_html=True)
    show(dataset_strip(df, dataset_name, engine))
    with st.expander("Preview the first 50 rows"):
        st.dataframe(df.head(50), use_container_width=True)
    if stale:
        show(callout(
            "Earlier results are hidden",
            f"They were computed on {state.dataset_name}. Run the analysis to "
            f"analyse {dataset_name}."))

    if st.button("Run analysis", type="primary", use_container_width=True):
        CONFIG.llm.backend = backend
        CONFIG.adaptive_planning = adaptive
        CONFIG.max_validation_retries = retries
        apply_quick_run(quick)

        slot = rail_slot or st.empty()

        def redraw(partial: Any) -> None:
            slot.markdown(rail(partial, live=True), unsafe_allow_html=True)

        with st.spinner("The agents are working. The rail above updates as each finishes."):
            try:
                result = run_workflow(df.copy(), dataset_name,
                                      user_goal=(goal or "").strip() or None,
                                      spark_df=spark_df, engine=engine,
                                      on_progress=redraw)
            except Exception as exc:
                st.error(f"The workflow stopped with an error: {type(exc).__name__}: {exc}")
                st.stop()

        st.session_state["state"] = result
        st.session_state["state_key"] = current_key
        st.session_state["state_quick"] = quick
        # A fresh pass lays the page out for results, without the landing
        # material this pass has already drawn above them.
        st.rerun()

if not showing_results:
    st.stop()

# -- Results -------------------------------------------------------------------

validation = state.findings("validation")
modelling = state.findings("modelling")
verdict = validation.get("verdict") or "unknown"

show(result_head(state))
show(readouts(state))
if st.session_state.get("state_quick"):
    st.caption("Quick run: the stacked ensemble and the hyperparameter search "
               "were skipped. The models compared are listed in the report.")

if not CONFIG.verification_enabled:
    show(callout("The verification layer is switched off",
                 "This is an ablation setting. Nothing in this run was checked "
                 "against the computed results.", "bad"))
elif verdict == "unreliable":
    show(callout("This analysis is not reliable",
                 "An automated reviewer found problems it could not reconcile "
                 "with the calculations. Read the Verification tab before using "
                 "any conclusion.", "bad"))
elif verdict == "needs_revision":
    show(callout("Some points need confirming",
                 "An automated reviewer checked every claim against the "
                 "calculations and listed what it could not reconcile. They are "
                 "points to confirm, not necessarily errors.", "warn"))

if (check := modelling.get("distributed_validation") or {}).get("status") == "ok":
    score = (f"accuracy {check['accuracy']} and macro F1 {check['f1_macro']}"
             if "accuracy" in check else f"R² {check['r2']}")
    show(callout(
        "Checked on the full dataset with Apache Spark",
        f"The model was chosen on a {len(state.data):,}-row sample. Spark MLlib then "
        f"retrained the same family ({check['mllib_model']}) on {check['rows_train']:,} "
        f"rows and scored {score} on {check['rows_test']:,} rows it had not seen. "
        f"MLlib's trees are depth-limited to fit this machine, so treat that as a "
        f"floor rather than a second estimate."))

show(rail(state))

tabs = st.tabs(["Report", "Charts", "Verification", "Workflow", "Agents", "Export"])

# -- Report: the same renderer as the Markdown and HTML files --------------------
with tabs[0]:
    # The marker lets the stylesheet typeset this panel as a paper.
    show('<div class="report-marker"></div>')
    # Chart images are dropped: the Charts tab draws them interactively, and
    # their relative paths would not resolve here.
    body = "\n".join(line for line in to_markdown(state, include_appendix=False).splitlines()
                     if not line.startswith("!["))
    st.markdown(md(body))

# -- Charts ----------------------------------------------------------------------
with tabs[1]:
    viz = state.outputs.get("visualization")
    figures = viz.evidence.get("figures", {}) if viz else {}
    charts = state.findings("visualization").get("charts", [])
    if not charts:
        show(callout("No charts were drawn",
                     "The coordinator judged that this dataset had nothing worth "
                     "plotting, or the charting stage did not run."))
    columns = st.columns(2, gap="large")
    for number, chart in enumerate(charts, 1):
        with columns[(number - 1) % 2]:
            show(f'<div class="fig-label">Fig. {number} · '
                 f'{esc(str(chart.get("kind", "")).replace("_", " "))}</div>')
            if figure := figures.get(chart.get("id")):
                st.plotly_chart(figure, use_container_width=True, theme=None,
                                key=f"figure-{chart.get('id')}")
            else:
                show(f'<div class="fig-caption"><b>{esc(chart.get("title"))}</b></div>')
            show(f'<div class="fig-caption">{esc(chart.get("description"))}</div>'
                 + (f'<div class="fig-why">Chosen because: {esc(chart.get("reason"))}</div>'
                    if chart.get("reason") else ""))

# -- Verification ----------------------------------------------------------------
with tabs[2]:
    show('<div class="section-label">The verification layer</div>')
    show(ledger(state))
    show('<div class="section-label">What the reviewer flagged</div>')
    if review := state.outputs.get("validation"):
        if review.narrative:
            st.write(md(review.narrative))
    issues = validation.get("issues") or []
    if issues:
        show("".join(objection(i) for i in issues if isinstance(i, dict)))
    else:
        show(callout("Nothing was flagged",
                     "The reviewer found no claim it could not reconcile with the "
                     "calculations."))

# -- Workflow --------------------------------------------------------------------
with tabs[3]:
    planning = state.planning or {}
    show('<div class="section-label">The coordinator\'s plan</div>')
    if reasoning := planning.get("reasoning"):
        st.write(md(reasoning))
    skipped = [s for s in (planning.get("skipped") or []) if isinstance(s, dict)]
    if skipped:
        st.markdown("\n".join(
            f"- **{STAGES.get(s.get('stage'), (s.get('stage'),))[0]}** was left out: "
            f"{md(s.get('reason'))}" for s in skipped))
    if repairs := planning.get("repairs"):
        st.markdown("\n".join(f"- {md(r)}" for r in repairs))
    if state.revisions:
        show(callout("The report was rewritten",
                     f"Validation returned it {_plural(state.revisions, 'time')} with "
                     f"objections, and it was recomposed with them in hand.", "warn"))

    show('<div class="section-label">Stage by stage</div>')
    timings = [{"#": f"{CANONICAL_ORDER.index(k) + 1:02d}" if k in CANONICAL_ORDER else "",
                "Stage": STAGES.get(k, (k,))[0], "Status": v.status,
                "Seconds": round(v.duration_s, 2), "Notes": len(v.warnings)}
               for k, v in state.outputs.items()]
    if timings:
        st.dataframe(pd.DataFrame(timings), use_container_width=True, hide_index=True)

    if providers := (state.usage or {}).get("by_provider"):
        st.caption("Interpretation written by: " + ", ".join(
            f"{name} ({_plural(count, 'call')})" for name, count in providers.items()))

# -- Agents ----------------------------------------------------------------------
with tabs[4]:
    for number, stage in enumerate(CANONICAL_ORDER, 1):
        output = state.outputs.get(stage)
        if output is None:
            continue
        with st.expander(f"{number:02d} · {STAGES[stage][0]} — {output.status}, "
                         f"{output.duration_s:.2f}s, {_plural(len(output.warnings), 'note')}"):
            if output.narrative:
                st.write(md(output.narrative))
            if output.error:
                st.error(output.error)
            if output.warnings:
                st.markdown("**Corrections and cautions recorded by this stage**")
                st.markdown("\n".join(f"- {md(w)}" for w in output.warnings))
            # Collapsed by default: the raw structure is what a developer
            # wants and what everyone else scrolls past. Not a nested
            # expander, which Streamlit forbids.
            st.caption("Raw computed output")
            st.json(output.findings, expanded=False)

# -- Export ----------------------------------------------------------------------
with tabs[5]:
    show('<div class="section-label">Download the report</div>')
    stem = Path(state.dataset_name).stem
    markdown = to_markdown(state)
    left, right = st.columns(2)
    left.download_button("Download Markdown", markdown, file_name=f"{stem}_report.md",
                         mime="text/markdown", use_container_width=True)
    try:
        page = to_html(state)
    except Exception as exc:  # the Markdown download must survive a render fault
        page = None
        right.caption(f"HTML export unavailable: {exc}")
    if page is not None:
        right.download_button("Download HTML", page, file_name=f"{stem}_report.html",
                              mime="text/html", use_container_width=True)
    if st.button("Save every format to the reports folder"):
        for kind, path in save_report(state).items():
            st.markdown(f"**{kind}** · `{path}`")
    with st.expander("Preview the Markdown"):
        st.code(markdown, language="markdown")

    # The fitted model is the other half of the deliverable: a report that
    # recommends a model without saying where it is leaves nothing to act on.
    show('<div class="section-label">Use the model</div>')
    if saved := modelling.get("saved_model"):
        st.markdown(
            f"The winning **{md(saved.get('model'))}** was refitted on every row and "
            f"saved. It takes rows shaped like the original file, with these columns:")
        st.code(", ".join(map(str, saved.get("expects_columns", []))), language="text")
        st.markdown("Score new rows from the command line:")
        st.code("python predict.py --latest your_new_rows.csv -o scored.csv",
                language="bash")
        st.caption(f"Saved to {saved.get('path')}")
    elif note := modelling.get("saved_model_note"):
        show(callout("No model was saved", note, "warn"))
    else:
        show(callout("No model was fitted",
                     "The modelling stage did not run for this dataset, so there "
                     "is nothing to save."))
