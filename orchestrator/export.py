"""Render a completed workflow as a readable report.

Markdown is the primary format because it is readable as plain text, converts to
anything, and can be pasted into a document without loss. HTML is produced from
the same content for viewing in a browser with the charts embedded.

The exporter adds no interpretation. It lays out what the agents established,
including the validator's objections: a report that hides its own review would
defeat the purpose of having produced one.
"""
from __future__ import annotations

import base64
import html
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from config import REPORTS_DIR
from agents.llm import as_text
from orchestrator import plain_language as pl
from state import WorkflowState


def _fmt_list(items: list[str], bullet: str = "-") -> str:
    return "\n".join(f"{bullet} {i}" for i in items if str(i).strip())


def _at_a_glance(state: WorkflowState, verdict: str | None,
                 counts: dict) -> list[str]:
    """The opening a reader who reads nothing else should still come away with.

    A report that begins with "eta-squared = 0.6031" has already lost the person
    who has to act on it. This states the question, the answer, how far to trust
    it and what it rests on, in words, before any statistic appears. The
    statistics follow immediately afterwards and are unchanged -- a specialist
    must still be able to check the work.
    """
    modelling = state.findings("modelling")
    profiling = state.findings("profiling")
    out: list[str] = ["## At a Glance\n"]

    rows = int(state.full_row_count or state.data.shape[0])
    target = state.target_column
    task = state.task_type

    if target and task in ("classification", "regression"):
        question = ("predict " if task == "classification" else "explain ")
        out.append(f"**The question:** using {rows:,} records, can we {question}"
                   f"`{target}` from the other information recorded?\n")
    else:
        out.append(f"**The question:** what does this dataset of {rows:,} records "
                   f"contain, and what patterns are in it?\n")

    # The answer, stated before any figure.
    if modelling and not modelling.get("skipped"):
        beats = modelling.get("beats_baseline")
        score = modelling.get("best_score")
        metric = modelling.get("primary_metric")
        model = modelling.get("best_model")
        if beats:
            meaning = pl.plain_score(metric, score)
            out.append(f"**The answer:** yes. A {model} predicts it reliably"
                       + (f" -- {meaning}" if meaning else "") + ".\n")
        else:
            out.append("**The answer:** no. Nothing in this data predicts the "
                       "outcome better than guessing would. That is a real "
                       "finding, not a failure of the analysis.\n")
        if significance := pl.plain_significance(modelling.get("permutation_p_value")):
            out.append(f"{significance}\n")

    # How far to trust it.
    if trust := pl.plain_verdict(verdict, counts):
        symbol = {"sound": "checks passed",
                  "needs_revision": "some points to confirm",
                  "unreliable": "not reliable"}.get(verdict, verdict or "")
        out.append(f"**How far to trust this ({symbol}):** {trust}\n")

    # What it rests on.
    n_features = len(profiling.get("usable_features") or []) or state.data.shape[1]
    caveats = [pl.plain_sample_size(rows, n_features)]
    balance = (state.findings("eda") or {}).get("target_balance") or {}
    if imbalance := pl.plain_imbalance(balance.get("imbalance_ratio")):
        caveats.append(imbalance)
    caveats = [c for c in caveats if c]
    if caveats:
        out.append("**What it rests on:** " + " ".join(caveats) + "\n")

    out.append("---\n")
    return out


def _metric_label(metric: str) -> str:
    """The reader's name for a metric, not scikit-learn's."""
    return {
        "f1_macro": "macro F1", "roc_auc": "ROC AUC", "r2": "R-squared",
        "precision_macro": "precision", "recall_macro": "recall",
        "accuracy": "accuracy", "rmse": "RMSE", "mae": "MAE",
    }.get(metric, metric.replace("_", " "))


def _evidence_tables(state: WorkflowState) -> list[str]:
    """The computed comparisons, as tables rather than a sentence about them.

    The modelling stage evaluates every candidate on five measures with a
    standard deviation each, and ranks every feature by permutation importance
    against a threshold for what counts as informative. All of that was
    reaching the report as a single line of prose naming the winner.

    A data scientist's report always shows the comparison, because the reader's
    question is not only "which model won" but "by how much, and was it close?"
    A leader half a standard deviation ahead of four rivals is a different
    finding from one clear of the field, and only the table distinguishes them.

    Rendered straight from the computed evidence and never passed through the
    model, so this adds depth without adding anywhere for an unsupported claim
    to enter.
    """
    out: list[str] = []
    modelling = state.outputs.get("modelling")
    if not modelling or not modelling.ok:
        return out

    evaluation = (modelling.evidence or {}).get("evaluation") or {}
    results = [r for r in evaluation.get("results", []) if r.get("status") == "ok"]

    if results:
        primary = evaluation.get("primary_metric", "")
        metrics = [k for k in results[0].get("scores", {})
                   if not k.endswith("_std")]
        metrics.sort(key=lambda m: (m != primary, m))

        out.append("## Model Comparison\n")
        out.append(
            "Every candidate was fitted on the same cross-validation folds and "
            "scored on the same measures, so the differences below are "
            "attributable to the models rather than to how each was tested. "
            "The figure after each score is the standard deviation across those "
            "folds: where two models sit within one of these of each other, the "
            "data does not really distinguish them.\n")

        head = "| Model | " + " | ".join(
            _metric_label(m) + (" *(selected on)*" if m == primary else "")
            for m in metrics) + " |"
        out.append(head)
        out.append("|" + "---|" * (len(metrics) + 1))
        for row in sorted(results,
                          key=lambda r: -(r.get("scores", {}).get(primary) or 0)):
            cells = []
            for m in metrics:
                value = row.get("scores", {}).get(m)
                std = row.get("scores", {}).get(m + "_std")
                if value is None:
                    cells.append("-")
                elif std is None:
                    cells.append(f"{value:.4f}")
                else:
                    cells.append(f"{value:.4f} +/- {std:.4f}")
            name = str(row.get("model", "?"))
            if row.get("tuned"):
                name += " *(tuned)*"
            if row.get("is_baseline"):
                name = "*" + name + "*"
            out.append("| " + name + " | " + " | ".join(cells) + " |")
        out.append("")
        out.append(
            "Baselines are shown in italics. They are deliberately "
            "unintelligent models, and every other row has to be read against "
            "them: a score means nothing until you know what guessing would "
            "have scored.\n")

    importance = (modelling.evidence or {}).get("feature_importance") or {}
    ranked = importance.get("importances") or []
    if ranked:
        out.append("## What the Model Relies On\n")
        out.append(
            f"Each feature was shuffled in turn and the model re-scored on "
            f"{importance.get('test_rows', 0):,} held-out rows. The drop in "
            f"score is how much the model depended on that feature. This is "
            f"measured on the fitted model rather than inferred from the "
            f"correlations, so a feature can matter here while looking "
            f"unremarkable in the exploratory analysis, and the reverse.\n")
        out.append("| Feature | Drop in score when shuffled | Informative |")
        out.append("|---|---|---|")
        for f in ranked[:15]:
            value, std = f.get("importance"), f.get("std")
            shown = ("-" if value is None
                     else f"{value:.4f}"
                          + (f" +/- {std:.4f}" if std is not None else ""))
            out.append("| " + str(f.get("feature", "?")) + " | " + shown + " | "
                       + ("yes" if f.get("informative") else "no") + " |")
        out.append("")
        unhelpful = [f for f in ranked if not f.get("informative")]
        if unhelpful:
            out.append(
                f"{len(unhelpful)} of {len(ranked)} features are marked not "
                f"informative: shuffling them changed the score by no more than "
                f"the variation between runs. That is not evidence they hold "
                f"nothing -- a feature can carry information this model cannot "
                f"use -- but nothing measured here shows that they do.\n")

    return out


#: Which computed block belongs after which of the model's own sections. The
#: model writes a narrative per stage; the framework computes the evidence for
#: some of those stages. Appending the evidence at the end of the document, as
#: this previously did, scattered the modelling material across four
#: non-adjacent sections with the charts wedged between them -- the reader met
#: the conclusion, then unrelated pictures, then the numbers the conclusion
#: rested on. Each block is now placed directly after the narrative it belongs
#: to, which is where a reader looks for it.
#:
#: Matching is on a keyword in the heading rather than an exact title, because
#: the heading is written by the model and varies ("Modelling", "Modeling
#: Results", "Model Performance").
EVIDENCE_AFTER: tuple[tuple[str, str], ...] = (
    ("model", "modelling"),
    ("visualis", "charts"),
    ("visualiz", "charts"),
)


def _blocks_for(name: str, state: WorkflowState) -> list[str]:
    """The computed block of a given kind, ready to render."""
    if name == "modelling":
        out = list(pl.practical_behaviour(
            state.findings("modelling").get("operating_behaviour"),
            state.target_column))
        out.extend(_evidence_tables(state))
        return out
    if name == "charts":
        return _chart_block(state)
    return []


def _chart_block(state: WorkflowState) -> list[str]:
    """The charts, each with what it shows and what to look for in it."""
    charts = state.findings("visualization").get("charts", [])
    if not charts:
        return []

    lines = ["## The Evidence, in Charts" + chr(10)]
    lines.append(
        "Each chart is followed by what it shows and what to look for in it. "
        "A caption that only names the chart tells a reader what they can "
        "already see; what is usually less obvious is which part of it carries "
        "the finding." + chr(10))
    explained: set[str] = set()
    for chart in charts:
        lines.append(f"### {chart['title']}" + chr(10))
        if chart.get("path"):
            rel = Path(chart["path"]).name
            lines.append(f"![{chart['title']}](charts/{rel})" + chr(10))
        lines.append(f"{chart['description']}" + chr(10))
        kind = chart.get("kind", "")
        if kind not in explained:
            if guidance := pl.chart_guidance(kind, chart):
                lines.append("**What to look for:** " + guidance + chr(10))
                explained.add(kind)
    return lines


def _sections_with_evidence(report: dict, state: WorkflowState) -> list[str]:
    """The model's sections, each followed by the evidence it rests on.

    Any block whose anchor section the model did not write is emitted at the
    end rather than dropped: a computed comparison must not disappear because
    of how the narrative happened to be titled.
    """
    lines: list[str] = []
    placed: set[str] = set()

    for section in report.get("sections", []):
        if not isinstance(section, dict):
            lines.append(as_text(section) + chr(10))
            continue
        heading = str(section.get("heading", "Section"))
        lines.append(f"## {heading}" + chr(10))
        lines.append(as_text(section.get("body")) + chr(10))

        lowered = heading.lower()
        for keyword, block in EVIDENCE_AFTER:
            if keyword in lowered and block not in placed:
                lines.extend(_blocks_for(block, state))
                placed.add(block)
                break

    for _, block in EVIDENCE_AFTER:
        if block not in placed:
            lines.extend(_blocks_for(block, state))
            placed.add(block)

    return lines


def to_markdown(state: WorkflowState, include_appendix: bool = True) -> str:
    """Render the full report as Markdown."""
    report = state.findings("report")
    validation = state.findings("validation")
    lines: list[str] = []

    title = report.get("title") or f"Analysis of {state.dataset_name}"
    lines.append(f"# {title}\n")
    lines.append(
        f"*Generated {datetime.now():%d %B %Y at %H:%M} by the Multi-Agent Data "
        f"Science framework.*\n"
    )

    # The verdict belongs at the top: a reader must know how far to trust what
    # follows before reading it, not after.
    verdict = validation.get("verdict")
    counts = validation.get("issue_counts", {})

    lines.extend(_at_a_glance(state, verdict, counts))

    if summary := report.get("executive_summary"):
        lines.append("## Executive Summary\n")
        lines.append(f"{summary}\n")

    if takeaways := report.get("key_takeaways"):
        lines.append("## Key Takeaways\n")
        lines.append(_fmt_list(takeaways) + "\n")

    # Sections are model-shaped: the model returns them as objects, but has
    # also returned bare strings. Indexing one raised AttributeError inside
    # `save_report`, after the whole analysis had completed -- minutes of
    # work lost at the final step, which is the failure class
    # `_make_console_unicode_safe` exists to prevent, arriving by a
    # different route.
    # Each stage narrative followed by the evidence for that stage, rather
    # than every computed block appended after the document, so the reader
    # meets a conclusion and the numbers behind it together.
    lines.extend(_sections_with_evidence(report, state))

    if limitations := report.get("limitations"):
        lines.append("## Limitations\n")
        lines.append(_fmt_list(limitations) + "\n")

    if steps := report.get("recommended_next_steps"):
        lines.append("## Recommended Next Steps\n")
        lines.append(_fmt_list(steps) + "\n")

    # The validator's objections are part of the deliverable.
    issues = validation.get("issues", [])
    if issues:
        lines.append("## What the Automated Review Flagged\n")
        lines.append(
            "Before this report was finalised, an automated reviewer checked it "
            "against the calculations behind it. Anything it could not "
            "reconcile is listed here rather than quietly removed, so a reader "
            "can judge it independently. These are points to confirm, not "
            "necessarily errors.\n")
        lines.append("| Severity | Location | Issue | Suggested fix |")
        lines.append("|---|---|---|---|")
        for issue in issues:
            lines.append(
                f"| {issue.get('severity','')} | {issue.get('location','')} | "
                f"{str(issue.get('problem','')).replace('|','/')} | "
                f"{str(issue.get('suggested_fix','')).replace('|','/')} |"
            )
        lines.append("")

    # Only the terms this particular report actually used. A glossary
    # listing everything would bury the handful that matter.
    if entries := pl.glossary_for(chr(10).join(lines)):
        lines.append("## Terms Used in This Report" + chr(10))
        for term, definition in entries:
            lines.append("**" + term + "** - " + definition + chr(10))

    if include_appendix:
        lines.append("---\n")
        lines.append("## Appendix: How This Analysis Was Produced\n")

        planning = state.planning or {}
        # The execution engine is chosen automatically from the size of the
        # data, so the report states which one ran and why. On a distributed
        # run it also states which stages saw the full dataset and which saw
        # a sample, because that distinction changes how the numbers should
        # be read.
        if state.engine == "spark":
            lines.append(
                f"**Execution engine:** Apache Spark. {state.engine_reason}\n")
            lines.append(
                f"Profiling and exploratory statistics were computed across all "
                f"{state.full_row_count:,} rows. Charting and model evaluation ran "
                f"on an in-memory sample of {len(state.data):,} rows, so model "
                f"scores describe the sample rather than every row.\n")
        else:
            lines.append(f"**Execution engine:** pandas, in-memory. "
                         f"{state.engine_reason}\n")

        lines.append(f"**Workflow plan:** {' -> '.join(state.plan)}\n")
        if reasoning := planning.get("reasoning"):
            lines.append(f"**Coordinator's reasoning:** {reasoning}\n")
        if skipped := planning.get("skipped"):
            lines.append("**Stages skipped:**\n")
            lines.append(_fmt_list(
                [f"`{s.get('stage')}` - {s.get('reason')}" for s in skipped
                 if isinstance(s, dict)]) + "\n")
        if state.revisions:
            lines.append(
                f"**Revisions:** the report was returned for revision "
                f"{state.revisions} time(s) by the validation stage.\n")

        lines.append("**Stage execution:**\n")
        lines.append("| Stage | Status | Duration (s) | Warnings |")
        lines.append("|---|---|---|---|")
        for name in state.completed:
            output = state.outputs[name]
            lines.append(f"| {name} | {output.status} | {output.duration_s:.2f} | "
                         f"{len(output.warnings)} |")
        lines.append("")

        if usage := state.usage:
            lines.append(
                f"**Computational cost:** {usage.get('calls', 0)} model calls, "
                f"{usage.get('total_tokens', 0):,} tokens, "
                f"{usage.get('latency_s', 0):.1f}s of model latency.\n"
            )
            # Which provider answered, not which was configured. On a
            # failover chain those differ: the setting says "auto" while the
            # narrative was written by whichever provider had allowance at
            # the time. A report naming only the setting would be vaguer
            # about its own provenance than the run actually was.
            if providers := usage.get("by_provider"):
                named = ", ".join(
                    f"{name} ({count} call{'s' if count != 1 else ''})"
                    for name, count in sorted(providers.items(),
                                              key=lambda kv: -kv[1]))
                lines.append(f"**Interpretation written by:** {named}.\n")

        # Warnings raised by agents are surfaced rather than left in the log,
        # since several of them describe corrections applied to model output.
        all_warnings = [
            (name, w) for name in state.completed
            for w in state.outputs[name].warnings
        ]
        if all_warnings:
            lines.append("**Agent warnings:**\n")
            lines.append(_fmt_list([f"`{n}`: {w}" for n, w in all_warnings]) + "\n")

    return "\n".join(lines)


def _glance_html(state: WorkflowState, verdict: str | None,
                 counts: dict, esc) -> str:
    """The plain-language opening, as a card rather than another paragraph.

    The Markdown report gained this first; the HTML is the file people actually
    open, and for a while it did not have it at all -- the two renderers build
    independently, so an addition to one silently skips the other. Both now draw
    on the same `plain_language` helpers, which is the only way to keep them
    saying the same thing.
    """
    modelling = state.findings("modelling")
    profiling = state.findings("profiling")
    rows = int(state.full_row_count or state.data.shape[0])
    target, task = state.target_column, state.task_type

    items: list[str] = []

    if target and task in ("classification", "regression"):
        verb = "predict" if task == "classification" else "explain"
        question = (f"Using {rows:,} records, can we {verb} "
                    f"<code>{esc(str(target))}</code> from the other information "
                    f"recorded?")
    else:
        question = (f"What does this dataset of {rows:,} records contain, and "
                    f"what patterns are in it?")
    items.append(f"<dt>The question</dt><dd>{question}</dd>")

    if modelling and not modelling.get("skipped"):
        if modelling.get("beats_baseline"):
            meaning = pl.plain_score(modelling.get("primary_metric"),
                                     modelling.get("best_score"))
            answer = (f"Yes &mdash; a {esc(str(modelling.get('best_model')))} "
                      f"predicts it reliably."
                      + (f" In practical terms, {esc(meaning)}." if meaning else ""))
        else:
            answer = ("No &mdash; nothing in this data predicts the outcome better "
                      "than guessing would. That is a real finding, not a failure "
                      "of the analysis.")
        items.append(f'<dt>The answer</dt><dd class="answer">{answer}</dd>')

        if sig := pl.plain_significance(modelling.get("permutation_p_value")):
            items.append(f'<dt>How it was tested</dt><dd>{esc(sig)}</dd>')

    if trust := pl.plain_verdict(verdict, counts):
        label = {"sound": "Checks passed",
                 "needs_revision": "Some points to confirm",
                 "unreliable": "Not reliable"}.get(verdict, verdict or "")
        items.append(f"<dt>How far to trust this &mdash; {esc(label)}</dt>"
                     f"<dd>{esc(trust)}</dd>")

    n_features = len(profiling.get("usable_features") or []) or state.data.shape[1]
    caveats = [pl.plain_sample_size(rows, n_features)]
    balance = (state.findings("eda") or {}).get("target_balance") or {}
    if imbalance := pl.plain_imbalance(balance.get("imbalance_ratio")):
        caveats.append(imbalance)
    caveats = [c for c in caveats if c]
    if caveats:
        items.append(f"<dt>What it rests on</dt><dd>{esc(' '.join(caveats))}</dd>")

    return ('<section class="glance"><h2>At a Glance</h2><dl>'
            + "".join(items) + "</dl></section>")


def _markdown_block_to_html(lines: list[str], esc) -> str:
    """Render the shared plain-language blocks as HTML.

    The plain-language helpers emit Markdown so the two renderers cannot say
    different things. This converts the small subset they use -- a heading,
    paragraphs, and **bold** or `code` inline -- rather than pulling in a
    Markdown library for four constructs.
    """
    import re as _re

    def inline(text: str) -> str:
        out = esc(text)
        out = _re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
        out = _re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
        return out

    html: list[str] = []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        if text.startswith("## "):
            html.append(f"<h2>{esc(text[3:].strip())}</h2>")
        else:
            html.append(f"<p>{inline(text)}</p>")
    return "".join(html)


def _evidence_tables_html(state: WorkflowState, esc) -> str:
    """The same tables as the Markdown report, from the same builder.

    Written as a converter over `_evidence_tables` rather than as a second
    renderer. These two have drifted on this project every single time an
    addition was made to one of them by hand, always in the same direction --
    the Markdown gained something and the HTML, which is the file people
    actually open, quietly did not.
    """
    html: list[str] = []
    in_table = False
    for line in _evidence_tables(state):
        text = line.strip()
        if not text:
            if in_table:
                html.append("</tbody></table>")
                in_table = False
            continue
        if text.startswith("## "):
            if in_table:
                html.append("</tbody></table>")
                in_table = False
            html.append(f"<h2>{esc(text[3:].strip())}</h2>")
        elif set(text) <= set("|-"):
            continue  # the Markdown separator row has no HTML equivalent
        elif text.startswith("|"):
            cells = [c.strip() for c in text.strip("|").split("|")]
            if not in_table:
                html.append('<table class="evidence"><thead><tr>'
                            + "".join(f"<th>{_inline_html(c, esc)}</th>"
                                      for c in cells)
                            + "</tr></thead><tbody>")
                in_table = True
            else:
                html.append("<tr>" + "".join(f"<td>{_inline_html(c, esc)}</td>"
                                             for c in cells) + "</tr>")
        else:
            html.append(f"<p>{_inline_html(text, esc)}</p>")
    if in_table:
        html.append("</tbody></table>")
    return "".join(html)


def _inline_html(text: str, esc) -> str:
    """Escape, then restore the only markup these tables use: *emphasis*."""
    import re as _re

    return _re.sub(r"\*([^*]+)\*", r"<em>\1</em>", esc(text))


def _chart_block_html(charts: list, esc) -> str:
    """The charts as figures, with images embedded so the file stands alone."""
    if not charts:
        return ""
    parts = ['<h2>The Evidence, in Charts</h2>',
             '<p class="note">Each chart is followed by what it shows and, for '
             'each kind of chart, what to look for in it.</p>']
    explained: set[str] = set()
    for chart in charts:
        parts.append(f"<figure><h3>{esc(chart['title'])}</h3>")
        path = chart.get("path")
        if path and Path(path).exists():
            # Data URIs so the file is self-contained and can be sent as one
            # attachment.
            data = base64.b64encode(Path(path).read_bytes()).decode()
            parts.append(f'<img src="data:image/png;base64,{data}" '
                         f'alt="{esc(chart["title"])}">')
        parts.append(f"<figcaption>{esc(chart['description'])}</figcaption>")
        kind = chart.get("kind", "")
        if kind not in explained:
            if guidance := pl.chart_guidance(kind, chart):
                parts.append(f'<p class="look-for"><strong>What to look for:'
                             f'</strong> {esc(guidance)}</p>')
                explained.add(kind)
        parts.append("</figure>")
    return "".join(parts)


def _blocks_for_html(name: str, state: WorkflowState, esc, charts: list) -> str:
    """The HTML form of a computed block, from the same source as the Markdown."""
    if name == "modelling":
        return (_markdown_block_to_html(
                    pl.practical_behaviour(
                        state.findings("modelling").get("operating_behaviour"),
                        state.target_column), esc)
                + _evidence_tables_html(state, esc))
    if name == "charts":
        return _chart_block_html(charts, esc)
    return ""


def _sections_with_evidence_html(report: dict, state: WorkflowState, esc,
                                 charts: list) -> str:
    """The HTML counterpart of `_sections_with_evidence`.

    Placement is driven by the same `EVIDENCE_AFTER` table, so the two
    renderers cannot order the document differently.
    """
    parts: list[str] = []
    placed: set[str] = set()

    for section in report.get("sections", []):
        if not isinstance(section, dict):
            parts.append(f"<p>{esc(as_text(section))}</p>")
            continue
        heading = str(section.get("heading", "Section"))
        parts.append(f"<h2>{esc(heading)}</h2>"
                     f"<p>{esc(as_text(section.get('body')))}</p>")

        lowered = heading.lower()
        for keyword, block in EVIDENCE_AFTER:
            if keyword in lowered and block not in placed:
                parts.append(_blocks_for_html(block, state, esc, charts))
                placed.add(block)
                break

    for _, block in EVIDENCE_AFTER:
        if block not in placed:
            parts.append(_blocks_for_html(block, state, esc, charts))
            placed.add(block)

    return "".join(parts)


def to_html(state: WorkflowState) -> str:
    """Render the report as a self-contained HTML page with charts embedded."""
    report = state.findings("report")
    validation = state.findings("validation")
    charts = state.findings("visualization").get("charts", [])

    def esc(text: Any) -> str:
        return html.escape(str(text))

    verdict = validation.get("verdict", "")
    verdict_colour = {"sound": "#059669", "needs_revision": "#d97706",
                      "unreliable": "#dc2626"}.get(verdict, "#6b7280")

    parts = [f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(report.get('title', state.dataset_name))}</title>
<style>
  :root {{ --ink:#1f2937; --muted:#6b7280; --line:#e5e7eb; --bg:#ffffff; }}
  * {{ box-sizing:border-box; }}
  body {{ font:16px/1.65 Inter,'Segoe UI',system-ui,sans-serif; color:var(--ink);
         background:var(--bg); max-width:60rem; margin:0 auto; padding:2.5rem 1.5rem; }}
  h1 {{ font-size:2rem; line-height:1.2; margin:0 0 .4rem; letter-spacing:-.02em; }}
  h2 {{ font-size:1.3rem; margin:2.4rem 0 .8rem; padding-bottom:.4rem;
        border-bottom:1px solid var(--line); letter-spacing:-.01em; }}
  h3 {{ font-size:1.05rem; margin:1.6rem 0 .5rem; }}
  .meta {{ color:var(--muted); font-size:.875rem; margin-bottom:1.5rem; }}
  .verdict {{ display:inline-block; padding:.45rem .85rem; border-radius:.4rem;
              background:{verdict_colour}14; color:{verdict_colour};
              border:1px solid {verdict_colour}44; font-weight:600; font-size:.875rem; }}
  .summary {{ background:#f9fafb; border-left:3px solid #2563eb; padding:1rem 1.25rem;
              border-radius:0 .4rem .4rem 0; margin:1rem 0; }}
  /* The opening block is the only part many readers will read, so it is set
     apart from the body text rather than running into it. */
  .glance {{ border:1px solid var(--line); border-radius:.6rem; padding:1.5rem 1.75rem;
             margin:1.5rem 0 2.5rem; background:linear-gradient(#fcfcfd,#f7f8fa); }}
  .glance h2 {{ margin:0 0 1rem; border:0; padding:0; font-size:1.05rem;
                text-transform:uppercase; letter-spacing:.06em; color:var(--muted); }}
  .glance dl {{ margin:0; }}
  .glance dt {{ font-weight:650; font-size:.8rem; text-transform:uppercase;
                letter-spacing:.05em; color:var(--muted); margin-top:1rem; }}
  .glance dt:first-child {{ margin-top:0; }}
  .glance dd {{ margin:.3rem 0 0; font-size:1.02rem; }}
  .glance .answer {{ font-size:1.2rem; font-weight:600; line-height:1.45; }}
  .look-for {{ background:#fffbeb; border-left:3px solid #f59e0b; padding:.7rem 1rem;
               border-radius:0 .3rem .3rem 0; font-size:.9rem; margin:.6rem 0 0; }}
  .note {{ color:var(--muted); font-size:.9rem; }}
  /* Wide comparison tables scroll inside their own box rather than forcing
     the page sideways on a narrow screen. */
  table.evidence {{ display:block; overflow-x:auto; white-space:nowrap;
                    border-collapse:collapse; margin:1rem 0; font-size:.88rem; }}
  table.evidence th, table.evidence td {{ padding:.45rem .7rem;
                    border-bottom:1px solid var(--line); text-align:right; }}
  table.evidence th:first-child, table.evidence td:first-child {{ text-align:left; }}
  table.evidence thead th {{ font-weight:650; color:var(--muted);
                    font-size:.78rem; text-transform:uppercase;
                    letter-spacing:.04em; }}
  table {{ border-collapse:collapse; width:100%; font-size:.9rem; margin:1rem 0; }}
  th,td {{ text-align:left; padding:.55rem .7rem; border-bottom:1px solid var(--line);
           vertical-align:top; }}
  th {{ font-weight:600; background:#f9fafb; }}
  figure {{ margin:1.5rem 0; }}
  figure img {{ width:100%; border:1px solid var(--line); border-radius:.4rem; }}
  figcaption {{ color:var(--muted); font-size:.875rem; margin-top:.5rem; }}
  .sev-critical {{ color:#dc2626; font-weight:600; }}
  .sev-warning {{ color:#d97706; font-weight:600; }}
  .sev-info {{ color:var(--muted); }}
  footer {{ margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--line);
            color:var(--muted); font-size:.875rem; }}
</style></head><body>
<h1>{esc(report.get('title', 'Analysis'))}</h1>
<div class="meta">Generated {datetime.now():%d %B %Y at %H:%M} &middot;
  {esc(state.dataset_name)} &middot; {int(state.full_row_count or state.data.shape[0]):,} rows &times;
  {state.data.shape[1]} columns</div>"""]

    counts = validation.get("issue_counts", {})
    parts.append(_glance_html(state, verdict, counts, esc))

    if summary := report.get("executive_summary"):
        parts.append(f'<h2>Executive Summary</h2><div class="summary">{esc(summary)}</div>')

    if takeaways := report.get("key_takeaways"):
        parts.append("<h2>Key Takeaways</h2><ul>"
                     + "".join(f"<li>{esc(t)}</li>" for t in takeaways) + "</ul>")

    # The same guard as the Markdown renderer. These two have drifted often
    # enough on this project that adding to one and not the other is the
    # expected mistake rather than an unlucky one.
    # The same interleaving as the Markdown, built from the same helpers.
    # Appending every computed block after the document split the modelling
    # material across four non-adjacent sections with the charts wedged
    # between them; each block now follows the narrative it belongs to.
    parts.append(_sections_with_evidence_html(report, state, esc, charts))

    if limitations := report.get("limitations"):
        parts.append("<h2>Limitations</h2><ul>"
                     + "".join(f"<li>{esc(l)}</li>" for l in limitations) + "</ul>")

    if steps := report.get("recommended_next_steps"):
        parts.append("<h2>Recommended Next Steps</h2><ul>"
                     + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ul>")

    if issues := validation.get("issues", []):
        parts.append('<h2>What the Automated Review Flagged</h2>'
                     '<p class="note">Before this report was finalised, an '
                     'automated reviewer checked it against the calculations '
                     'behind it. Anything it could not reconcile is listed here '
                     'rather than quietly removed. These are points to confirm, '
                     'not necessarily errors.</p><table>'
                     "<tr><th>Severity</th><th>Location</th><th>Issue</th>"
                     "<th>Suggested fix</th></tr>")
        for issue in issues:
            sev = esc(issue.get("severity", "info"))
            parts.append(
                f'<tr><td class="sev-{sev}">{sev}</td><td>{esc(issue.get("location",""))}</td>'
                f'<td>{esc(issue.get("problem",""))}</td>'
                f'<td>{esc(issue.get("suggested_fix",""))}</td></tr>')
        parts.append("</table>")

    # The terms this report actually used, defined once at the end rather
    # than interrupting every sentence that mentions one.
    if entries := pl.glossary_for(" ".join(parts)):
        parts.append("<h2>Terms Used in This Report</h2><dl>")
        for term, definition in entries:
            parts.append("<dt><strong>" + esc(term) + "</strong></dt>"
                         "<dd>" + esc(definition) + "</dd>")
        parts.append("</dl>")

    parts.append("<h2>How This Analysis Was Produced</h2>")
    parts.append(f"<p><strong>Workflow plan:</strong> {esc(' → '.join(state.plan))}</p>")
    if reasoning := (state.planning or {}).get("reasoning"):
        parts.append(f"<p><strong>Coordinator's reasoning:</strong> {esc(reasoning)}</p>")
    if state.revisions:
        parts.append(f"<p><strong>Revisions:</strong> the report was returned for "
                     f"revision {state.revisions} time(s).</p>")

    parts.append("<table><tr><th>Stage</th><th>Status</th><th>Duration</th>"
                 "<th>Warnings</th></tr>")
    for name in state.completed:
        o = state.outputs[name]
        parts.append(f"<tr><td>{esc(name)}</td><td>{esc(o.status)}</td>"
                     f"<td>{o.duration_s:.2f}s</td><td>{len(o.warnings)}</td></tr>")
    parts.append("</table>")


    usage = state.usage or {}
    # Named in the HTML footer too. These two renderers have drifted three
    # times on this project, always the same way: an addition to the
    # Markdown that the HTML silently lacked -- in the file people open.
    provider_note = ""
    if providers := usage.get("by_provider"):
        provider_note = (" &middot; via "
                         + ", ".join(sorted(providers,
                                            key=lambda k: -providers[k])))
    parts.append(
        f'<footer>{usage.get("calls",0)} model calls{provider_note} &middot; '
        f'{usage.get("total_tokens",0):,} tokens &middot; '
        f'completed in {state.summary()["elapsed_s"]:.1f}s'
        f'</footer></body></html>')

    return "".join(parts)


def _copy_charts(state: WorkflowState, directory: Path) -> None:
    """Place the run's charts beside the report that links them."""
    charts = state.findings("visualization").get("charts", [])
    sources = [Path(c["path"]) for c in charts if c.get("path")]
    if not sources:
        return

    destination = directory / "charts"
    for source in sources:
        if not source.exists():
            continue
        target = destination / source.name
        if source.resolve() == target.resolve():
            continue  # already written here
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def save_report(state: WorkflowState, directory: Path | None = None,
                stem: str | None = None) -> dict[str, str]:
    """Write the report as Markdown, HTML and a machine-readable run record."""
    directory = Path(directory or REPORTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    stem = stem or f"{Path(state.dataset_name).stem}_{datetime.now():%Y%m%d_%H%M%S}"
    written: dict[str, str] = {}

    # The Markdown links charts as `charts/<name>.png`, relative to the report.
    # Charts are written to the default reports directory when they are built,
    # so a report written anywhere else pointed at a `charts/` folder that did
    # not exist and every image was broken -- silently, since a missing image in
    # Markdown renders as nothing rather than as an error. `--output` should
    # produce a self-contained directory, so the charts travel with the report.
    _copy_charts(state, directory)

    md_path = directory / f"{stem}.md"
    md_path.write_text(to_markdown(state), encoding="utf-8")
    written["markdown"] = str(md_path)

    html_path = directory / f"{stem}.html"
    html_path.write_text(to_html(state), encoding="utf-8")
    written["html"] = str(html_path)

    # The full run record, for the evaluation harness and for reproducing a run.
    json_path = directory / f"{stem}_run.json"
    json_path.write_text(
        json.dumps(state.summary(), indent=2, default=str), encoding="utf-8")
    written["json"] = str(json_path)

    return written
