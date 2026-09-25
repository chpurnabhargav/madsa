"""Chart generation.

Each function builds one chart and returns it together with the numbers behind
it. That pairing matters for the framework: a chart in the final report is a
claim about the data, and the Validation Agent can only check the claim if the
values plotted are available alongside the image.

Charts are written as PNG for the report and returned as Plotly figures for the
interactive interface, so both consume the same code path and cannot disagree.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import CONFIG

# A restrained qualitative palette, drawn from the interface's own tokens so a
# chart and the page around it read as one document. Blue and vermilion lead
# because they stay distinct under every common form of colour blindness.
PALETTE = ["#2346A0", "#C2491D", "#1F7F8C", "#C8891F", "#6D4AA8", "#1E6B4B"]

LAYOUT = dict(
    template="plotly_white",
    font=dict(family="IBM Plex Sans, Segoe UI, system-ui, sans-serif", size=13, color="#0F1B2D"),
    title=dict(font=dict(size=16, color="#0F1B2D"), x=0.0, xanchor="left"),
    margin=dict(l=60, r=30, t=60, b=55),
    colorway=PALETTE,
    plot_bgcolor="white",
    paper_bgcolor="white",
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:60]


def _style(fig: go.Figure, title: str, x: str = "", y: str = "") -> go.Figure:
    fig.update_layout(**LAYOUT, title_text=title, showlegend=fig.data and len(fig.data) > 1)
    fig.update_xaxes(title_text=x, showgrid=False, linecolor="#D8DEE6")
    fig.update_yaxes(title_text=y, gridcolor="#EEF1F5", linecolor="#D8DEE6")
    return fig


class ChartResult(dict):
    """A chart plus the values it displays and a factual description of them."""


def _result(chart_id: str, kind: str, title: str, fig: go.Figure,
            data: dict[str, Any], description: str,
            columns: list[str]) -> ChartResult:
    return ChartResult(
        id=chart_id, kind=kind, title=title, figure=fig,
        data=data, description=description, columns=columns, path=None,
    )


# --------------------------------------------------------------------------- #
# Chart builders
# --------------------------------------------------------------------------- #

def histogram(df: pd.DataFrame, column: str, bins: int = 30) -> ChartResult | None:
    """Distribution of one numeric column, with the mean and median marked."""
    s = df[column].dropna()
    if len(s) < 5 or s.nunique() < 3:
        return None

    fig = px.histogram(s, x=column, nbins=bins, color_discrete_sequence=[PALETTE[0]])
    mean, median = float(s.mean()), float(s.median())

    # Marking both makes skew visible: they separate when a tail pulls the mean.
    fig.add_vline(x=median, line_dash="solid", line_color="#0F1B2D", line_width=2,
                  annotation_text=f"median {median:,.2f}", annotation_position="top")
    fig.add_vline(x=mean, line_dash="dash", line_color="#C2491D", line_width=2,
                  annotation_text=f"mean {mean:,.2f}", annotation_position="top right")

    skew = float(s.skew())
    shape = ("right-skewed" if skew > 0.5 else
             "left-skewed" if skew < -0.5 else "roughly symmetric")

    return _result(
        f"hist_{_slug(column)}", "histogram", f"Distribution of {column}",
        _style(fig, f"Distribution of {column}", column, "Count"),
        {"mean": round(mean, 4), "median": round(median, 4),
         "std": round(float(s.std()), 4), "skewness": round(skew, 4),
         "min": round(float(s.min()), 4), "max": round(float(s.max()), 4), "n": int(len(s))},
        f"{column} is {shape} (skewness {skew:.2f}) with a median of {median:,.2f} "
        f"and a mean of {mean:,.2f} across {len(s):,} values.",
        [column],
    )


def bar_counts(df: pd.DataFrame, column: str, top_n: int = 12) -> ChartResult | None:
    """Frequency of each category."""
    counts = df[column].value_counts().head(top_n)
    if len(counts) < 2:
        return None

    total = int(df[column].notna().sum())
    fig = px.bar(x=[str(i) for i in counts.index], y=counts.values,
                 color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(
        text=[f"{v:,}" for v in counts.values], textposition="outside",
        hovertemplate="%{x}: %{y:,}<extra></extra>",
    )

    top_pct = 100 * counts.iloc[0] / total
    return _result(
        f"bar_{_slug(column)}", "bar", f"Counts by {column}",
        _style(fig, f"Counts by {column}", column, "Count"),
        {"counts": {str(k): int(v) for k, v in counts.items()}, "total": total},
        f"{column} is dominated by '{counts.index[0]}' with {int(counts.iloc[0]):,} "
        f"records ({top_pct:.1f}% of {total:,}) across {df[column].nunique()} categories.",
        [column],
    )


def grouped_rate(df: pd.DataFrame, feature: str, target: str,
                 top_n: int = 10) -> ChartResult | None:
    """Rate of each target class within each level of a categorical feature.

    Plotted as percentages rather than counts, because a raw count chart mostly
    shows which groups are large and hides the difference in rates that is the
    actual finding.
    """
    sub = df[[feature, target]].dropna()
    if len(sub) < 20:
        return None

    # The target must be categorical. Crossing a feature against a continuous
    # column produces one column per distinct value, which is unreadable as a
    # chart and enormous as data.
    if sub[target].nunique() > 20:
        return None

    keep = sub[feature].value_counts().head(top_n).index
    sub = sub[sub[feature].isin(keep)]
    table = pd.crosstab(sub[feature], sub[target])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None

    pct = (table.div(table.sum(axis=1), axis=0) * 100).round(1)

    fig = go.Figure()
    for i, cls in enumerate(pct.columns):
        fig.add_bar(name=str(cls), x=[str(i) for i in pct.index], y=pct[cls],
                    marker_color=PALETTE[i % len(PALETTE)],
                    text=[f"{v:.1f}%" for v in pct[cls]], textposition="inside",
                    hovertemplate=f"%{{x}} - {cls}: %{{y:.1f}}%<extra></extra>")
    fig.update_layout(barmode="stack")

    first = pct.columns[-1]
    hi, lo = pct[first].idxmax(), pct[first].idxmin()
    return _result(
        f"rate_{_slug(feature)}_{_slug(target)}", "grouped_rate",
        f"{target} rate by {feature}",
        _style(fig, f"{target} rate by {feature}", feature, "Percentage of group"),
        {"percentages": {str(i): {str(c): float(pct.loc[i, c]) for c in pct.columns}
                         for i in pct.index},
         "counts": {str(i): {str(c): int(table.loc[i, c]) for c in table.columns}
                    for i in table.index}},
        f"The rate of {target}={first} ranges from {pct[first].min():.1f}% for "
        f"{feature}='{lo}' to {pct[first].max():.1f}% for {feature}='{hi}'.",
        [feature, target],
    )


def box_by_group(df: pd.DataFrame, numeric: str, group: str,
                 max_groups: int = 8) -> ChartResult | None:
    """Distribution of a numeric column across groups, including spread and outliers."""
    sub = df[[numeric, group]].dropna()
    if len(sub) < 20:
        return None

    keep = sub[group].value_counts().head(max_groups).index
    sub = sub[sub[group].isin(keep)]
    if sub[group].nunique() < 2:
        return None

    fig = px.box(sub, x=group, y=numeric, color=group,
                 color_discrete_sequence=PALETTE, points=False)
    fig.update_layout(showlegend=False)

    medians = sub.groupby(group)[numeric].median().sort_values(ascending=False)
    return _result(
        f"box_{_slug(numeric)}_{_slug(group)}", "box",
        f"{numeric} by {group}",
        _style(fig, f"{numeric} by {group}", group, numeric),
        {"medians": {str(k): round(float(v), 4) for k, v in medians.items()},
         "means": {str(k): round(float(v), 4)
                   for k, v in sub.groupby(group)[numeric].mean().items()},
         "counts": {str(k): int(v) for k, v in sub[group].value_counts().items()}},
        f"Median {numeric} is highest for {group}='{medians.index[0]}' "
        f"({medians.iloc[0]:,.2f}) and lowest for '{medians.index[-1]}' "
        f"({medians.iloc[-1]:,.2f}).",
        [numeric, group],
    )


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None,
            max_points: int = 5000) -> ChartResult | None:
    """Relationship between two numeric columns, with a fitted trend line."""
    cols = [x, y] + ([color] if color and color in df.columns else [])
    sub = df[cols].dropna()
    if len(sub) < 10:
        return None

    plotted = sub.sample(max_points, random_state=42) if len(sub) > max_points else sub
    fig = px.scatter(plotted, x=x, y=y,
                     color=color if color and color in plotted.columns else None,
                     color_discrete_sequence=PALETTE, opacity=0.6,
                     trendline="ols", trendline_color_override="#0F1B2D")
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))

    r = float(sub[x].corr(sub[y]))

    # A column with no variation gives an undefined correlation. `build_chart`
    # validates dtype but not variance, so NaN reached the chart's data, the
    # description read "Pearson r = nan", and that dict is handed to the report
    # agent's number check -- where `int(nan)` raised and failed the report
    # stage. Saying the correlation is undefined, and why, is both true and
    # more useful than a coefficient the data cannot support.
    if not math.isfinite(r):
        return _result(
            f"scatter_{_slug(x)}_{_slug(y)}", "scatter", f"{y} versus {x}",
            _style(fig, f"{y} versus {x}", x, y),
            {"pearson": None, "n": int(len(sub)),
             "plotted_points": int(len(plotted)),
             "sampled": bool(len(sub) > max_points)},
            f"{x} and {y} cannot be correlated: one of them takes a single "
            f"value throughout, so there is no variation to relate "
            f"(n = {len(sub):,}).",
            [x, y],
        )

    strength = ("strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4
                else "weak" if abs(r) >= 0.2 else "negligible")

    return _result(
        f"scatter_{_slug(x)}_{_slug(y)}", "scatter", f"{y} versus {x}",
        _style(fig, f"{y} versus {x}", x, y),
        {"pearson": round(r, 4), "n": int(len(sub)),
         "plotted_points": int(len(plotted)),
         "sampled": bool(len(sub) > max_points)},
        f"{x} and {y} show a {strength} {'positive' if r > 0 else 'negative'} "
        f"linear relationship (Pearson r = {r:.3f}, n = {len(sub):,}).",
        [x, y],
    )


def correlation_heatmap(df: pd.DataFrame, columns: list[str] | None = None,
                        max_columns: int = 12) -> ChartResult | None:
    """Correlation matrix across numeric columns.

    Uses a diverging scale centred on zero so that positive and negative
    relationships are distinguishable at a glance rather than by reading values.
    """
    numeric = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric = [c for c in numeric if df[c].nunique(dropna=True) > 1][:max_columns]
    if len(numeric) < 3:
        return None

    corr = df[numeric].corr().round(3)
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
                    color_continuous_scale=["#C2491D", "#ffffff", "#2346A0"])
    fig.update_layout(coloraxis_colorbar=dict(title="r", thickness=14))

    # Report the strongest off-diagonal pair, which is what a reader looks for.
    m = corr.where(~np.eye(len(corr), dtype=bool)).abs()
    a = m.stack().idxmax()
    strongest = corr.loc[a[0], a[1]]

    return _result(
        "corr_heatmap", "heatmap", "Correlation matrix",
        _style(fig, "Correlation matrix"),
        {"matrix": {i: {j: float(corr.loc[i, j]) for j in corr.columns} for i in corr.index},
         "strongest_pair": [a[0], a[1]], "strongest_value": float(strongest)},
        f"Across {len(numeric)} numeric columns the strongest correlation is between "
        f"{a[0]} and {a[1]} (r = {strongest:.3f}).",
        numeric,
    )


def target_balance_chart(df: pd.DataFrame, target: str) -> ChartResult | None:
    """Class distribution of the target."""
    counts = df[target].value_counts()
    if not 2 <= len(counts) <= 20:
        return None

    total = int(counts.sum())
    pct = (counts / total * 100).round(1)
    fig = px.bar(x=[str(i) for i in counts.index], y=counts.values,
                 color_discrete_sequence=[PALETTE[0]])
    fig.update_traces(
        text=[f"{v:,}<br>{p:.1f}%" for v, p in zip(counts.values, pct.values)],
        textposition="outside",
    )

    ratio = float(counts.iloc[0] / counts.iloc[-1])
    return _result(
        f"balance_{_slug(target)}", "target_balance", f"Class balance of {target}",
        _style(fig, f"Class balance of {target}", target, "Count"),
        {"counts": {str(k): int(v) for k, v in counts.items()},
         "percentages": {str(k): float(v) for k, v in pct.items()},
         "imbalance_ratio": round(ratio, 2)},
        f"{target} has {len(counts)} classes with an imbalance ratio of {ratio:.2f}:1 "
        f"('{counts.index[0]}' is most common at {pct.iloc[0]:.1f}%).",
        [target],
    )


# --------------------------------------------------------------------------- #
# Dispatch and export
# --------------------------------------------------------------------------- #

BUILDERS = {
    "histogram": histogram,
    "bar": bar_counts,
    "grouped_rate": grouped_rate,
    "box": box_by_group,
    "scatter": scatter,
    "heatmap": correlation_heatmap,
    "target_balance": target_balance_chart,
}

# Required positional arguments for each chart kind, used to validate a
# requested specification before attempting to build it.
SIGNATURES = {
    "histogram": ("column",),
    "bar": ("column",),
    "grouped_rate": ("feature", "target"),
    "box": ("numeric", "group"),
    "scatter": ("x", "y"),
    "heatmap": (),
    "target_balance": ("target",),
}


def build_chart(df: pd.DataFrame, spec: dict[str, Any]) -> tuple[ChartResult | None, str | None]:
    """Build one chart from a specification.

    Returns `(chart, None)` on success or `(None, reason)` when the request
    cannot be honoured. Specifications come from a language model, so a bad
    request must produce a recorded reason rather than an exception.
    """
    kind = spec.get("kind")
    if kind not in BUILDERS:
        return None, f"Unknown chart kind '{kind}'."

    kwargs = {k: v for k, v in spec.items() if k not in ("kind", "reason", "title")}

    missing = [p for p in SIGNATURES[kind] if not kwargs.get(p)]
    if missing:
        return None, f"Chart '{kind}' is missing required argument(s): {', '.join(missing)}."

    for param in SIGNATURES[kind]:
        col = kwargs.get(param)
        if isinstance(col, str) and col not in df.columns:
            return None, f"Chart '{kind}' references column '{col}', which is not in the dataset."

    # Charts that describe a distribution or a relationship need numeric input;
    # requesting one on a text column is a specification error, not a failure.
    numeric_params = {"histogram": ("column",), "box": ("numeric",), "scatter": ("x", "y")}
    for param in numeric_params.get(kind, ()):
        col = kwargs.get(param)
        if col and not pd.api.types.is_numeric_dtype(df[col]):
            return None, f"Chart '{kind}' requires a numeric column but '{col}' is not numeric."

    try:
        chart = BUILDERS[kind](df, **kwargs)
    except TypeError as exc:
        return None, f"Invalid arguments for chart '{kind}': {exc}"
    except Exception as exc:
        return None, f"Chart '{kind}' failed: {type(exc).__name__}: {exc}"

    if chart is None:
        return None, f"Chart '{kind}' could not be produced: insufficient or unsuitable data."
    return chart, None


def export_chart(chart: ChartResult, directory: Path | None = None,
                 width: int = 900, height: int = 500) -> str | None:
    """Write a chart to PNG for the report. Returns the path, or None if export fails.

    Static export needs the kaleido engine, which may be absent. That is a
    degraded report rather than a failed run, so the error is swallowed and
    reported through the return value.
    """
    directory = directory or (CONFIG.REPORTS_DIR if hasattr(CONFIG, "REPORTS_DIR") else None)
    if directory is None:
        from config import REPORTS_DIR
        directory = REPORTS_DIR / "charts"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{chart['id']}.png"
    try:
        chart["figure"].write_image(str(path), width=width, height=height, scale=2)
    except Exception as exc:
        # Almost always a missing kaleido install. Recorded on the chart so the
        # report can say the image is unavailable instead of referencing a file
        # that was never written.
        chart["export_error"] = f"{type(exc).__name__}: {exc}"
        return None
    chart["path"] = str(path)
    return str(path)
