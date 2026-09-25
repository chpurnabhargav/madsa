"""Multivariate and unsupervised analysis.

The rest of the framework analyses columns one or two at a time. This module
looks at the feature space as a whole, which is where several standard data
science findings live: redundancy between predictors, natural groupings in the
data, and how much of the variation a handful of components accounts for.

It matters for a dataset with many columns, where pairwise statistics become
unreadable long before they become uninformative, and for an unsupervised dataset
where there is no target to organise the analysis around.

Techniques used here:
  scikit-learn  PCA, KMeans, silhouette analysis, mutual information
  statsmodels   variance inflation factors
  scipy         hierarchical linkage for correlation clustering
"""
from __future__ import annotations

from typing import Any

# The joblib core-count workaround lives in `config`, which every module in the
# project imports before anything touches scikit-learn. A second copy here set
# the value to the logical core count -- precisely the value `config` documents
# as *not* working, since joblib only skips its probe when the value is strictly
# lower. Whichever module reached `setdefault` first then decided whether the
# spurious traceback appeared, which made the fix depend on import order.
from config import CONFIG  # noqa: F401  (imported for its side effect)

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 30
MAX_FEATURES = 50


def _numeric_matrix(df: pd.DataFrame, exclude: tuple[str, ...] = ()
                    ) -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    """Standardised numeric matrix with missing values imputed.

    Standardisation is applied because every technique here is distance- or
    variance-based: without it a column measured in thousands would dominate a
    column measured in units regardless of which carries more information.
    """
    columns = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ][:MAX_FEATURES]

    if len(columns) < 2 or len(df) < MIN_ROWS:
        return None, None

    values = df[columns].to_numpy(dtype=float)
    values = SimpleImputer(strategy="median").fit_transform(values)
    if values.shape[1] < 2:
        return None, None
    return StandardScaler().fit_transform(values), columns


def principal_components(df: pd.DataFrame, exclude: tuple[str, ...] = (),
                         variance_target: float = 0.90) -> dict[str, Any] | None:
    """Principal component analysis of the numeric feature space.

    Reports how many components are needed to retain most of the variance, which
    is the direct measure of redundancy among the features, and the loadings of
    the leading components, which say what each one represents.
    """
    matrix, columns = _numeric_matrix(df, exclude)
    if matrix is None:
        return None

    n_components = min(len(columns), matrix.shape[0], 10)
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(matrix)

    ratios = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratios)

    # Only `n_components` components were computed, so the target may not be
    # reached within them. `searchsorted` returns `len(cumulative)` in that
    # case, and adding one produced a component count one higher than anything
    # the analysis established -- 11 components reported from 10 computed, with
    # a redundancy figure derived from the invented number. This framework
    # spends twelve verification mechanisms refusing to state what it has not
    # measured; the same standard applies to its own tools. When the target is
    # not reached, that is what gets reported.
    reached = bool(cumulative[-1] >= variance_target)
    needed = int(np.searchsorted(cumulative, variance_target) + 1) if reached else None

    components = []
    for i in range(min(3, n_components)):
        loadings = sorted(
            ({"feature": c, "loading": round(float(v), 4)}
             for c, v in zip(columns, pca.components_[i])),
            key=lambda d: -abs(d["loading"]),
        )
        components.append({
            "component": i + 1,
            "explained_variance": round(float(ratios[i]), 4),
            "top_loadings": loadings[:5],
        })

    return {
        "n_features": len(columns),
        "features": columns,
        "explained_variance_ratio": [round(float(r), 4) for r in ratios],
        "cumulative_variance": [round(float(c), 4) for c in cumulative],
        "components_for_target": needed,
        "variance_target": variance_target,
        "components_computed": int(n_components),
        # Stated only when the target was actually reached. Explaining why a
        # figure is absent is more useful than a figure that is wrong, and it
        # is itself a finding: variance that does not concentrate within the
        # leading components means the features are largely independent.
        "target_not_reached": None if reached else (
            f"The leading {n_components} components account for "
            f"{cumulative[-1]:.1%} of the variance, short of the "
            f"{variance_target:.0%} target, so the number of components needed "
            f"was not established. The variance is spread broadly, which means "
            f"these features carry largely independent information."),
        # A feature space whose variance concentrates in far fewer components
        # than it has columns is carrying substantial redundancy. Undefined when
        # the component count it derives from was never established.
        "redundancy": round(1 - needed / len(columns), 4) if needed else None,
        "leading_components": components,
    }


def cluster_structure(df: pd.DataFrame, exclude: tuple[str, ...] = (),
                      max_k: int = 6) -> dict[str, Any] | None:
    """Search for natural groupings using k-means across a range of k.

    Silhouette score selects k rather than inertia, because inertia falls
    monotonically with k and so cannot indicate a best value. A silhouette below
    roughly 0.25 means the groupings are weak enough that reporting them as
    segments would overstate what is there.
    """
    matrix, columns = _numeric_matrix(df, exclude)
    if matrix is None or len(matrix) < 50:
        return None

    # Silhouette on a very large sample is slow and no more informative.
    sample = matrix
    if len(matrix) > 5000:
        rng = np.random.default_rng(42)
        sample = matrix[rng.choice(len(matrix), 5000, replace=False)]

    scores = []
    for k in range(2, min(max_k, len(sample) // 10) + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(sample)
        if len(set(labels)) < 2:
            continue
        scores.append({
            "k": k,
            "silhouette": round(float(silhouette_score(sample, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(sample, labels)), 1),
            "inertia": round(float(model.inertia_), 2),
        })

    if not scores:
        return None

    best = max(scores, key=lambda s: s["silhouette"])
    model = KMeans(n_clusters=best["k"], random_state=42, n_init=10)
    labels = model.fit_predict(matrix)

    # Cluster centres are reported in standardised units, so a value of +1.2
    # reads directly as "1.2 standard deviations above the mean".
    centres = []
    for i, centre in enumerate(model.cluster_centers_):
        distinguishing = sorted(
            ({"feature": c, "z_score": round(float(v), 3)}
             for c, v in zip(columns, centre)),
            key=lambda d: -abs(d["z_score"]),
        )
        centres.append({
            "cluster": i,
            "size": int((labels == i).sum()),
            "share": round(float((labels == i).mean()), 4),
            "distinguishing_features": distinguishing[:4],
        })

    return {
        "features": columns,
        "scores_by_k": scores,
        "best_k": best["k"],
        "silhouette": best["silhouette"],
        # Silhouette guide: below 0.25 the structure is weak, 0.25-0.5 moderate,
        # above 0.5 strong and worth reporting as genuine segments.
        "structure_strength": ("strong" if best["silhouette"] >= 0.5
                               else "moderate" if best["silhouette"] >= 0.25
                               else "weak"),
        "meaningful": bool(best["silhouette"] >= 0.25),
        "clusters": sorted(centres, key=lambda c: -c["size"]),
        "labels": labels.tolist() if len(labels) <= 10000 else None,
    }


def multicollinearity(df: pd.DataFrame, exclude: tuple[str, ...] = ()
                      ) -> dict[str, Any] | None:
    """Variance inflation factors for the numeric features.

    VIF measures how much a coefficient's variance is inflated by correlation
    with the other predictors. It catches redundancy that pairwise correlation
    misses, because a feature can be predictable from a combination of others
    while correlating strongly with none of them individually. Above 10 is the
    conventional threshold for a coefficient that cannot be interpreted.
    """
    matrix, columns = _numeric_matrix(df, exclude)
    if matrix is None or len(columns) < 3:
        return None

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return None

    factors = []
    unevaluated = []
    for i, column in enumerate(columns):
        try:
            vif = float(variance_inflation_factor(matrix, i))
        except Exception as exc:
            # A perfectly collinear column makes the regression singular. The
            # column is named rather than dropped, because silently omitting it
            # would read as "this column has no multicollinearity problem" --
            # the opposite of what a singular fit indicates.
            unevaluated.append({"column": column,
                                "reason": f"{type(exc).__name__}"})
            continue
        if not np.isfinite(vif):
            vif = float("inf")
        factors.append({
            "feature": column,
            "vif": round(vif, 3) if np.isfinite(vif) else None,
            "severity": ("severe" if vif > 10 else "moderate" if vif > 5 else "acceptable"),
        })

    # Worst-first, which needs care with two special values. `-(vif or inf)`
    # got both wrong: a genuinely computed VIF of 0.0 is falsy, so it sorted as
    # though infinite and led the list, while a column whose VIF really was
    # infinite is stored as None and sorted last -- truncated out of the top
    # slice passed to the agent, so the most severely collinear column was the
    # one the interpretation never saw.
    #
    # None means infinite here, by the assignment a few lines above, so it
    # sorts first; everything else sorts by magnitude.
    factors.sort(key=lambda f: (0 if f["vif"] is None else 1,
                                -(f["vif"] if f["vif"] is not None else 0.0)))
    severe = [f["feature"] for f in factors if f["severity"] == "severe"]

    return {
        "factors": factors,
        "unevaluated_columns": unevaluated,
        "severe_multicollinearity": severe,
        "n_severe": len(severe),
        "interpretation": (
            f"{len(severe)} feature(s) are strongly predictable from the others; "
            "linear-model coefficients on them cannot be interpreted individually."
            if severe else
            "No severe multicollinearity; coefficients are individually interpretable."
        ),
    }


def correlation_clusters(df: pd.DataFrame, exclude: tuple[str, ...] = (),
                         threshold: float = 0.7) -> dict[str, Any] | None:
    """Group features by correlation using hierarchical clustering.

    On a wide dataset a correlation matrix is too large to read. Grouping
    features that move together turns it into a short list of redundant sets, one
    representative of which is usually enough for modelling.
    """
    columns = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        and df[c].nunique(dropna=True) > 1
    ][:MAX_FEATURES]
    if len(columns) < 3 or len(df) < MIN_ROWS:
        return None

    corr = df[columns].corr().fillna(0).to_numpy()
    distance = 1 - np.abs(corr)
    np.fill_diagonal(distance, 0)
    distance = (distance + distance.T) / 2  # enforce symmetry for squareform

    try:
        linkage = hierarchy.linkage(squareform(distance, checks=False), method="average")
        labels = hierarchy.fcluster(linkage, 1 - threshold, criterion="distance")
    except Exception:
        return None

    groups: dict[int, list[str]] = {}
    for column, label in zip(columns, labels):
        groups.setdefault(int(label), []).append(column)

    redundant = [sorted(g) for g in groups.values() if len(g) > 1]
    return {
        "threshold": threshold,
        "n_groups": len(groups),
        "redundant_groups": redundant,
        "n_redundant_features": sum(len(g) - 1 for g in redundant),
        "representative_features": sorted(g[0] for g in groups.values()),
    }


def feature_relevance(df: pd.DataFrame, target: str, task: str,
                      exclude: tuple[str, ...] = ()) -> dict[str, Any] | None:
    """Mutual information between each feature and the target.

    Mutual information captures any dependence, including non-linear and
    non-monotonic relationships that correlation reports as absent. It is
    therefore a useful cross-check: a feature ranked high here but low by
    correlation is related to the target in a way a linear model cannot use.
    """
    if target not in df.columns or len(df) < MIN_ROWS:
        return None

    matrix, columns = _numeric_matrix(df, exclude=(target,) + exclude)
    if matrix is None:
        return None

    y = df[target]
    mask = y.notna().to_numpy()
    if mask.sum() < MIN_ROWS:
        return None
    matrix, y = matrix[mask], y[mask]

    try:
        if task == "classification":
            scores = mutual_info_classif(matrix, y.astype(str), random_state=42)
        else:
            scores = mutual_info_regression(matrix, y.astype(float), random_state=42)
    except Exception:
        return None

    ranked = sorted(
        ({"feature": c, "mutual_information": round(float(s), 5)}
         for c, s in zip(columns, scores)),
        key=lambda d: -d["mutual_information"],
    )
    # Mutual information is non-negative and near zero under independence, so a
    # small positive value is not evidence of a relationship.
    informative = [r["feature"] for r in ranked if r["mutual_information"] > 0.01]

    return {
        "target": target,
        "task": task,
        "ranked": ranked,
        "informative_features": informative,
        "uninformative_features": [
            r["feature"] for r in ranked if r["mutual_information"] <= 0.01],
    }


def run_multivariate(df: pd.DataFrame, target: str | None = None,
                     task: str | None = None) -> dict[str, Any]:
    """Run the multivariate analyses that apply to this dataset.

    Each analysis can decline for a legitimate reason -- too few numeric
    columns, too few rows -- and each can also fail outright on degenerate data.
    Both cases previously produced the same result: the key was simply absent,
    and a reader could not tell whether an analysis found nothing or never ran.

    For a framework whose contribution is refusing to overstate, a silently
    missing analysis is the wrong kind of silence. Every analysis that produced
    no result is therefore listed in `unavailable`, so absence is visible rather
    than merely quiet.
    """
    exclude = (target,) if target else ()

    results: dict[str, Any] = {
        "pca": principal_components(df, exclude),
        "clustering": cluster_structure(df, exclude),
        "multicollinearity": multicollinearity(df, exclude),
        "correlation_groups": correlation_clusters(df, exclude),
    }
    if target and task in ("classification", "regression"):
        results["feature_relevance"] = feature_relevance(df, target, task)

    results["unavailable"] = [
        name for name, value in results.items()
        if name != "unavailable" and not value
    ]
    results["applicable"] = any(
        v for k, v in results.items()
        if k not in ("applicable", "unavailable")
    )
    return results
