"""Distributed execution backend: the same analysis, on data that exceeds memory.

Every other tool in this package works on a pandas DataFrame held entirely in
RAM. That is the right choice for most datasets, but it puts a hard ceiling on
the framework: an analysis that cannot load its data cannot run at all.

This module removes that ceiling by expressing the same computations as Spark
DataFrame operations, which execute across partitions and spill to disk instead
of failing. It deliberately returns the *identical* result contract as
``tools.profiling`` and ``tools.eda``, so no agent knows or cares which engine
produced its input. Swapping the execution engine is therefore invisible above
this line, which is what makes the choice safe to make automatically.

The engine is not always the faster one. Spark pays JVM startup, serialisation
and shuffle costs that pandas does not, so on a small dataset it is slower by a
wide margin. ``choose_engine`` encodes that tradeoff rather than defaulting to
the more impressive-sounding option.
"""
from __future__ import annotations

import os
import shutil

import numpy as np
from typing import Any

from config import CONFIG

# Spark is an optional dependency. The framework must degrade to pandas rather
# than fail when it or a JVM is absent, exactly as the agents degrade to computed
# results when a model is unreachable.
try:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    _IMPORTED = True
except ImportError:  # pragma: no cover - exercised only where pyspark is absent
    SparkDataFrame = Any  # type: ignore[assignment,misc]
    SparkSession = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _IMPORTED = False

_SESSION = None


def spark_available() -> tuple[bool, str]:
    """Whether a distributed run is possible here, and why not when it is not.

    Both conditions are checked because they fail independently: pyspark is a
    Python package, but it launches a JVM, and a machine can easily have one
    without the other.
    """
    if not _IMPORTED:
        return False, "pyspark is not installed"
    if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        return False, "no Java runtime found on PATH"
    return True, "available"


def get_session() -> "SparkSession":
    """Return the shared SparkSession, starting it on first use.

    A session is expensive to create and holds a JVM, so one is reused for the
    whole run. Settings are tuned for a single machine: the shuffle partition
    default of 200 is sized for a cluster and produces mostly-empty tasks here,
    which costs more in scheduling than it saves in parallelism.
    """
    global _SESSION
    if _SESSION is not None:
        return _SESSION

    ok, reason = spark_available()
    if not ok:
        raise RuntimeError(f"Spark is unavailable: {reason}")

    _SESSION = (
        SparkSession.builder.appName("MADSA")
        .master(CONFIG.spark_master)
        .config("spark.sql.shuffle.partitions", str(CONFIG.spark_shuffle_partitions))
        .config("spark.ui.enabled", "false")
        # The console progress bar redraws itself into every log and terminal
        # the run writes to; the trace already records what each stage did.
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", CONFIG.spark_driver_memory)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    # Spark logs its own progress at INFO, which would bury the framework's
    # trace output under scheduler chatter.
    _SESSION.sparkContext.setLogLevel("ERROR")
    return _SESSION


def stop_session() -> None:
    """Shut the JVM down. Called at the end of a run so the process can exit."""
    global _SESSION
    if _SESSION is not None:
        _SESSION.stop()
        _SESSION = None


#: File types the distributed reader understands, and how to read each. Spark
#: reads CSV and Parquet natively. It cannot read Excel at all, and JSON is
#: ambiguous between the line-delimited form Spark expects and the array form
#: pandas writes, so both stay in memory rather than being misparsed.
SPARK_FORMATS = {".csv": "csv", ".txt": "csv", ".parquet": "parquet"}


def choose_engine(path: str | None = None, n_rows: int | None = None) -> dict[str, Any]:
    """Decide which execution engine this dataset warrants, and record why.

    The decision is part of the framework's adaptive behaviour and is reported
    alongside the workflow plan, so a reader can see that a distributed run was
    chosen on evidence rather than by default. File size is preferred over row
    count as the trigger because it is knowable before anything is loaded, which
    is precisely the situation where the choice matters.

    Size alone used to decide, whatever the file was. `load` then parsed every
    file as CSV, so a 60 MB Excel or JSON file crossed the threshold and was
    read by Spark as comma-separated text -- garbage columns or a failed read,
    for a file the in-memory loader handles correctly. The format is now part
    of the decision.
    """
    ok, reason = spark_available()
    size_mb = None
    if path and os.path.exists(path):
        size_mb = round(os.path.getsize(path) / 1024**2, 2)
    suffix = os.path.splitext(path)[1].lower() if path else ""

    exceeds = (
        (size_mb is not None and size_mb >= CONFIG.spark_size_threshold_mb)
        or (n_rows is not None and n_rows >= CONFIG.spark_row_threshold)
    )

    if not CONFIG.spark_enabled:
        return {"engine": "pandas", "size_mb": size_mb,
                "reason": "Distributed execution is disabled in the configuration."}
    if not exceeds:
        return {"engine": "pandas", "size_mb": size_mb,
                "reason": (
                    f"Dataset is below the distributed-execution threshold "
                    f"({CONFIG.spark_size_threshold_mb} MB / "
                    f"{CONFIG.spark_row_threshold:,} rows); in-memory execution is "
                    f"faster at this scale.")}
    if path and suffix not in SPARK_FORMATS:
        return {"engine": "pandas", "size_mb": size_mb,
                "reason": (
                    f"Dataset exceeds the distributed-execution threshold, but "
                    f"the distributed reader handles CSV and Parquet only; this "
                    f"'{suffix or 'unknown'}' file is read in memory instead.")}
    if not ok:
        return {"engine": "pandas", "size_mb": size_mb,
                "reason": (
                    f"Dataset is large enough for distributed execution but Spark "
                    f"is unavailable ({reason}); falling back to in-memory execution.")}
    return {"engine": "spark", "size_mb": size_mb,
            "reason": (
                f"Dataset exceeds the distributed-execution threshold "
                f"({size_mb} MB); running on Spark so the analysis is not bounded "
                f"by available memory.")}


def load(path: str) -> "SparkDataFrame":
    """Read a CSV or Parquet file into a cached Spark DataFrame.

    Cached because the frame is read by many separate actions -- the row count
    alone is taken by the loader, the orchestrator, the sampler and the
    profiler -- and an uncached DataFrame re-reads and re-parses the source
    file for every one of them. With `inferSchema` a CSV is parsed twice more
    on top of that. `cache()` spills to disk rather than failing when the data
    exceeds memory, so it keeps the property this engine exists for.
    """
    fmt = SPARK_FORMATS.get(os.path.splitext(path)[1].lower())
    if fmt is None:
        raise ValueError(
            f"The distributed reader handles CSV and Parquet, not '{path}'.")
    reader = get_session().read
    if fmt == "parquet":
        frame = reader.parquet(path)
    else:
        frame = (reader.option("header", "true")
                 .option("inferSchema", "true")
                 .option("nullValue", "")
                 .csv(path))
    return frame.cache()


def _is_numeric(dtype: str) -> bool:
    return dtype.startswith(("int", "bigint", "smallint", "tinyint",
                             "double", "float", "decimal", "long"))


def _semantic_type(dtype: str, n_unique: int, n_non_null: int, n_rows: int) -> str:
    """The same classification rule as the pandas profiler, on Spark types.

    Kept deliberately parallel rather than shared, because the inputs differ:
    here the counts arrive already aggregated from the cluster instead of being
    computed from a Series.
    """
    if n_non_null == 0:
        return "empty"
    if dtype == "boolean":
        return "boolean"
    if dtype.startswith(("timestamp", "date")):
        return "datetime"
    if _is_numeric(dtype):
        if n_unique == n_non_null and dtype.startswith(("int", "bigint", "long")):
            return "identifier"
        if n_unique <= CONFIG.max_classification_classes:
            return "categorical_numeric"
        return "numeric"
    if n_unique == n_non_null and n_non_null > 0.9 * n_rows:
        return "identifier"
    if n_unique <= CONFIG.max_classification_classes:
        return "categorical"
    return "text"


def profile_dataset(sdf: "SparkDataFrame") -> dict[str, Any]:
    """A structural profile identical in shape to the pandas profiler's.

    Every per-column count is requested in a single ``select``. Spark evaluates
    that as one pass over the data, where a loop of per-column aggregations
    would launch one job per column and re-read the input each time. On a
    fifteen-column file that is the difference between one scan and fifteen.
    """
    n_rows = sdf.count()
    n_cols = len(sdf.columns)
    dtypes = dict(sdf.dtypes)

    aggs = []
    for name in sdf.columns:
        c = F.col(f"`{name}`")
        aggs.append(F.count(c).alias(f"{name}__nonnull"))
        aggs.append(F.countDistinct(c).alias(f"{name}__distinct"))
    counts = sdf.select(*aggs).collect()[0].asDict()

    # Numeric summaries, also in one pass. Quantiles are computed separately
    # because approxQuantile is a different algorithm (Greenwald-Khanna) and
    # cannot be folded into a column aggregation.
    numeric_cols = [n for n in sdf.columns if _is_numeric(dtypes[n])]
    stats_row: dict[str, Any] = {}
    if numeric_cols:
        num_aggs = []
        for name in numeric_cols:
            c = F.col(f"`{name}`")
            num_aggs += [
                F.mean(c).alias(f"{name}__mean"),
                F.stddev(c).alias(f"{name}__std"),
                F.min(c).alias(f"{name}__min"),
                F.max(c).alias(f"{name}__max"),
                F.skewness(c).alias(f"{name}__skew"),
                F.kurtosis(c).alias(f"{name}__kurt"),
                F.sum(F.when(c == 0, 1).otherwise(0)).alias(f"{name}__zeros"),
                F.sum(F.when(c < 0, 1).otherwise(0)).alias(f"{name}__negs"),
            ]
        stats_row = sdf.select(*num_aggs).collect()[0].asDict()

    quantiles = {}
    if numeric_cols:
        # relativeError trades exactness for a single pass; 0.001 keeps the
        # quartiles well inside the precision the report ever prints.
        q = sdf.approxQuantile([f"`{n}`" for n in numeric_cols], [0.25, 0.5, 0.75], 0.001)
        quantiles = dict(zip(numeric_cols, q))

    columns = []
    for name in sdf.columns:
        dtype = dtypes[name]
        n_non_null = int(counts[f"{name}__nonnull"])
        n_unique = int(counts[f"{name}__distinct"])
        n_missing = n_rows - n_non_null
        sem = _semantic_type(dtype, n_unique, n_non_null, n_rows)

        info: dict[str, Any] = {
            "name": str(name),
            "dtype": dtype,
            "semantic_type": sem,
            "missing": n_missing,
            "missing_fraction": round(n_missing / n_rows, 4) if n_rows else 0.0,
            "unique": n_unique,
            "unique_fraction": round(n_unique / n_rows, 4) if n_rows else 0.0,
        }

        if sem in ("numeric", "categorical_numeric") and name in quantiles:
            qs = quantiles[name]
            info["stats"] = {
                "mean": float(stats_row[f"{name}__mean"] or 0.0),
                "std": float(stats_row[f"{name}__std"] or 0.0),
                "min": float(stats_row[f"{name}__min"] or 0.0),
                "q1": float(qs[0]) if len(qs) > 0 else 0.0,
                "median": float(qs[1]) if len(qs) > 1 else 0.0,
                "q3": float(qs[2]) if len(qs) > 2 else 0.0,
                "max": float(stats_row[f"{name}__max"] or 0.0),
                "skewness": float(stats_row[f"{name}__skew"] or 0.0),
                "kurtosis": float(stats_row[f"{name}__kurt"] or 0.0),
                "zeros": int(stats_row[f"{name}__zeros"] or 0),
                "negatives": int(stats_row[f"{name}__negs"] or 0),
            }
        elif sem in ("categorical", "boolean", "text"):
            info["stats"] = _top_values(sdf, name, n_non_null)
        columns.append(info)

    n_duplicates = n_rows - sdf.distinct().count()
    total_missing = sum(c["missing"] for c in columns)

    type_counts: dict[str, int] = {}
    for c in columns:
        type_counts[c["semantic_type"]] = type_counts.get(c["semantic_type"], 0) + 1

    return {
        "shape": {"rows": n_rows, "columns": n_cols},
        "memory_mb": None,  # meaningless for a distributed frame; never loaded whole
        "sampled": False,   # every statistic above is computed on the full data
        "engine": "spark",
        "duplicate_rows": n_duplicates,
        "duplicate_fraction": round(n_duplicates / n_rows, 4) if n_rows else 0.0,
        "total_missing_cells": total_missing,
        "missing_fraction_overall": round(total_missing / (n_rows * n_cols), 4)
        if n_rows and n_cols else 0.0,
        "semantic_type_counts": type_counts,
        "columns": columns,
    }


def _top_values(sdf: "SparkDataFrame", name: str, n_non_null: int,
                top_n: int = 5) -> dict[str, Any]:
    """Most frequent values for a categorical column, computed by aggregation."""
    if n_non_null == 0:
        return {}
    counts = (
        sdf.select(F.col(f"`{name}`").alias("v")).where(F.col("v").isNotNull())
        .groupBy("v").count()
    )
    rows = counts.orderBy(F.desc("count")).limit(top_n).collect()
    if not rows:
        return {}
    smallest = counts.orderBy(F.asc("count")).limit(1).collect()[0]["count"]
    return {
        "top_values": [
            {"value": str(r["v"]), "count": int(r["count"]),
             "fraction": round(int(r["count"]) / n_non_null, 4)}
            for r in rows
        ],
        "imbalance_ratio": round(float(rows[0]["count"]) / smallest, 3)
        if smallest else 1.0,
    }


def correlations(sdf: "SparkDataFrame", numeric_cols: list[str],
                 min_abs: float = 0.3,
                 exclude: tuple[str, ...] = ()) -> dict[str, Any]:
    """Pairwise correlations across the cluster, in the pandas tool's contract.

    Returns exactly what ``tools.eda.correlations`` returns, key for key, so the
    EDA Agent's interpretation and the verification that follows it cannot tell
    which engine ran. An earlier version of this function returned a shape of
    its own, which quietly broke that guarantee: the agent would have received
    keys it did not understand, and the identical-contract claim the whole
    design rests on would have been false in the one place it was most load
    bearing.

    Both coefficients come from Spark ML's correlation matrices rather than a
    pair-by-pair loop. One pass computes every pair at once, where looping would
    launch a job per pair -- on thirty columns that is 435 scans of the data
    against two. Spearman is obtained from the same API, which rank-transforms
    the columns internally before correlating them.

    The p-value the pandas tool reports is not available here: Spark computes
    the coefficient without the significance test around it. It is returned as
    None rather than approximated, because a fabricated p-value is worse than an
    absent one, and effect size is what the framework ranks on in any case.
    """
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.stat import Correlation

    cols = [c for c in numeric_cols if c in sdf.columns and c not in exclude][:30]
    if len(cols) < 2:
        return {"pairs": [], "n_numeric_columns": len(cols),
                "note": "At least two numeric columns are needed for correlation."}

    # Rows with a null in any selected column are dropped once, so every pair is
    # computed on the same rows. The pandas tool drops per pair, which can score
    # different pairs on different subsets; doing it once here is the stricter
    # choice and is stated in the returned `n`.
    complete = sdf.select(*[F.col(f"`{c}`").cast("double").alias(c) for c in cols]).dropna()
    n_complete = complete.count()
    if n_complete < 10:
        return {"pairs": [], "n_numeric_columns": len(cols),
                "note": "Too few complete rows for correlation."}

    assembled = VectorAssembler(inputCols=cols, outputCol="__features").transform(complete)
    pearson = Correlation.corr(assembled, "__features", "pearson").head()[0].toArray()
    spearman = Correlation.corr(assembled, "__features", "spearman").head()[0].toArray()

    pairs = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            b = cols[j]
            pear, spear = float(pearson[i][j]), float(spearman[i][j])
            if np.isnan(pear) or np.isnan(spear):
                continue
            if abs(pear) < min_abs and abs(spear) < min_abs:
                continue
            pairs.append({
                "column_a": a,
                "column_b": b,
                "pearson": round(pear, 4),
                "spearman": round(spear, 4),
                "p_value": None,
                "n": int(n_complete),
                "strength": _strength(pear),
                "direction": "positive" if pear > 0 else "negative",
                "non_linear": bool(abs(spear) - abs(pear) > 0.15),
            })

    pairs.sort(key=lambda p: -abs(p["pearson"]))
    return {"pairs": pairs, "n_numeric_columns": len(cols),
            "threshold": min_abs, "engine": "spark"}


#: The bands the pandas tool applies to a Pearson correlation. Named here
#: because both engines must use the same ones and a copied constant is how
#: they stopped doing so.
CORRELATION_BANDS = (0.3, 0.5, 0.7)


def _strength(value: float) -> str:
    """The pandas tool's own banding function, not a reimplementation of it.

    This previously reimplemented the bands and got them wrong: it returned
    weak / moderate / strong / very strong where pandas returns negligible /
    weak / moderate / strong on the same thresholds, so every correlation on a
    distributed run was described one level stronger than the identical
    correlation on a single machine -- r=0.32 "moderate" against "weak",
    r=0.55 "strong" against "moderate". The numbers agreed to six decimal
    places, which is what had been checked; the words did not, which is what
    the report actually says to a reader.

    The engines must produce the same result, so there is one implementation
    and both call it.
    """
    from tools.eda import _strength as band

    return band(value, CORRELATION_BANDS)


def to_pandas_sample(sdf: "SparkDataFrame", n: int) -> Any:
    """Draw a bounded sample into memory for tools that cannot be distributed.

    Charting and scikit-learn modelling need a local frame. Sampling is done in
    Spark and only the sample crosses back, so the driver never holds the full
    dataset. The fraction is over-requested slightly because Spark's sampling is
    probabilistic and returns approximately, not exactly, the share asked for.
    """
    total = sdf.count()
    if total <= n:
        return sdf.toPandas()
    fraction = min(1.0, (n / total) * 1.1)
    return sdf.sample(withReplacement=False, fraction=fraction, seed=42).limit(n).toPandas()


# --------------------------------------------------------------------------- #
# Distributed validation of the selected model
# --------------------------------------------------------------------------- #

#: scikit-learn families that are boosted ensembles; MLlib's counterpart is GBT.
_BOOSTED = ("Gradient Boosting", "Hist Gradient Boosting", "XGBoost", "LightGBM")
_LINEAR = ("Logistic Regression", "Linear Regression", "Ridge Regression")


def _mllib_estimator(model_name: str, task: str, n_classes: int) -> tuple[str, Any]:
    """The MLlib estimator standing in for the scikit-learn family selected.

    The same family, not the same implementation: MLlib has no extra-trees,
    stacking or RBF-kernel SVM, so those -- and anything unmatched -- are
    represented by a random forest, the closest distributed ensemble. MLlib's
    GBT is binary-only for classification, so a multi-class boosted winner is
    also represented by a random forest. The name of the stand-in is reported.
    """
    from pyspark.ml.classification import (GBTClassifier, LogisticRegression,
                                           RandomForestClassifier)
    from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor

    family = model_name.replace(" (log target)", "").replace(" (balanced)", "")
    forest_args = {"numTrees": CONFIG.spark_ml_trees, "maxDepth": CONFIG.spark_ml_depth,
                   "seed": 42}
    if task == "classification":
        if family.startswith(_LINEAR):
            return "LogisticRegression", LogisticRegression(maxIter=100)
        if family.startswith(_BOOSTED) and n_classes == 2:
            return "GBTClassifier", GBTClassifier(maxIter=50, maxDepth=5, seed=42)
        return "RandomForestClassifier", RandomForestClassifier(**forest_args)
    if family.startswith(_LINEAR):
        return "LinearRegression", LinearRegression(
            regParam=0.0 if family.startswith("Linear") else 1.0, elasticNetParam=0.0)
    if family.startswith(_BOOSTED):
        return "GBTRegressor", GBTRegressor(maxIter=50, maxDepth=5, seed=42)
    return "RandomForestRegressor", RandomForestRegressor(**forest_args)


def distributed_validation(sdf: "SparkDataFrame", target: str, features: list[str],
                           task: str, model_name: str, n_classes: int = 2,
                           test_fraction: float = 0.2) -> dict[str, Any]:
    """Train the selected model family on every row with Spark MLlib, and score it.

    scikit-learn fits on a bounded in-memory sample, so on a distributed run its
    cross-validated score describes the sample. This answers the question that
    leaves open -- does the result hold on the full dataset? -- by fitting the
    same family across the cluster on a random 80% of *all* rows and scoring it
    on the remaining 20%.

    It confirms; it does not replace. The delivered model and the reported
    scores stay scikit-learn's, so a result never depends on which engine ran.
    Derived features exist only in the in-memory frame, so this sees the raw
    columns; missing values are imputed inside the Spark pipeline.
    """
    import time

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler

    started = time.time()
    dtypes = dict(sdf.dtypes)
    feats = [f for f in features if f in dtypes and f != target]
    numeric = [f for f in feats if _is_numeric(dtypes[f])]
    # A categorical with thousands of levels is an identifier, not a feature;
    # one-hot encoding it across the cluster would cost much and mean nothing.
    categorical = []
    candidates = [f for f in feats if f not in numeric]
    if candidates:
        counts = sdf.agg(*[F.approx_count_distinct(c).alias(c) for c in candidates]).first()
        categorical = [c for c in candidates if counts[c] <= CONFIG.spark_ml_max_levels]
    if not numeric and not categorical:
        return {"status": "skipped", "reason": "no usable feature exists in the full data"}

    # Columns are renamed to internal names first. MLlib's stages write
    # columns called label, features, prediction and probability, and a
    # dataset whose target is named "label" -- among the most common target
    # names there is -- would collide with them and fail.
    num_in = [f"n{i}" for i in range(len(numeric))]
    cat_in = [f"c{i}" for i in range(len(categorical))]
    # Categoricals and a class label are cast to strings: MLlib's indexer
    # rejects boolean columns, and MiniBooNE's `signal` target is one.
    label = F.col(target).cast("string") if task == "classification" else F.col(target)
    data = sdf.select(*[F.col(c).cast("double").alias(a) for c, a in zip(numeric, num_in)],
                      *[F.col(c).cast("string").alias(a) for c, a in zip(categorical, cat_in)],
                      label.alias("y")).where(F.col("y").isNotNull())
    numeric, categorical, target = num_in, cat_in, "y"

    log_target = task == "regression" and "(log target)" in model_name
    stages: list[Any] = []
    if task == "classification":
        stages.append(StringIndexer(inputCol=target, outputCol="label",
                                    handleInvalid="skip"))
    else:
        label = F.col(target).cast("double")
        data = data.withColumn("label", F.log1p(label) if log_target else label)

    imputed = [f"{c}__imp" for c in numeric]
    if numeric:
        stages.append(Imputer(inputCols=numeric, outputCols=imputed, strategy="median"))
    encoded = []
    if categorical:
        indexed = [f"{c}__idx" for c in categorical]
        encoded = [f"{c}__ohe" for c in categorical]
        stages.append(StringIndexer(inputCols=categorical, outputCols=indexed,
                                    handleInvalid="keep"))
        stages.append(OneHotEncoder(inputCols=indexed, outputCols=encoded,
                                    handleInvalid="keep"))
    stages.append(VectorAssembler(inputCols=imputed + encoded, outputCol="features",
                                  handleInvalid="keep"))
    stand_in, estimator = _mllib_estimator(model_name, task, n_classes)
    stages.append(estimator)

    train, test = data.randomSplit([1 - test_fraction, test_fraction], seed=42)
    fitted = Pipeline(stages=stages).fit(train)
    scored = fitted.transform(test)

    result: dict[str, Any] = {"status": "ok", "selected_model": model_name,
                              "mllib_model": stand_in, "rows_train": train.count(),
                              "rows_test": test.count(), "features_used": len(feats)}
    if task == "classification":
        # Macro F1 from the confusion counts, so it is the same quantity the
        # in-memory evaluation reports; MLlib's built-in "f1" is weighted.
        cells = scored.groupBy("label", "prediction").count().collect()
        labels = {r["label"] for r in cells} | {r["prediction"] for r in cells}
        total = sum(r["count"] for r in cells)
        correct = sum(r["count"] for r in cells if r["label"] == r["prediction"])
        f1s = []
        for k in labels:
            tp = sum(r["count"] for r in cells if r["label"] == k and r["prediction"] == k)
            fp = sum(r["count"] for r in cells if r["label"] != k and r["prediction"] == k)
            fn = sum(r["count"] for r in cells if r["label"] == k and r["prediction"] != k)
            f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        result.update({"accuracy": round(correct / total, 4),
                       "f1_macro": round(float(np.mean(f1s)), 4)})
    else:
        pred = F.expm1("prediction") if log_target else F.col("prediction")
        actual = F.expm1("label") if log_target else F.col("label")
        stats = scored.select(((actual - pred) ** 2).alias("se"), actual.alias("y")).agg(
            F.mean("se").alias("mse"), F.variance("y").alias("var")).first()
        result.update({"r2": round(1 - stats["mse"] / stats["var"], 4) if stats["var"] else None,
                       "rmse": round(float(stats["mse"]) ** 0.5, 4)})
    result["elapsed_s"] = round(time.time() - started, 1)
    return result
