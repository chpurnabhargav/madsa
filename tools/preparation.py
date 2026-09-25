"""Replaying a run's data preparation inside the saved model.

The model a report recommends was fitted on the analysis's working frame, which
is not the file the user has. Before modelling, the quality stage may have
converted numbers stored as text, normalised the case of categories and clipped
outliers, and the feature stage may have added derived columns -- a missingness
flag, a logarithm, a grouped category. The saved pipeline's signature told users
to pass raw rows, but it listed the derived columns among the ones it needed,
and `predict.py` had no way to build them: scoring the original file failed with
"Missing column(s) the model needs: Fare_log", and a text-to-number repair meant
the pipeline received strings where it had been fitted on floats.

`RawRowPreparer` closes that gap. It carries the preparation steps recorded
during the run, with the parameters fitted at the time, and applies them to raw
rows as the first step of the saved pipeline. The saved object then genuinely
takes rows in the shape of the original dataset.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from tools.features import apply_recipe
from tools.quality import replay_cleaning


def plan_inputs(model_columns: list[str],
                steps: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """The raw columns a model needs, and the steps that turn them into its inputs.

    A derived column is replaced by the column it was built from. Only the steps
    that touch what the model actually uses are kept, so a repair to a column
    the model never saw does not make that column a required input.
    """
    derived = {s["feature"]: s for s in steps if s.get("stage") == "features"}
    raw: list[str] = []
    for column in model_columns:
        source = derived[column]["source"] if column in derived else column
        if source not in raw:
            raw.append(source)
    kept = [s for s in steps
            if (s.get("stage") == "features" and s["feature"] in model_columns)
            or (s.get("stage") == "quality" and s.get("column") in raw)]
    return raw, kept


class RawRowPreparer(BaseEstimator, TransformerMixin):
    """Turn raw rows into the columns the model was fitted on.

    Stateless: every parameter it applies was fitted during the analysis and is
    carried in `steps`, so fitting it learns nothing and it is fitted by
    construction. Applied to rows that are already prepared it changes nothing,
    because each step is idempotent -- converting a numeric column to numbers,
    lower-casing lower-case text, clipping within the fence already applied.
    """

    def __init__(self, steps: list[dict[str, Any]] | None = None,
                 input_columns: list[str] | None = None,
                 output_columns: list[str] | None = None) -> None:
        self.steps = steps
        self.input_columns = input_columns
        self.output_columns = output_columns

    def fit(self, X: Any, y: Any = None) -> "RawRowPreparer":
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        missing = [c for c in (self.input_columns or []) if c not in frame.columns]
        if missing:
            raise ValueError("Missing column(s) the model needs: "
                             + ", ".join(map(str, missing)))
        steps = list(self.steps or [])
        frame = replay_cleaning(frame, [s for s in steps if s.get("stage") == "quality"])
        frame = apply_recipe(frame, [s for s in steps if s.get("stage") == "features"])
        return frame[list(self.output_columns or frame.columns)]

    def __sklearn_is_fitted__(self) -> bool:
        return True
