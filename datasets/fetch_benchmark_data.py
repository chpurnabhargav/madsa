"""Download the public datasets used for the comparison with published systems.

    python datasets/fetch_benchmark_data.py

Kaggle House Prices is the dataset Data Interpreter reports on (their ML-Benchmark),
so it gives a like-for-like row in the comparison. The others are larger than
anything in the original evaluation, to show the framework holds up past the
point where the in-memory engine must sample and Spark takes over.

All come from OpenML or scikit-learn, so no Kaggle account is needed.
"""
from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_covtype, fetch_openml

HERE = Path(__file__).parent


def _save(df, name: str) -> None:
    path = HERE / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} columns, "
          f"{path.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    # Kaggle "House Prices: Advanced Regression Techniques", training split.
    houses = fetch_openml(data_id=42165, as_frame=True, parser="auto").frame
    _save(houses.drop(columns=["Id"], errors="ignore"), "kaggle_house_prices")

    # UCI Bank Marketing: will the client subscribe? 45k rows, 12% positive.
    bank = fetch_openml(data_id=1461, as_frame=True, parser="auto").frame
    bank = bank.rename(columns={"Class": "subscribed"})
    _save(bank, "bank_marketing")

    # Credit card fraud: 284k transactions, 0.17% fraud. Over 50 MB, so Spark.
    fraud = fetch_openml(data_id=1597, as_frame=True, parser="auto").frame
    _save(fraud, "credit_card_fraud")

    # Forest cover type: 581k rows, 7 classes. Over 500k rows, so Spark.
    cover = fetch_covtype(as_frame=True).frame
    _save(cover, "covertype")

    # IBM Telco churn (TabArena curation): will the customer leave?
    telco = fetch_openml(data_id=46915, as_frame=True, parser="auto").frame
    _save(telco, "telco_churn")

    # UCI MiniBooNE: particle identification, ~130k rows, binary `signal`.
    # (OpenML also hosts guillermo/41159, but its served file no longer matches
    # its registered checksum, so fetching it is not reproducible.)
    mini = fetch_openml(data_id=41150, as_frame=True, parser="auto").frame
    _save(mini, "miniboone")


if __name__ == "__main__":
    main()
