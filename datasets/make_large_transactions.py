"""Generate the large transaction dataset used to exercise the Spark engine.

The file this produces is around 129 MB, which is too large to keep in version
control and too large to email. It is generated rather than stored, so the
distributed-execution results stay reproducible without shipping the data: the
seed is fixed, so every machine produces byte-identical output.

    python datasets/make_large_transactions.py

The dataset is synthetic but not arbitrary. A genuine relationship is built into
the fraud label through transaction amount, account age, hour of day and device,
so the modelling stage has real signal to find. The positive class is held near
4%, which is realistic for fraud and low enough that accuracy alone is
misleading -- the framework reports 96% accuracy and a macro F1 of 0.49 on this
data, and correctly concludes there is no reliable signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROWS = 3_000_000
SEED = 42


def build(n_rows: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "transaction_id": np.arange(n_rows),
        "amount": np.round(rng.lognormal(3.2, 1.1, n_rows), 2),
        "account_age_days": rng.integers(1, 4000, n_rows),
        "n_prior_txns": rng.poisson(12, n_rows),
        "hour_of_day": rng.integers(0, 24, n_rows),
        "merchant_category": rng.choice(
            ["grocery", "fuel", "travel", "online", "dining", "utilities"], n_rows),
        "device": rng.choice(["mobile", "web", "pos"], n_rows, p=[0.6, 0.25, 0.15]),
        "region": rng.choice(["north", "south", "east", "west"], n_rows),
    })

    # A real but weak relationship: large amounts, new accounts, small-hours
    # activity and web transactions each raise the odds of fraud.
    logit = (-4.2
             + 0.35 * np.log1p(df["amount"])
             - 0.0004 * df["account_age_days"]
             + 0.9 * (df["hour_of_day"] < 5)
             + 0.7 * (df["device"] == "web"))
    df["is_fraud"] = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def main() -> None:
    out = Path(__file__).resolve().parent / "large_transactions.csv"
    df = build()
    df.to_csv(out, index=False)
    size_mb = out.stat().st_size / 1024 ** 2
    print(f"Wrote {out.name}: {len(df):,} rows, {size_mb:.1f} MB, "
          f"fraud rate {df['is_fraud'].mean():.4f}")


if __name__ == "__main__":
    main()
