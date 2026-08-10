import logging

import numpy as np
import pandas as pd

from config import Config

print("[transform] Transform module loaded")

logger = logging.getLogger(__name__)

_CATEGORICALS = ["transaction_type", "channel", "status", "location_state"]


def _clean_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in _CATEGORICALS:
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip().str.upper()
    return df


def _fill_unknowns(df: pd.DataFrame) -> pd.DataFrame:
    df["merchant_category_code"] = (
        df["merchant_category_code"].fillna("").astype(str).str.strip().str.upper()
    )
    df["merchant_category_code"] = np.where(
        df["merchant_category_code"] == "", "UNKNOWN", df["merchant_category_code"]
    )
    df["merchant_name"] = np.where(
        df["merchant_name"].astype(str).str.strip() == "", "UNKNOWN", df["merchant_name"]
    )
    df["device_id"] = np.where(
        df["device_id"].astype(str).str.strip() == "", "UNKNOWN", df["device_id"]
    )
    return df


def transform(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logger.info("Starting transformation ...")
    n_before = df.shape[0]

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    n_null = df["transaction_id"].isna().sum() + df["timestamp"].isna().sum()
    if n_null:
        logger.warning("Dropping %s rows with null id/timestamp", n_null)
        df = df.dropna(subset=["transaction_id", "timestamp"])

    n_nonpos = (df["amount_ngn"] <= 0).sum()
    if n_nonpos:
        logger.warning("Dropping %s rows with non-positive amounts", n_nonpos)
        df = df[df["amount_ngn"] > 0]

    n_dup = df["transaction_id"].duplicated().sum()
    if n_dup:
        logger.warning("Dropping %s duplicate transaction ids", n_dup)
        df = df.drop_duplicates(subset="transaction_id")

    df["transaction_date"] = df["timestamp"].dt.normalize()
    df["tx_year"] = df["timestamp"].dt.year
    df["tx_quarter"] = df["timestamp"].dt.quarter
    df["tx_month"] = df["timestamp"].dt.month
    df["tx_hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    df = _clean_categoricals(df)
    df = _fill_unknowns(df)

    sign = np.where(df["transaction_type"] == "CREDIT", 1, -1)
    expected_balance = df["balance_before_ngn"] + sign * df["amount_ngn"]
    df["balance_integrity_ok"] = np.isclose(
        df["balance_after_ngn"], expected_balance, rtol=1e-9, atol=1.0
    ).astype(int)

    logger.info("Kept %s of %s rows", df.shape[0], n_before)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = cfg.OUTPUT_DIR / cfg.CLEAN_FILE
    df.to_parquet(clean_path, index=False)
    logger.info("Saved clean dataset (%s rows) to %s", df.shape[0], clean_path)

    return df


def analyze(df: pd.DataFrame):
    logger.info("Generating analysis summary ...")
    print("\n" + "=" * 60)
    print("TRANSACTION SUMMARY")
    print("=" * 60)
    print(f"Rows: {df.shape[0]:,} | Period: "
          f"{df['timestamp'].min()} -> {df['timestamp'].max()}")

    print("\n" + "=" * 60)
    print("MIX BY TRANSACTION TYPE")
    print("=" * 60)
    print(df.groupby("transaction_type")["amount_ngn"].agg(["count", "sum", "mean"]).round(2))

    print("\n" + "=" * 60)
    print("MIX BY STATUS")
    print("=" * 60)
    print(df["status"].value_counts())

    print("\n" + "=" * 60)
    print("MIX BY CHANNEL")
    print("=" * 60)
    print(df["channel"].value_counts())

    print("\n" + "=" * 60)
    print("FRAUD RATE BY CHANNEL")
    print("=" * 60)
    print(df.groupby("channel")["fraud_flag"].mean().sort_values(ascending=False).round(4))

    print("\n" + "=" * 60)
    print("TOP 10 STATES BY FRAUD RATE")
    print("=" * 60)
    print(
        df.groupby("location_state")["fraud_flag"]
        .mean()
        .round(4)
        .sort_values(ascending=False)
        .head(10)
    )

    print("\n" + "=" * 60)
    print("TOP 10 MERCHANTS BY VOLUME")
    print("=" * 60)
    print(df["merchant_name"].value_counts().head(10))

    print("\n" + "=" * 60)
    print("BALANCE INTEGRITY")
    print("=" * 60)
    print(df["balance_integrity_ok"].value_counts())
