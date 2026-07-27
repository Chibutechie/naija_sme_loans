import logging

import numpy as np
import pandas as pd

from config import Config

print("[transform] Transform module loaded")

logger = logging.getLogger(__name__)


def _risk_tier(score):
    if score < 580:
        return "High Risk"
    if score < 720:
        return "Medium Risk"
    return "Low Risk"


def _loan_segment(amount):
    if amount < 5_000_000:
        return "Micro (< N5M)"
    if amount < 20_000_000:
        return "Small (N5M-N20M)"
    return "Medium (N20M+)"


def transform(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    logger.info("Starting transformation ...")

    df = df.copy()
    df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
    df["disbursement_date"] = pd.to_datetime(df["disbursement_date"], errors="coerce")

    null_dates = df["application_date"].isna().sum() + df["disbursement_date"].isna().sum()
    if null_dates:
        logger.warning("Dropping %s rows with null dates", null_dates)
        df = df.dropna(subset=["application_date", "disbursement_date"])

    invalid_dates = df["disbursement_date"] < df["application_date"]
    n_invalid = invalid_dates.sum()
    if n_invalid:
        logger.warning(
            "Found %s records where disbursement < application. Fixing by swapping dates.",
            n_invalid,
        )
        mask = invalid_dates
        df.loc[mask, ["application_date", "disbursement_date"]] = df.loc[
            mask, ["disbursement_date", "application_date"]
        ].values

    df["loan_to_collateral_ratio"] = np.where(
        df["collateral_value_ngn"] == 0,
        np.nan,
        df["principal_ngn"] / df["collateral_value_ngn"],
    )
    df["revenue_to_loan_ratio"] = df["annual_revenue_ngn"] / df["principal_ngn"]
    df["processing_days"] = (
        df["disbursement_date"] - df["application_date"]
    ).dt.days
    df["loan_year"] = df["application_date"].dt.year
    df["loan_quarter"] = df["application_date"].dt.quarter
    df["loan_month"] = df["application_date"].dt.month

    df["risk_tier"] = df["credit_score"].apply(_risk_tier)
    df["loan_size_segment"] = df["principal_ngn"].apply(_loan_segment)

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_path = cfg.OUTPUT_DIR / cfg.CLEAN_FILE
    df.to_parquet(clean_path, index=False)
    logger.info("Saved clean dataset (%s rows) to %s", df.shape[0], clean_path)

    return df


def analyze(df: pd.DataFrame):
    logger.info("Generating analysis summary ...")
    print("\n" + "=" * 60)
    print("DEFAULT RATE BY RISK TIER")
    print("=" * 60)
    print(df.groupby("risk_tier")["default_90d"].mean().round(3))

    print("\n" + "=" * 60)
    print("DEFAULT RATE BY LOAN SIZE SEGMENT")
    print("=" * 60)
    print(df.groupby("loan_size_segment")["default_90d"].mean().round(3))

    print("\n" + "=" * 60)
    print("TOP 10 STATES BY DEFAULT RATE")
    print("=" * 60)
    print(
        df.groupby("business_state")["default_90d"]
        .mean()
        .round(3)
        .sort_values(ascending=False)
        .head(10)
    )

    corr = df[
        [
            "years_in_business",
            "annual_revenue_ngn",
            "num_employees",
            "principal_ngn",
            "interest_rate_annual",
            "tenor_months",
            "collateral_value_ngn",
            "credit_score",
            "loan_to_collateral_ratio",
            "revenue_to_loan_ratio",
            "processing_days",
            "default_90d",
            "default_180d",
        ]
    ].copy()
    corr["default_90d"] = corr["default_90d"].astype(int)
    corr["default_180d"] = corr["default_180d"].astype(int)

    print("\n" + "=" * 60)
    print("CORRELATION WITH default_90d")
    print("=" * 60)
    print(corr.corr()["default_90d"].sort_values(ascending=False))
