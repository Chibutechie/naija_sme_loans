import logging

import pandas as pd
from sqlalchemy import create_engine, text

from config import Config

print("[load] Load module loaded")

logger = logging.getLogger(__name__)


def _get_engine(cfg: Config):
    url = (
        f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}"
        f"@{cfg.DB_HOST}:{cfg.DB_PORT}/{cfg.DB_NAME}"
    )
    return create_engine(url, pool_pre_ping=True)


def _create_tables(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_business (
        business_id         VARCHAR(20) PRIMARY KEY,
        business_sector     VARCHAR(50),
        business_state      VARCHAR(50),
        years_in_business   INTEGER,
        annual_revenue_ngn  decimal(12,2),
        num_employees       INTEGER
    );

    CREATE TABLE IF NOT EXISTS dim_lender (
        lender_id   SERIAL PRIMARY KEY,
        lender_name VARCHAR(100) UNIQUE
    );

    CREATE TABLE IF NOT EXISTS fact_loans (
        loan_id                  VARCHAR(20) PRIMARY KEY,
        business_id              VARCHAR(20) REFERENCES dim_business(business_id),
        lender_id                INTEGER REFERENCES dim_lender(lender_id),
        application_date         DATE,
        disbursement_date        DATE,
        principal_ngn            decimal(12,2),
        interest_rate_annual     decimal(12,2),
        tenor_months             INTEGER,
        collateral_value_ngn     decimal(12,2),
        credit_score             INTEGER,
        default_90d              BOOLEAN,
        default_180d             BOOLEAN,
        loan_to_collateral_ratio decimal(12,2),
        revenue_to_loan_ratio    decimal(12,2),
        processing_days          INTEGER,
        loan_year                INTEGER,
        loan_quarter             INTEGER,
        loan_month               INTEGER,
        risk_tier                VARCHAR(20),
        loan_size_segment        VARCHAR(30)
    );
    """
    with engine.connect() as conn:
        for statement in ddl.split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt + ";"))
        conn.commit()
    logger.info("Database tables created / verified")


def _drop_tables(engine):
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fact_loans CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_business CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_lender CASCADE"))
        conn.commit()
    logger.info("Existing tables dropped")


def load(df: pd.DataFrame, cfg: Config):
    logger.info("Loading data into PostgreSQL ...")
    engine = _get_engine(cfg)

    _drop_tables(engine)
    _create_tables(engine)

    lenders = df[["lender"]].drop_duplicates().reset_index(drop=True)
    lenders.columns = ["lender_name"]
    lenders.to_sql("dim_lender", engine, if_exists="append", index=False)
    logger.info("Loaded %s lenders", len(lenders))

    dim_business = df[
        [
            "business_id",
            "business_sector",
            "business_state",
            "years_in_business",
            "annual_revenue_ngn",
            "num_employees",
        ]
    ].drop_duplicates(subset="business_id")
    dim_business.to_sql("dim_business", engine, if_exists="append", index=False)
    logger.info("Loaded %s businesses", len(dim_business))

    lender_map = pd.read_sql(
        "SELECT lender_id, lender_name FROM dim_lender", engine
    )
    df_fact = df.merge(lender_map, left_on="lender", right_on="lender_name")

    fact_cols = [
        "loan_id", "business_id", "lender_id",
        "application_date", "disbursement_date", "principal_ngn",
        "interest_rate_annual", "tenor_months", "collateral_value_ngn",
        "credit_score", "default_90d", "default_180d",
        "loan_to_collateral_ratio", "revenue_to_loan_ratio",
        "processing_days", "loan_year", "loan_quarter", "loan_month",
        "risk_tier", "loan_size_segment",
    ]
    df_fact = df_fact[fact_cols]

    CHUNK_SIZE = 100_000
    total = len(df_fact)
    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk = df_fact.iloc[start:end]
        chunk.to_sql("fact_loans", engine, if_exists="append", index=False)
        logger.info("Loaded rows %s–%s of %s into fact_loans", start + 1, end, total)

    logger.info("Finished loading %s loan records into fact_loans", total)

    result = pd.read_sql(
        """
        SELECT 'dim_lender' AS tbl, COUNT(*) AS cnt FROM dim_lender
        UNION ALL
        SELECT 'dim_business', COUNT(*) FROM dim_business
        UNION ALL
        SELECT 'fact_loans', COUNT(*) FROM fact_loans
        """,
        engine,
    )
    print("\n" + "=" * 60)
    print("POSTGRES LOAD VERIFICATION")
    print("=" * 60)
    print(result.to_string(index=False))
