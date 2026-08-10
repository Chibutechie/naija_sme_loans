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
    CREATE TABLE IF NOT EXISTS dim_customer (
        customer_id VARCHAR(30) PRIMARY KEY
    );

    CREATE TABLE IF NOT EXISTS dim_account (
        account_id  VARCHAR(30) PRIMARY KEY,
        customer_id VARCHAR(30) REFERENCES dim_customer(customer_id)
    );

    CREATE TABLE IF NOT EXISTS dim_merchant (
        merchant_id            SERIAL PRIMARY KEY,
        merchant_name          VARCHAR(100),
        merchant_category_code VARCHAR(10),
        UNIQUE (merchant_name, merchant_category_code)
    );

    CREATE TABLE IF NOT EXISTS dim_location (
        location_id    SERIAL PRIMARY KEY,
        location_lga   VARCHAR(50),
        location_state VARCHAR(50),
        UNIQUE (location_lga, location_state)
    );

    CREATE TABLE IF NOT EXISTS fact_transactions (
        transaction_id       VARCHAR(50) PRIMARY KEY,
        account_id           VARCHAR(30) REFERENCES dim_account(account_id),
        merchant_id          INTEGER REFERENCES dim_merchant(merchant_id),
        location_id          INTEGER REFERENCES dim_location(location_id),
        timestamp            TIMESTAMP,
        transaction_date     DATE,
        amount_ngn           decimal(14,2),
        balance_before_ngn   decimal(14,2),
        balance_after_ngn    decimal(14,2),
        transaction_type     VARCHAR(10),
        channel              VARCHAR(10),
        status               VARCHAR(10),
        device_id            VARCHAR(50),
        fraud_flag           BOOLEAN,
        tx_year              INTEGER,
        tx_quarter           INTEGER,
        tx_month             INTEGER,
        tx_hour              INTEGER,
        day_of_week          INTEGER,
        is_weekend           BOOLEAN,
        balance_integrity_ok BOOLEAN
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
        conn.execute(text("DROP TABLE IF EXISTS fact_transactions CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_merchant CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_location CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_account CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS dim_customer CASCADE"))
        conn.commit()
    logger.info("Existing tables dropped")


def _load_dim_customer(df: pd.DataFrame, engine) -> None:
    dim_customer = df[["customer_id"]].drop_duplicates().reset_index(drop=True)
    dim_customer.to_sql("dim_customer", engine, if_exists="append", index=False)
    logger.info("Loaded %s customers", len(dim_customer))


def _load_dim_account(df: pd.DataFrame, engine) -> None:
    dim_account = df[["account_id", "customer_id"]].drop_duplicates(
        subset="account_id"
    )
    dim_account.to_sql("dim_account", engine, if_exists="append", index=False)
    logger.info("Loaded %s accounts", len(dim_account))


def _load_dim_merchant(df: pd.DataFrame, engine) -> None:
    dim_merchant = df[["merchant_name", "merchant_category_code"]].drop_duplicates()
    dim_merchant.to_sql("dim_merchant", engine, if_exists="append", index=False)
    logger.info("Loaded %s merchants", len(dim_merchant))


def _load_dim_location(df: pd.DataFrame, engine) -> None:
    dim_location = df[["location_lga", "location_state"]].drop_duplicates()
    dim_location.to_sql("dim_location", engine, if_exists="append", index=False)
    logger.info("Loaded %s locations", len(dim_location))


def load(df: pd.DataFrame, cfg: Config):
    logger.info("Loading data into PostgreSQL ...")
    engine = _get_engine(cfg)

    _drop_tables(engine)
    _create_tables(engine)

    _load_dim_customer(df, engine)
    _load_dim_account(df, engine)
    _load_dim_merchant(df, engine)
    _load_dim_location(df, engine)

    merchant_map = pd.read_sql(
        "SELECT merchant_id, merchant_name, merchant_category_code FROM dim_merchant",
        engine,
    )
    location_map = pd.read_sql(
        "SELECT location_id, location_lga, location_state FROM dim_location",
        engine,
    )

    df_fact = df.copy()
    df_fact = df_fact.merge(
        merchant_map,
        on=["merchant_name", "merchant_category_code"],
        how="left",
    )
    df_fact = df_fact.merge(
        location_map,
        on=["location_lga", "location_state"],
        how="left",
    )
    df_fact["is_weekend"] = df_fact["is_weekend"].astype(bool)
    df_fact["balance_integrity_ok"] = df_fact["balance_integrity_ok"].astype(bool)

    fact_cols = [
        "transaction_id", "account_id", "merchant_id", "location_id",
        "timestamp", "transaction_date", "amount_ngn",
        "balance_before_ngn", "balance_after_ngn",
        "transaction_type", "channel", "status", "device_id",
        "fraud_flag", "tx_year", "tx_quarter", "tx_month", "tx_hour",
        "day_of_week", "is_weekend", "balance_integrity_ok",
    ]
    df_fact = df_fact[fact_cols]

    CHUNK_SIZE = 100_000
    total = len(df_fact)
    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)
        chunk = df_fact.iloc[start:end]
        chunk.to_sql("fact_transactions", engine, if_exists="append", index=False)
        logger.info("Loaded rows %s-%s of %s into fact_transactions", start + 1, end, total)

    logger.info("Finished loading %s transaction records into fact_transactions", total)

    result = pd.read_sql(
        """
        SELECT 'dim_customer' AS tbl, COUNT(*) AS cnt FROM dim_customer
        UNION ALL
        SELECT 'dim_account', COUNT(*) FROM dim_account
        UNION ALL
        SELECT 'dim_merchant', COUNT(*) FROM dim_merchant
        UNION ALL
        SELECT 'dim_location', COUNT(*) FROM dim_location
        UNION ALL
        SELECT 'fact_transactions', COUNT(*) FROM fact_transactions
        """,
        engine,
    )
    print("\n" + "=" * 60)
    print("POSTGRES LOAD VERIFICATION")
    print("=" * 60)
    print(result.to_string(index=False))
