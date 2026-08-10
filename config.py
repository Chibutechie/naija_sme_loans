import os
from pathlib import Path

print("[config] Config module loaded")

class Config:
    DATASET_NAME = "electricsheepafrica/nigerian-banking-retail-transactions"
    SPLIT = "train"
    SAMPLE_FRAC = 0.10
    RANDOM_STATE = 42

    DB_USER = "postgres"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "bank_platform"

    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data"))
    FULL_FILE = "transactions_full.parquet"
    SAMPLE_FILE = "transactions_sample.parquet"
    CLEAN_FILE = "transactions_clean.parquet"