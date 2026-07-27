import os
from pathlib import Path

print("[config] Config module loaded")

class Config:
    DATASET_NAME = "electricsheepafrica/africa-synth-banking-sme-loans-nigeria"
    SPLIT = "train"
    SAMPLE_FRAC = 0.10
    RANDOM_STATE = 42

    DB_USER = "postgres"
    DB_PASSWORD = "root"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "naija_sme_loans"

    OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "data"))
    SAMPLE_FILE = "sme_loans_sample.parquet"
    FULL_FILE = "sme_loans_full.parquet"
    CLEAN_FILE = "sme_loans_clean.parquet"
