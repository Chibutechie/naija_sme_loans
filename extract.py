import logging

import pandas as pd
from datasets import load_dataset

from config import Config

print("[extract] Extract module loaded")

logger = logging.getLogger(__name__)


def extract(cfg: Config) -> pd.DataFrame:
    logger.info("Extracting dataset '%s' ...", cfg.DATASET_NAME)
    ds = load_dataset(cfg.DATASET_NAME, split=cfg.SPLIT)
    df = ds.to_pandas()
    logger.info("Extracted %s rows x %s cols", df.shape[0], df.shape[1])

    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    full_path = cfg.OUTPUT_DIR / cfg.FULL_FILE
    df.to_parquet(full_path, index=False)
    logger.info("Saved full dataset to %s", full_path)

    sample = df.sample(frac=cfg.SAMPLE_FRAC, random_state=cfg.RANDOM_STATE)
    sample_path = cfg.OUTPUT_DIR / cfg.SAMPLE_FILE
    sample.to_parquet(sample_path, index=False)
    logger.info("Saved %s-row sample to %s", sample.shape[0], sample_path)

    return df
