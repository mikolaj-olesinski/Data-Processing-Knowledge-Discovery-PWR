import logging
import os

import pandas as pd
import yaml


def load_yaml_section(section: str) -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)[section]


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def setup_logger(name: str = __name__) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(name)
