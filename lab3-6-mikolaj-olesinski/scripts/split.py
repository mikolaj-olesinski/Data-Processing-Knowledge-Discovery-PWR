from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_yaml_section, save_csv, setup_logger

logger = setup_logger()

CSV_OPTIONS = {"low_memory": False}


@dataclass
class SplitConfig:
    input: str
    train_output: str
    test_output: str
    test_size: float
    random_state: int
    stratify: bool
    stratify_column: str | None = None

    @classmethod
    def from_yaml(cls) -> "SplitConfig":
        return cls(**load_yaml_section("split"))


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_OPTIONS)


def perform_split(
    df: pd.DataFrame, cfg: SplitConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratify_labels = (
        df[cfg.stratify_column] if cfg.stratify and cfg.stratify_column else None
    )
    return train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_labels,
    )


def main():
    cfg = SplitConfig.from_yaml()

    # 1. Load
    logger.info("=== 1. Loading data ===")
    df = load_data(cfg.input)
    logger.info(f"\tpath: {cfg.input}")
    logger.info(f"\trows: {len(df)}, cols: {len(df.columns)}")

    # 2. Split
    logger.info("\n=== 2. Splitting ===")
    train, test = perform_split(df, cfg)
    logger.info(f"\ttest_size: {cfg.test_size}, stratified on: '{cfg.stratify_column}'")
    logger.info(f"\ttrain rows: {len(train)}")
    logger.info(f"\ttest rows:  {len(test)}")
    if cfg.stratify_column and cfg.stratify_column in df.columns:
        dist = df[cfg.stratify_column].value_counts(normalize=True).sort_index()
        logger.info(f"\tclass distribution (full):\n{dist.to_string()}")

    # 3. Save
    logger.info("\n=== 3. Saving ===")
    save_csv(train, cfg.train_output)
    save_csv(test, cfg.test_output)
    logger.info(f"\ttrain -> {cfg.train_output}")
    logger.info(f"\ttest  -> {cfg.test_output}")


if __name__ == "__main__":
    main()
