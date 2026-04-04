import os
from dataclasses import dataclass

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

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
        with open("params.yaml") as f:
            cfg = yaml.safe_load(f)["split"]
        return cls(**cfg)


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


def save_splits(train: pd.DataFrame, test: pd.DataFrame, cfg: SplitConfig) -> None:
    os.makedirs(os.path.dirname(cfg.train_output), exist_ok=True)
    train.to_csv(cfg.train_output, index=False)
    test.to_csv(cfg.test_output, index=False)


def main():
    cfg = SplitConfig.from_yaml()

    # 1. Load
    print("=== 1. Loading data ===")
    df = load_data(cfg.input)
    print(f"\tpath: {cfg.input}")
    print(f"\trows: {len(df)}, cols: {len(df.columns)}")

    # 2. Split
    print("\n=== 2. Splitting ===")
    train, test = perform_split(df, cfg)
    print(f"\ttest_size: {cfg.test_size}, stratified on: '{cfg.stratify_column}'")
    print(f"\ttrain rows: {len(train)}")
    print(f"\ttest rows:  {len(test)}")
    if cfg.stratify_column and cfg.stratify_column in df.columns:
        dist = df[cfg.stratify_column].value_counts(normalize=True).sort_index()
        print(f"\tclass distribution (full):\n{dist.to_string()}")

    # 3. Save
    print("\n=== 3. Saving ===")
    save_splits(train, test, cfg)
    print(f"\ttrain -> {cfg.train_output}")
    print(f"\ttest  -> {cfg.test_output}")


if __name__ == "__main__":
    main()
