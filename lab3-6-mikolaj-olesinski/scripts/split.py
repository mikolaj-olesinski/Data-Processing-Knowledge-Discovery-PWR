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


def main():
    cfg = SplitConfig.from_yaml()

    df = pd.read_csv(cfg.input, **CSV_OPTIONS)

    stratify_labels = df[cfg.stratify_column] if cfg.stratify and cfg.stratify_column else None

    train, test = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=stratify_labels,
    )

    os.makedirs(os.path.dirname(cfg.train_output), exist_ok=True)
    train.to_csv(cfg.train_output, index=False)
    test.to_csv(cfg.test_output, index=False)
    print(f"Train: {len(train)} rows, Test: {len(test)} rows")


if __name__ == "__main__":
    main()