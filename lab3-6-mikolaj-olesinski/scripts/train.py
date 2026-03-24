import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

@dataclass
class TrainConfig:
    input_train: str
    input_test: str
    label_column: str
    model: str
    strategy: str
    output: str

    @classmethod
    def from_yaml(cls) -> "TrainConfig":
        with open("params.yaml") as f:
            cfg = yaml.safe_load(f)["train"]
        return cls(**cfg)


def predict_uniform_continuous(y_train: list, n_samples: int, seed: int = 42) -> list:
    classes = sorted(set(y_train))
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=n_samples)
    indices = (values * len(classes)).astype(int).clip(0, len(classes) - 1)
    return [classes[i] for i in indices]


def main():
    cfg = TrainConfig.from_yaml()

    train = pd.read_csv(cfg.input_train)
    test = pd.read_csv(cfg.input_test)

    y_train = train[cfg.label_column].tolist()
    y_test = test[cfg.label_column].tolist()

    if cfg.model == "uniform_continuous":
        y_pred = predict_uniform_continuous(y_train, n_samples=len(y_test))
    else:
        clf = DummyClassifier(strategy=cfg.strategy)
        clf.fit([[0]] * len(y_train), y_train)
        y_pred = clf.predict([[0]] * len(y_test)).tolist()

    metrics = {
        "model": cfg.model if cfg.model == "uniform_continuous" else f"dummy_{cfg.strategy}",
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "train_rows": len(train),
        "test_rows": len(test),
    }

    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with open(cfg.output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"F1 macro: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()