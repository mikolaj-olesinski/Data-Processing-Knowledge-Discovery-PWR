import json
import os
import time
import warnings
from dataclasses import dataclass

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

# Suppress UndefinedMetricWarning from DummyClassifier (predicts single class)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Columns to exclude from features (IDs, secondary text, noisy text lists)
@dataclass
class TrainConfig:
    input_train: str
    input_test: str
    label_column: str
    output: str
    text_column: str
    bow_max_features: int
    cv_folds: int
    cv_sample: int | None
    random_state: int
    models_dir: str
    drop_from_features: list[str] = None
    models: dict = None

    @classmethod
    def from_yaml(cls) -> "TrainConfig":
        with open("params.yaml") as f:
            cfg = yaml.safe_load(f)["train"]
        return cls(**cfg)


def detect_columns(df: pd.DataFrame, target: str, text_col: str, drop_from_features: list[str]) -> tuple[list, list]:
    """Return (num_cols, cat_cols) excluding target, text_col and drop_from_features."""
    exclude = set(drop_from_features or []) | {target, text_col}
    remaining = [c for c in df.columns if c not in exclude]

    num_cols = [
        c for c in df.select_dtypes(include="number").columns if c in remaining
    ]
    cat_cols = [
        c for c in df.select_dtypes(include=["object", "string"]).columns if c in remaining
    ]
    return num_cols, cat_cols


def make_preprocessors(
    text_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    bow_max_features: int,
) -> dict[str, ColumnTransformer]:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    bow = CountVectorizer(max_features=bow_max_features, ngram_range=(1, 1))

    return {
        "text_only": ColumnTransformer([
            ("bow", bow, text_col),
        ], remainder="drop"),
        "non_text": ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ], remainder="drop"),
        "all_features": ColumnTransformer([
            ("bow", bow, text_col),
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ], remainder="drop"),
    }


def make_models(models_cfg: dict, random_state: int) -> dict:
    dummy_cfg = models_cfg.get("dummy", {})
    svc_cfg   = models_cfg.get("linear_svc", {})
    rf_cfg    = models_cfg.get("random_forest", {})
    return {
        "dummy":         DummyClassifier(**dummy_cfg, random_state=random_state),
        "linear_svc":    LinearSVC(**svc_cfg, random_state=random_state),
        "random_forest": RandomForestClassifier(**rf_cfg, n_jobs=2, random_state=random_state),
    }


SCORING = {
    "f1_macro":    "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy":    "accuracy",
    "precision":   "precision_macro",
    "recall":      "recall_macro",
}

def run_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    raw = cross_validate(
        pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1
    )
    result = {}
    for metric in SCORING:
        vals = raw[f"test_{metric}"]
        result[f"{metric}_mean"] = round(float(vals.mean()), 4)
        result[f"{metric}_std"]  = round(float(vals.std()), 4)
    return result


def main():
    cfg = TrainConfig.from_yaml()

    # 1. Load data
    print("=== 1. Loading data ===")
    train = pd.read_csv(cfg.input_train, low_memory=False)
    print(f"\tpath: {cfg.input_train}")
    print(f"\trows: {len(train):,}, cols: {len(train.columns)}")

    # 2. Prepare features
    print("\n=== 2. Preparing features ===")
    train["review_text"] = train["review_text"].fillna("")
    train["review_title"] = train["review_title"].fillna("")
    train[cfg.text_column] = train["review_text"] + " " + train["review_title"]
    print(f"\tcombined 'review_text' + 'review_title' -> '{cfg.text_column}'")
    if cfg.cv_sample and len(train) > cfg.cv_sample:
        train = train.sample(cfg.cv_sample, random_state=cfg.random_state)
        print(f"\tsampled {cfg.cv_sample:,} rows for CV (from {len(train):,})")
    print(f"\tfinal CV set: {len(train):,} rows")

    X = train.drop(columns=[cfg.label_column])
    y = train[cfg.label_column]

    # 3. Detect columns
    print("\n=== 3. Detecting columns ===")
    num_cols, cat_cols = detect_columns(train, cfg.label_column, cfg.text_column, cfg.drop_from_features)
    print(f"\ttext:        '{cfg.text_column}'")
    print(f"\tnumerical:   {len(num_cols)} cols: {num_cols}")
    print(f"\tcategorical: {len(cat_cols)} cols: {cat_cols}")

    # 4. Build preprocessors and models
    print("\n=== 4. Building preprocessors and models ===")
    preprocessors = make_preprocessors(cfg.text_column, num_cols, cat_cols, cfg.bow_max_features)
    models = make_models(cfg.models or {}, cfg.random_state)
    print(f"\texperiments: {list(preprocessors.keys())}")
    print(f"\tmodels:      {list(models.keys())}")
    print(f"\tBoW max features: {cfg.bow_max_features:,}, CV folds: {cfg.cv_folds}")

    # 5. Run experiments
    print("\n=== 5. Running experiments ===")
    total = len(preprocessors) * len(models)
    results = {}

    with tqdm(total=total, desc="Experiments", unit="run") as pbar:
        for exp_name, preprocessor in preprocessors.items():
            results[exp_name] = {}
            for model_name, model in models.items():
                pbar.set_description(f"{exp_name} / {model_name}")
                t0 = time.time()

                pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("clf", model),
                ])
                scores = run_cv(pipeline, X, y, cfg.cv_folds, cfg.random_state)
                elapsed = time.time() - t0

                results[exp_name][model_name] = scores
                tqdm.write(
                    f"\t[{exp_name}/{model_name}] "
                    f"f1_macro={scores['f1_macro_mean']:.4f} ± {scores['f1_macro_std']:.4f} "
                    f"({elapsed:.0f}s)"
                )

                pipeline.fit(X, y)
                os.makedirs(cfg.models_dir, exist_ok=True)
                model_path = os.path.join(cfg.models_dir, f"{exp_name}_{model_name}.pkl")
                joblib.dump(pipeline, model_path)
                tqdm.write(f"\t  saved -> {model_path}")

                pbar.update(1)

    # 6. Save results
    print("\n=== 6. Saving results ===")
    output = {
        "config": {
            "cv_folds": cfg.cv_folds,
            "cv_sample": cfg.cv_sample,
            "bow_max_features": cfg.bow_max_features,
            "random_state": cfg.random_state,
            "models": cfg.models,
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with open(cfg.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\tresults -> {cfg.output}")


if __name__ == "__main__":
    main()
