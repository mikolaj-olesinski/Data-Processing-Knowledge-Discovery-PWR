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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
    experiment: str = "all_features"
    active_model: str = "linear_svc"
    vectorizer: str = "bow"
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
    num_cols = [c for c in df.select_dtypes(include="number").columns if c in remaining]
    cat_cols = [c for c in df.select_dtypes(include=["object", "string"]).columns if c in remaining]
    return num_cols, cat_cols


def make_vectorizer(vectorizer: str, max_features: int):
    if vectorizer == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
    return CountVectorizer(max_features=max_features, ngram_range=(1, 1))


def make_preprocessor(
    experiment: str,
    text_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    bow_max_features: int,
    vectorizer: str,
) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    vec = make_vectorizer(vectorizer, bow_max_features)

    options = {
        "text_only": ColumnTransformer([
            ("vec", vec, text_col),
        ], remainder="drop"),
        "non_text": ColumnTransformer([
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ], remainder="drop"),
        "all_features": ColumnTransformer([
            ("vec", vec, text_col),
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ], remainder="drop"),
    }
    if experiment not in options:
        raise ValueError(f"Unknown experiment '{experiment}'. Choose from: {list(options.keys())}")
    return options[experiment]


def make_model(active_model: str, models_cfg: dict, random_state: int):
    dummy_cfg = models_cfg.get("dummy", {})
    svc_cfg   = models_cfg.get("linear_svc", {})
    rf_cfg    = models_cfg.get("random_forest", {})
    options = {
        "dummy":         DummyClassifier(**dummy_cfg, random_state=random_state),
        "linear_svc":    LinearSVC(**svc_cfg, random_state=random_state),
        "random_forest": RandomForestClassifier(**rf_cfg, n_jobs=2, random_state=random_state),
    }
    if active_model not in options:
        raise ValueError(f"Unknown model '{active_model}'. Choose from: {list(options.keys())}")
    return options[active_model]


SCORING = {
    "f1_macro":    "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy":    "accuracy",
    "precision":   "precision_macro",
    "recall":      "recall_macro",
}


def run_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    raw = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
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
    original_len = len(train)
    if cfg.cv_sample and len(train) > cfg.cv_sample:
        train = train.sample(cfg.cv_sample, random_state=cfg.random_state)
        print(f"\tsampled {cfg.cv_sample:,} rows for CV (from {original_len:,})")
    print(f"\tfinal CV set: {len(train):,} rows")

    X = train.drop(columns=[cfg.label_column])
    y = train[cfg.label_column]

    # 3. Detect columns
    print("\n=== 3. Detecting columns ===")
    num_cols, cat_cols = detect_columns(train, cfg.label_column, cfg.text_column, cfg.drop_from_features)
    print(f"\ttext:        '{cfg.text_column}'")
    print(f"\tnumerical:   {len(num_cols)} cols: {num_cols}")
    print(f"\tcategorical: {len(cat_cols)} cols: {cat_cols}")

    # 4. Build preprocessor and model
    print("\n=== 4. Building preprocessor and model ===")
    preprocessor = make_preprocessor(cfg.experiment, cfg.text_column, num_cols, cat_cols, cfg.bow_max_features, cfg.vectorizer)
    model = make_model(cfg.active_model, cfg.models or {}, cfg.random_state)
    print(f"\texperiment:  {cfg.experiment}")
    print(f"\tmodel:       {cfg.active_model}")
    print(f"\tvectorizer:  {cfg.vectorizer} (max_features={cfg.bow_max_features:,})")
    print(f"\tCV folds:    {cfg.cv_folds}")

    # 5. Run CV
    print("\n=== 5. Running CV ===")
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", model),
    ])
    t0 = time.time()
    scores = run_cv(pipeline, X, y, cfg.cv_folds, cfg.random_state)
    elapsed = time.time() - t0
    print(f"\tf1_macro={scores['f1_macro_mean']:.4f} ± {scores['f1_macro_std']:.4f} ({elapsed:.0f}s)")

    # 6. Fit and save model
    print("\n=== 6. Fitting and saving model ===")
    pipeline.fit(X, y)
    os.makedirs(cfg.models_dir, exist_ok=True)
    model_path = os.path.join(cfg.models_dir, f"{cfg.experiment}_{cfg.active_model}.pkl")
    joblib.dump(pipeline, model_path)
    print(f"\tmodel -> {model_path}")

    # 7. Save metrics
    print("\n=== 7. Saving results ===")
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    output = {
        "model": cfg.active_model,
        "experiment": cfg.experiment,
        "vectorizer": cfg.vectorizer,
        "f1_macro_mean": scores["f1_macro_mean"],
        "recall_mean": scores["recall_mean"],
        "precision_mean": scores["precision_mean"],
        "accuracy_mean": scores["accuracy_mean"],
        "f1_weighted_mean": scores["f1_weighted_mean"],
    }
    with open(cfg.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\tresults -> {cfg.output}")


if __name__ == "__main__":
    main()