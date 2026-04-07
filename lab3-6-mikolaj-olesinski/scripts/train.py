import json
import os
import tempfile
import time
import warnings
from dataclasses import dataclass

import gensim.downloader as gensim_api
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


_W2V_CACHE: dict = {}


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Average Word2Vec embeddings for a text column."""

    def __init__(self, model_name: str = "word2vec-google-news-300"):
        self.model_name = model_name

    def fit(self, X, y=None):
        if self.model_name not in _W2V_CACHE:
            print(f"\tloading W2V model '{self.model_name}' ...")
            _W2V_CACHE[self.model_name] = gensim_api.load(self.model_name)
        self._wv = _W2V_CACHE[self.model_name]
        self._dim = self._wv.vector_size
        return self

    def transform(self, X):
        texts = X if isinstance(X, (list, pd.Series)) else X.iloc[:, 0]
        vectors = []
        for text in texts:
            tokens = str(text).split()
            token_vecs = [self._wv[t] for t in tokens if t in self._wv]
            if token_vecs:
                vectors.append(np.mean(token_vecs, axis=0))
            else:
                vectors.append(np.zeros(self._dim))
        return np.vstack(vectors)


@dataclass
class TrainConfig:
    input_train: str
    input_test: str
    input_train_w2v: str
    input_test_w2v: str
    label_column: str
    output: str
    text_column: str
    bow_max_features: int
    cv_folds: int
    cv_sample: int | None
    random_state: int
    log_model: bool = False
    skip_cv: bool = False
    num_scaler: str = "minmax"  # options: minmax, standard, robust
    tag_key: str = "run_group"
    tag_value: str = "default"
    mlflow_experiment: str = "sentiment_analysis"
    experiment: str = "all_features"
    active_model: str = "linear_svc"
    vectorizer: str = "bow"
    w2v_model: str = "word2vec-google-news-300"
    feature_selection: str = "none"
    n_top_features: int = 10000
    n_components: int = 100
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


def make_vectorizer(vectorizer: str, max_features: int, w2v_model: str = "word2vec-google-news-300"):
    if vectorizer == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
    elif vectorizer == "word2vec" or vectorizer == "w2v":
        return Word2VecTransformer(model_name=w2v_model)
    elif vectorizer == "bow":
        return CountVectorizer(max_features=max_features, ngram_range=(1, 1))
    else:
        raise ValueError(f"Unknown vectorizer '{vectorizer}'. Choose from: 'bow', 'tfidf', 'word2vec'.")


def make_preprocessor(
    experiment: str,
    text_col: str,
    num_cols: list[str],
    cat_cols: list[str],
    bow_max_features: int,
    vectorizer: str,
    w2v_model: str = "word2vec-google-news-300",
    num_scaler: str = "minmax",
) -> ColumnTransformer:
    scalers = {
        "minmax":   MinMaxScaler(),
        "standard": StandardScaler(),
        "robust":   RobustScaler(),
    }
    if num_scaler not in scalers:
        raise ValueError(f"Unknown num_scaler '{num_scaler}'. Choose from: {list(scalers.keys())}")
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", scalers[num_scaler]),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    vec = make_vectorizer(vectorizer, bow_max_features, w2v_model)

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


def make_feature_selector(feature_selection: str, n_top_features: int, n_components: int, random_state: int):
    """Return a feature selection/reduction step, or None if disabled.

    Options:
      - none:          no feature selection
      - select_k_best: SelectKBest(chi2) — picks top-k features by chi2 score
      - truncated_svd: TruncatedSVD (LSA) — reduces to n_components dimensions
      - both:          SelectKBest first, then TruncatedSVD
    """
    if feature_selection == "select_k_best":
        return [
            ("feature_selection", SelectKBest(chi2, k=n_top_features)),
        ]
    if feature_selection == "truncated_svd":
        return [
            ("feature_selection", TruncatedSVD(n_components=n_components, random_state=random_state)),
        ]
    if feature_selection == "both":
        return [
            ("feature_selection", SelectKBest(chi2, k=n_top_features)),
            ("dim_reduction", TruncatedSVD(n_components=n_components, random_state=random_state)),
        ]
    if feature_selection != "none":
        raise ValueError(f"Unknown feature_selection '{feature_selection}'. Choose from: none, select_k_best, truncated_svd, both")
    return []


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


def run_cv(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int, random_state: int, n_jobs: int = 2) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    raw = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    result = {}
    for metric in SCORING:
        vals = raw[f"test_{metric}"]
        result[f"{metric}_mean"] = round(float(vals.mean()), 4)
        result[f"{metric}_std"]  = round(float(vals.std()), 4)
    return result


def plot_confusion_matrix(y_true, y_pred, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def prepare_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("")
    df["review_title"] = df["review_title"].fillna("")
    df[text_column] = df["review_text"] + " " + df["review_title"]
    return df


def main():
    cfg = TrainConfig.from_yaml()

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    run_name = f"{cfg.experiment}__{cfg.active_model}__{cfg.vectorizer}"

    with mlflow.start_run(run_name=run_name):
        if cfg.tag_key and cfg.tag_value:
            mlflow.set_tags({cfg.tag_key: cfg.tag_value})

        # Log parameters
        mlflow.log_params({
            "experiment": cfg.experiment,
            "active_model": cfg.active_model,
            "vectorizer": cfg.vectorizer,
            "bow_max_features": cfg.bow_max_features,
            "cv_folds": cfg.cv_folds,
            "cv_sample": cfg.cv_sample,
            **{f"model_{k}": v for k, v in (cfg.models or {}).get(cfg.active_model, {}).items()},
        })

        # 1. Load data
        print("=== 1. Loading data ===")
        is_w2v = cfg.vectorizer == "word2vec"
        train_path = cfg.input_train_w2v if is_w2v else cfg.input_train
        test_path  = cfg.input_test_w2v  if is_w2v else cfg.input_test
        train = pd.read_csv(train_path, low_memory=False)
        print(f"\tpath: {train_path}")
        print(f"\trows: {len(train):,}, cols: {len(train.columns)}")

        # 2. Prepare features
        print("\n=== 2. Preparing features ===")
        train = prepare_features(train, cfg.text_column)
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
        preprocessor = make_preprocessor(cfg.experiment, cfg.text_column, num_cols, cat_cols, cfg.bow_max_features, cfg.vectorizer, cfg.w2v_model, cfg.num_scaler)
        selector_steps = make_feature_selector(cfg.feature_selection, cfg.n_top_features, cfg.n_components, cfg.random_state)
        model = make_model(cfg.active_model, cfg.models or {}, cfg.random_state)
        print(f"\texperiment:       {cfg.experiment}")
        print(f"\tmodel:            {cfg.active_model}")
        print(f"\tvectorizer:       {cfg.vectorizer} (max_features={cfg.bow_max_features:,})")
        print(f"\tfeature_selection:{cfg.feature_selection}")
        print(f"\tnum_scaler:       {cfg.num_scaler}")
        print(f"\tCV folds:         {cfg.cv_folds}")

        mlflow.log_params({
            "feature_selection": cfg.feature_selection,
            "n_top_features": cfg.n_top_features,
            "n_components": cfg.n_components,
            "num_scaler": cfg.num_scaler,
        })

        # 5. Run CV
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            *selector_steps,
            ("clf", model),
        ])
        if cfg.skip_cv:
            print("\n=== 5. Skipping CV (skip_cv=true) ===")
        else:
            print("\n=== 5. Running CV ===")
            t0 = time.time()
            cv_jobs = 1 if cfg.vectorizer in ("word2vec", "w2v") else 2
            scores = run_cv(pipeline, X, y, cfg.cv_folds, cfg.random_state, n_jobs=cv_jobs)
            elapsed = time.time() - t0
            print(f"\tf1_macro={scores['f1_macro_mean']:.4f} ± {scores['f1_macro_std']:.4f} ({elapsed:.0f}s)")
            for k, v in scores.items():
                mlflow.log_metric(f"cv_{k}", v)

        # 6. Fit and save model
        print("\n=== 6. Fitting and saving model ===")
        pipeline.fit(X, y)
        if cfg.log_model:
            mlflow.sklearn.log_model(pipeline, name="model")
            print("\tmodel logged to MLflow")
        else:
            print("\tlog_model=false, skipping")

        # 7. Confusion matrix - train set
        print("\n=== 7. Confusion matrices ===")
        y_train_pred = pipeline.predict(X)
        fig_train = plot_confusion_matrix(y, y_train_pred, f"Train – {cfg.active_model} / {cfg.vectorizer}")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "confusion_matrix_train.png")
            fig_train.savefig(path)
            mlflow.log_artifact(path, artifact_path="confusion_matrices")
        plt.close(fig_train)
        print("\tlogged confusion matrix (train)")

        # Test set
        test = pd.read_csv(test_path, low_memory=False)
        test = prepare_features(test, cfg.text_column)
        X_test = test.drop(columns=[cfg.label_column])
        y_test = test[cfg.label_column]
        y_test_pred = pipeline.predict(X_test)
        fig_test = plot_confusion_matrix(y_test, y_test_pred, f"Test – {cfg.active_model} / {cfg.vectorizer}")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "confusion_matrix_test.png")
            fig_test.savefig(path)
            mlflow.log_artifact(path, artifact_path="confusion_matrices")
        plt.close(fig_test)
        print("\tlogged confusion matrix (test)")

        # Log test metrics
        test_metrics = {
            "test_f1_macro":    round(f1_score(y_test, y_test_pred, average="macro"), 4),
            "test_f1_weighted": round(f1_score(y_test, y_test_pred, average="weighted"), 4),
            "test_accuracy":    round(accuracy_score(y_test, y_test_pred), 4),
            "test_precision":   round(precision_score(y_test, y_test_pred, average="macro"), 4),
            "test_recall":      round(recall_score(y_test, y_test_pred, average="macro"), 4),
        }
        mlflow.log_metrics(test_metrics)
        print(f"\ttest f1_macro={test_metrics['test_f1_macro']:.4f}")

        # 8. Save metrics JSON
        print("\n=== 8. Saving results ===")
        os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
        output = {
            "vectorizer": cfg.vectorizer,
            "cv_sample": cfg.cv_sample,
            **(scores if not cfg.skip_cv else {}),
            **test_metrics,
        }
        with open(cfg.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\tresults -> {cfg.output}")


if __name__ == "__main__":
    main()
