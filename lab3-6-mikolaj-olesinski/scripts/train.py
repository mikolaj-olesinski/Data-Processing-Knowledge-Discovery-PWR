import json
import os
import warnings
from contextlib import nullcontext
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.pipeline import Pipeline

from pipeline import (
    compute_metrics,
    log_confusion_matrix,
    make_feature_selector,
    make_model,
    make_preprocessor,
    plot_confusion_matrix,
)
from utils import load_yaml_section, setup_logger

logger = setup_logger()
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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
    random_state: int
    log_model: bool = False
    mlflow_enabled: bool = True
    num_scaler: str = "minmax"
    tag_key: str = "run_group"
    tag_value: str = "default"
    mlflow_experiment: str = "sentiment_analysis"
    experiment: str = "all_features"
    active_model: str = "linear_svc"
    vectorizer: str = "bow"
    w2v_model: str = "word2vec-google-news-300"
    feature_selection: str = "none"
    n_top_features: int = 10000
    n_components: int | None = None
    drop_from_features: list[str] = None
    models: dict = None

    @classmethod
    def from_yaml(cls) -> "TrainConfig":
        return cls(**load_yaml_section("train"))


def detect_columns(
    df: pd.DataFrame, target: str, text_col: str, drop_from_features: list[str]
) -> tuple[list, list]:
    """Return (num_cols, cat_cols) excluding target, text_col and drop_from_features."""
    exclude = set(drop_from_features or []) | {target, text_col}
    remaining = [c for c in df.columns if c not in exclude]
    num_cols = [c for c in df.select_dtypes(include="number").columns if c in remaining]
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "string"]).columns
        if c in remaining
    ]
    return num_cols, cat_cols


def prepare_features(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    df = df.copy()
    df["review_text"] = df["review_text"].fillna("")
    df["review_title"] = df["review_title"].fillna("")
    df[text_column] = df["review_text"] + " " + df["review_title"]
    return df


def _load_and_prepare(path: str, text_column: str) -> pd.DataFrame:
    return prepare_features(pd.read_csv(path, low_memory=False), text_column)


def main():
    cfg = TrainConfig.from_yaml()

    if cfg.mlflow_enabled:
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment(cfg.mlflow_experiment)

    run_name = f"{cfg.experiment}__{cfg.active_model}__{cfg.vectorizer}"
    run_ctx = (
        mlflow.start_run(run_name=run_name) if cfg.mlflow_enabled else nullcontext()
    )

    with run_ctx:
        if cfg.mlflow_enabled:
            if cfg.tag_key and cfg.tag_value:
                mlflow.set_tags({cfg.tag_key: cfg.tag_value})
            mlflow.log_params(
                {
                    "experiment": cfg.experiment,
                    "active_model": cfg.active_model,
                    "vectorizer": cfg.vectorizer,
                    "bow_max_features": cfg.bow_max_features,
                    "feature_selection": cfg.feature_selection,
                    "n_top_features": cfg.n_top_features,
                    "n_components": cfg.n_components,
                    "num_scaler": cfg.num_scaler,
                    **{
                        f"model_{k}": v
                        for k, v in (cfg.models or {}).get(cfg.active_model, {}).items()
                    },
                }
            )

        # 1. Load data
        logger.info("=== 1. Loading data ===")
        is_w2v = cfg.vectorizer in ("word2vec", "w2v")
        train = _load_and_prepare(
            cfg.input_train_w2v if is_w2v else cfg.input_train, cfg.text_column
        )
        test = _load_and_prepare(
            cfg.input_test_w2v if is_w2v else cfg.input_test, cfg.text_column
        )
        logger.info(f"\ttrain: {len(train):,} rows | test: {len(test):,} rows")

        # 2. Prepare features
        logger.info("\n=== 2. Preparing features ===")
        X = train.drop(columns=[cfg.label_column])
        y = train[cfg.label_column]
        X_test = test.drop(columns=[cfg.label_column])
        y_test = test[cfg.label_column]

        # 3. Detect columns
        logger.info("\n=== 3. Detecting columns ===")
        num_cols, cat_cols = detect_columns(
            train, cfg.label_column, cfg.text_column, cfg.drop_from_features
        )
        logger.info(
            f"\tnumerical: {len(num_cols)} | categorical: {len(cat_cols)} | text: '{cfg.text_column}'"
        )

        # 4. Build pipeline
        logger.info("\n=== 4. Building pipeline ===")
        preprocessor = make_preprocessor(
            cfg.experiment,
            cfg.text_column,
            num_cols,
            cat_cols,
            cfg.bow_max_features,
            cfg.vectorizer,
            cfg.w2v_model,
            cfg.num_scaler,
        )
        selector_steps = make_feature_selector(
            cfg.feature_selection,
            cfg.n_top_features,
            cfg.n_components,
            cfg.random_state,
        )
        model = make_model(cfg.active_model, cfg.models or {}, cfg.random_state)
        pipeline = Pipeline(
            [("preprocessor", preprocessor), *selector_steps, ("clf", model)]
        )
        logger.info(
            f"\t{cfg.experiment} | {cfg.active_model} | vec={cfg.vectorizer} | sel={cfg.feature_selection} | scaler={cfg.num_scaler}"
        )

        # 5. Fit on full training data
        logger.info("\n=== 5. Fitting model ===")
        pipeline.fit(X, y)
        if cfg.log_model and cfg.mlflow_enabled:
            mlflow.sklearn.log_model(pipeline, name="model")
            logger.info("\tmodel logged to MLflow")

        # 6. Evaluate on test set
        logger.info("\n=== 6. Evaluating ===")
        log_confusion_matrix(
            plot_confusion_matrix(y_test, pipeline.predict(X_test), f"Test – {cfg.active_model}"),
            "confusion_matrix_test.png",
            cfg.mlflow_enabled,
        )
        y_test_pred = pipeline.predict(X_test)
        test_metrics = compute_metrics(y_test, y_test_pred)
        if cfg.mlflow_enabled:
            mlflow.log_metrics(test_metrics)
        logger.info(f"\ttest f1_macro={test_metrics['test_f1_macro']:.4f}")

        # 7. Save results
        logger.info("\n=== 7. Saving results ===")
        os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(
                {
                    "vectorizer": cfg.vectorizer,
                    **test_metrics,
                },
                f,
                indent=2,
            )
        logger.info(f"\tresults -> {cfg.output}")


if __name__ == "__main__":
    main()
