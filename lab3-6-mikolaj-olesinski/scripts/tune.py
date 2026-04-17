import json
import os

import time
import warnings

from contextlib import contextmanager
from dataclasses import dataclass, field

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pipeline import compute_metrics, make_feature_selector, make_preprocessor
from train import detect_columns, prepare_features
from utils import load_yaml_section, setup_logger

logger = setup_logger()
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.model_selection._search")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.feature_selection._univariate_selection")
os.environ.setdefault(
    "PYTHONWARNINGS",
    "ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning",
)


@dataclass
class TuneConfig:
    input_train: str
    input_test: str
    output: str
    sample_size: int
    cv_folds: int
    random_state: int
    n_iter_random: int
    n_candidates_halving: int | str
    mlflow_enabled: bool
    mlflow_experiment: str
    experiment: str
    vectorizer: str
    bow_max_features: int
    num_scaler: str
    feature_selection: str
    n_top_features: int
    text_column: str
    label_column: str
    drop_from_features: list[str] = field(default_factory=list)
    n_components: int | None = None
    w2v_model: str = "word2vec-google-news-300"
    models: dict = field(default_factory=dict)
    active_models: list[str] = field(default_factory=list)
    search_methods: list[str] = field(default_factory=list)
    tag_key: str = "run_group"
    tag_value: str = "default"

    @classmethod
    def from_yaml(cls) -> "TuneConfig":
        return cls(**load_yaml_section("tune"))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _load_and_prepare(path: str, text_column: str) -> pd.DataFrame:
    return prepare_features(pd.read_csv(path, low_memory=False), text_column)


def _expand_param_value(value):
    # Allow either plain lists ([0.01, 0.1, 1]) or DSL dicts:
    #   {logspace: [start, stop, num]}            -> np.logspace(start, stop, num)
    #   {logspace: [start, stop, num, base]}      -> np.logspace(..., base=base)
    #   {linspace: [start, stop, num]}            -> np.linspace(start, stop, num)
    #   {range: [start, stop, step]}              -> list(range(...))
    if not isinstance(value, dict):
        return value
    if len(value) != 1:
        raise ValueError(f"param_grid dict must have exactly one key, got: {value}")
    (kind, args), = value.items()
    if kind == "logspace":
        start, stop, num, *rest = args
        base = rest[0] if rest else 10.0
        return np.logspace(start, stop, int(num), base=base).tolist()
    if kind == "linspace":
        start, stop, num = args
        return np.linspace(start, stop, int(num)).tolist()
    if kind == "range":
        return list(range(*args))
    raise ValueError(f"Unknown param_grid spec: {kind}")


def _expand_param_grid(param_grid: dict) -> dict:
    return {k: _expand_param_value(v) for k, v in param_grid.items()}


@contextmanager
def _tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def main():
    cfg = TuneConfig.from_yaml()

    if cfg.mlflow_enabled:
        mlflow.set_tracking_uri(
            os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment(cfg.mlflow_experiment)

    # 1. Load data
    logger.info("=== 1. Loading data ===")
    train_full = _load_and_prepare(cfg.input_train, cfg.text_column)
    test = _load_and_prepare(cfg.input_test, cfg.text_column)
    train = train_full.sample(cfg.sample_size, random_state=cfg.random_state)
    logger.info(f"\ttrain sample: {len(train):,} rows | test: {len(test):,} rows")

    X_train = train.drop(columns=[cfg.label_column])
    y_train = train[cfg.label_column]
    X_test = test.drop(columns=[cfg.label_column])
    y_test = test[cfg.label_column]

    # 2. Detect columns
    num_cols, cat_cols = detect_columns(
        train, cfg.label_column, cfg.text_column, cfg.drop_from_features
    )
    logger.info(f"\tnumerical: {len(num_cols)} | categorical: {len(cat_cols)}")

    # 3. Build models and parameter grids from config
    _MODEL_ALIASES = {
        "linear_svc": "svm",
        "logistic_regression": "lr",
        "multinomial_nb": "nb",
    }

    _MODEL_CLASSES = {
        "linear_svc": LinearSVC,
        "logistic_regression": LogisticRegression,
        "multinomial_nb": MultinomialNB,
    }
    # Models that accept random_state
    _RANDOM_STATE_MODELS = {"linear_svc", "logistic_regression"}

    active_models = cfg.active_models or list(cfg.models.keys())

    MODELS = {}
    for model_name, model_cfg in cfg.models.items():
        if model_name not in active_models:
            continue
        ModelClass = _MODEL_CLASSES[model_name]
        extra = {"random_state": cfg.random_state} if model_name in _RANDOM_STATE_MODELS else {}
        model = ModelClass(**model_cfg.get("params", {}), **extra)
        MODELS[model_name] = (model, _expand_param_grid(model_cfg["param_grid"]))

    cv = StratifiedKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state
    )

    SEARCH_METHODS = {
        "grid_search": (GridSearchCV, {}),
        "random_search": (RandomizedSearchCV, {"n_iter": cfg.n_iter_random}),
        "halving_grid_search": (
            HalvingGridSearchCV,
            {"random_state": cfg.random_state},
        ),
        "halving_random_search": (
            HalvingRandomSearchCV,
            {
                "n_candidates": cfg.n_candidates_halving,
                "random_state": cfg.random_state,
            },
        ),
    }

    active_methods = cfg.search_methods or list(SEARCH_METHODS.keys())

    all_results = []

    # 4. Run tuning for each model × method
    for model_name, (model, param_grid) in MODELS.items():
        logger.info(f"\n=== Model: {model_name} ===")

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
            cfg.n_components or 100,
            cfg.random_state,
        )
        pipeline = Pipeline(
            [("preprocessor", preprocessor), *selector_steps, ("clf", model)]
        )

        for method_name, (SearchClass, extra_kwargs) in SEARCH_METHODS.items():
            if method_name not in active_methods:
                continue
            logger.info(f"\t→ {method_name}")

            search = SearchClass(
                pipeline,
                param_grid,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                refit=True,
                **extra_kwargs,
            )

            t0 = time.perf_counter()
            with _tqdm_joblib(tqdm(desc=f"{model_name}/{method_name}", leave=False)):
                search.fit(X_train, y_train)
            fit_time = round(time.perf_counter() - t0, 2)

            n_candidates_total = len(search.cv_results_["params"])
            cv_best_f1 = round(float(search.best_score_), 4)

            y_pred = search.best_estimator_.predict(X_test)
            test_metrics = compute_metrics(y_test, y_pred)

            logger.info(
                f"\t\tcv_f1={cv_best_f1:.4f} | test_f1={test_metrics['test_f1_macro']:.4f}"
                f" | time={fit_time:.1f}s | n_models={n_candidates_total}"
            )

            result = {
                "model": model_name,
                "method": method_name,
                "cv_best_f1_macro": cv_best_f1,
                "test_f1_macro": test_metrics["test_f1_macro"],
                "fit_time_seconds": fit_time,
                "n_candidates_total": n_candidates_total,
                "best_params": search.best_params_,
            }
            all_results.append(result)

            # Log to MLflow
            if cfg.mlflow_enabled:
                alias = _MODEL_ALIASES.get(model_name, model_name)
                base_tags = {"model": model_name, "method": method_name}
                if cfg.tag_key and cfg.tag_value:
                    base_tags[cfg.tag_key] = cfg.tag_value

                cv_res = search.cv_results_
                best_idx = search.best_index_
                n_candidates_total = len(cv_res["params"])

                for i, params in enumerate(cv_res["params"]):
                    is_best = i == best_idx
                    params_clean = {k.replace("clf__", ""): v for k, v in params.items()}
                    param_str = "__".join(f"{k}={v}" for k, v in params_clean.items())
                    run_name = f"{alias}__{method_name}__{param_str}"

                    with mlflow.start_run(run_name=run_name):
                        mlflow.set_tags({**base_tags, "is_best": str(is_best).lower()})
                        mlflow.log_params(
                            {
                                "model": model_name,
                                "method": method_name,
                                "sample_size": cfg.sample_size,
                                "n_candidates_total": n_candidates_total,
                                **params_clean,
                            }
                        )

                        cv_metrics = {
                            "cv_mean_f1_macro": float(cv_res["mean_test_score"][i]),
                            "cv_std_f1_macro": float(cv_res["std_test_score"][i]),
                            "rank_test_score": float(cv_res["rank_test_score"][i]),
                            "mean_fit_time": float(cv_res["mean_fit_time"][i]),
                            "mean_score_time": float(cv_res["mean_score_time"][i]),
                        }
                        for fold in range(cfg.cv_folds):
                            key = f"split{fold}_test_score"
                            if key in cv_res:
                                cv_metrics[f"cv_fold{fold}_f1_macro"] = float(cv_res[key][i])

                        if is_best:
                            mlflow.log_metrics({**cv_metrics, **test_metrics, "fit_time_seconds": fit_time})
                            mlflow.sklearn.log_model(search.best_estimator_, name="best_estimator")
                        else:
                            mlflow.log_metrics(cv_metrics)

    # 5. Save summary results JSON
    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    with open(cfg.output, "w") as f:
        json.dump(all_results, f, indent=2, cls=_NumpyEncoder)
    logger.info(f"\n=== Results saved -> {cfg.output} ===")


if __name__ == "__main__":
    main()
