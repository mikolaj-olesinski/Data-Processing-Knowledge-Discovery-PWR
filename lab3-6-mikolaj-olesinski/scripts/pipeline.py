import os
import tempfile

import gensim.downloader as gensim_api
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import LinearSVC

from utils import setup_logger

logger = setup_logger()


_W2V_CACHE: dict = {}


class Word2VecTransformer(BaseEstimator, TransformerMixin):
    """Average Word2Vec embeddings for a text column."""

    def __init__(self, model_name: str = "word2vec-google-news-300"):
        self.model_name = model_name

    def fit(self, X, y=None):
        if self.model_name not in _W2V_CACHE:
            logger.info(f"\tloading W2V model '{self.model_name}' ...")
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


def make_vectorizer(
    vectorizer: str, max_features: int, w2v_model: str = "word2vec-google-news-300"
):
    if vectorizer == "tfidf":
        return TfidfVectorizer(max_features=max_features, ngram_range=(1, 1))
    elif vectorizer in ("word2vec", "w2v"):
        return Word2VecTransformer(model_name=w2v_model)
    elif vectorizer == "bow":
        return CountVectorizer(max_features=max_features, ngram_range=(1, 1))
    else:
        raise ValueError(
            f"Unknown vectorizer '{vectorizer}'. Choose from: 'bow', 'tfidf', 'word2vec'."
        )


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
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
    }
    if num_scaler not in scalers:
        raise ValueError(
            f"Unknown num_scaler '{num_scaler}'. Choose from: {list(scalers.keys())}"
        )
    num_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", scalers[num_scaler]),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    vec = make_vectorizer(vectorizer, bow_max_features, w2v_model)

    options = {
        "text_only": ColumnTransformer(
            [
                ("vec", vec, text_col),
            ],
            remainder="drop",
        ),
        "non_text": ColumnTransformer(
            [
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        ),
        "all_features": ColumnTransformer(
            [
                ("vec", vec, text_col),
                ("num", num_pipe, num_cols),
                ("cat", cat_pipe, cat_cols),
            ],
            remainder="drop",
        ),
    }
    if experiment not in options:
        raise ValueError(
            f"Unknown experiment '{experiment}'. Choose from: {list(options.keys())}"
        )
    return options[experiment]


def make_feature_selector(
    feature_selection: str, n_top_features: int, n_components: int, random_state: int
):
    """Return a feature selection/reduction step, or [] if disabled.

    Options:
      - none:          no feature selection
      - select_k_best: SelectKBest(chi2) — picks top-k features by chi2 score
      - truncated_svd: TruncatedSVD (LSA) — reduces to n_components dimensions
      - both:          SelectKBest first, then TruncatedSVD
    """
    if feature_selection == "select_k_best":
        return [("feature_selection", SelectKBest(chi2, k=n_top_features))]
    if feature_selection == "truncated_svd":
        return [
            (
                "feature_selection",
                TruncatedSVD(n_components=n_components, random_state=random_state),
            )
        ]
    if feature_selection == "both":
        return [
            ("feature_selection", SelectKBest(chi2, k=n_top_features)),
            (
                "dim_reduction",
                TruncatedSVD(n_components=n_components, random_state=random_state),
            ),
        ]
    if feature_selection != "none":
        raise ValueError(
            f"Unknown feature_selection '{feature_selection}'. Choose from: none, select_k_best, truncated_svd, both"
        )
    return []


def make_model(active_model: str, models_cfg: dict, random_state: int):
    options = {
        "dummy": DummyClassifier(
            **models_cfg.get("dummy", {}), random_state=random_state
        ),
        "linear_svc": LinearSVC(
            **models_cfg.get("linear_svc", {}), random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            **models_cfg.get("random_forest", {}), n_jobs=2, random_state=random_state
        ),
    }
    if active_model not in options:
        raise ValueError(
            f"Unknown model '{active_model}'. Choose from: {list(options.keys())}"
        )
    return options[active_model]


SCORING = {
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
}


def run_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
    n_jobs: int = 2,
) -> dict:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    raw = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=n_jobs)
    result = {}
    for metric in SCORING:
        vals = raw[f"test_{metric}"]
        result[f"{metric}_mean"] = round(float(vals.mean()), 4)
        result[f"{metric}_std"] = round(float(vals.std()), 4)
    return result


def plot_confusion_matrix(y_true, y_pred, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def log_confusion_matrix(
    fig: plt.Figure, filename: str, mlflow_enabled: bool = True
) -> None:
    if mlflow_enabled:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, filename)
            fig.savefig(path)
            mlflow.log_artifact(path, artifact_path="confusion_matrices")
    plt.close(fig)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "test_f1_macro": round(f1_score(y_true, y_pred, average="macro"), 4),
        "test_f1_weighted": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "test_accuracy": round(accuracy_score(y_true, y_pred), 4),
        "test_precision": round(precision_score(y_true, y_pred, average="macro"), 4),
        "test_recall": round(recall_score(y_true, y_pred, average="macro"), 4),
    }
