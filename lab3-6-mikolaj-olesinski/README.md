# Sentiment Analysis — Sephora Skincare Reviews

A reproducible ML pipeline for binary sentiment classification of cosmetics reviews
from the [Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews) dataset.
Ratings 1–3 → negative sentiment, 4–5 → positive.

Built for the PWDIOW course (assignments 3–6), with a strong focus on
**reproducibility, data versioning, and experiment tracking**.

## Stack

- **ML / NLP:** scikit-learn, pandas, gensim (Word2Vec), spaCy, SHAP
- **MLOps:** DVC (pipeline + data versioning), MLflow (experiment tracking)
- **Infra:** Docker + Docker Compose (MLflow, MinIO as an S3 remote for DVC, Jupyter)
- **Reporting:** Jupyter + papermill, matplotlib, seaborn
- **Code quality:** ruff

## Pipeline (DVC)

```
preprocess → split → clean_text / clean_text_w2v → train / tune → shap
                                                  ↘ EDA (papermill)
```

The whole pipeline is defined declaratively in `dvc.yaml` and parametrized via `params.yaml`.
Modeling is built on `sklearn.Pipeline` + `ColumnTransformer` to prevent
data leakage during cross-validation.

## What's inside

- **Preprocessing & feature engineering** — data merge, missing-value handling,
  derived features (text length, exclamation marks, caps ratio), spaCy text cleaning
  (lemmatization, stopwords, negation tagging)
- **Vectorization** — comparison of BoW, TF-IDF, and Word2Vec (pretrained `word2vec-google-news-300`)
- **Feature selection** — SelectKBest (chi²) + dimensionality reduction with TruncatedSVD
- **Models** — DummyClassifier (baseline), LinearSVC, LogisticRegression, MultinomialNB, RandomForest
- **Hyperparameter tuning** — comparison of GridSearch / RandomSearch / HalvingGridSearch /
  HalvingRandomSearch (quality vs. runtime vs. number of iterations)
- **Explainability (SHAP)** — analysis of features driving sentiment and of
  misclassified examples (DVC `foreach`)
- **EDA** — exploratory report in Jupyter

## Best result

LinearSVC + TF-IDF + SelectKBest(5000):

| Metric      | Value |
| ----------- | ----- |
| F1-macro    | 0.87  |
| Accuracy    | 0.92  |
| F1-weighted | 0.92  |

## Running

```bash
# build the image
make build

# start the environment (MLflow + MinIO + Jupyter)
docker compose up

# reproduce the full pipeline
dvc repro
```

MLflow UI: `http://localhost:5001` · Jupyter: `http://localhost:8888`
