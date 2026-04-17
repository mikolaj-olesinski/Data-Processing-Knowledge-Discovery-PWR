import argparse
import os
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from train import prepare_features
from utils import load_yaml_section, setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger()


def _get_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names_all = np.array(preprocessor.get_feature_names_out())
    selector = pipeline.named_steps.get("feature_selection")
    if selector is not None:
        feature_names_all = feature_names_all[selector.get_support()]
    return [
        n.replace("vec__", "").replace("num__", "").replace("cat__", "")
        for n in feature_names_all
    ]


def _patch_nb_as_linear(clf):
    """Add coef_/intercept_ to MultinomialNB (binary) so LinearExplainer works."""
    if isinstance(clf, MultinomialNB):
        clf.coef_ = (clf.feature_log_prob_[1] - clf.feature_log_prob_[0]).reshape(1, -1)
        clf.intercept_ = np.array([clf.class_log_prior_[1] - clf.class_log_prior_[0]])
    return clf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()

    tune_cfg = load_yaml_section("tune")
    shap_cfg = load_yaml_section("shap")

    model_name = args.model
    method_name = args.method
    n_samples = shap_cfg["n_shap_samples"]
    n_misclassified = shap_cfg["n_misclassified"]
    output_dir = os.path.join(shap_cfg["output_dir"], f"{model_name}__{method_name}")
    misclassified_dir = os.path.join(output_dir, "misclassified")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(misclassified_dir, exist_ok=True)

    # 1. Load model from MLflow
    logger.info(f"=== Loading model: {model_name} / {method_name} ===")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(tune_cfg["mlflow_experiment"])
    if experiment is None:
        raise ValueError(f"Experiment '{tune_cfg['mlflow_experiment']}' not found in MLflow")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f'tags.model = "{model_name}" AND '
            f'tags.method = "{method_name}" AND '
            f'tags.is_best = "true"'
        ),
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No best run found for {model_name} / {method_name}")
    run_id = runs[0].info.run_id
    logger.info(f"\trun_id: {run_id}")
    pipeline = mlflow.sklearn.load_model(f"runs:/{run_id}/best_estimator")

    # 2. Load and sample test data
    logger.info("=== Loading test data ===")
    test_df = prepare_features(
        pd.read_csv(tune_cfg["input_test"], low_memory=False),
        tune_cfg["text_column"],
    )
    # Original (pre-cleaning) test set — same row order, aligned index
    original_df = pd.read_csv("data/splits/test.csv", low_memory=False)
    label_col = tune_cfg["label_column"]
    X_test = test_df.drop(columns=[label_col])
    y_test = test_df[label_col]

    if n_samples is None:
        X_sample = X_test.copy()
        y_sample = y_test.copy()
    else:
        X_sample = X_test.sample(min(n_samples, len(X_test)), random_state=42)
        y_sample = y_test.loc[X_sample.index]
    logger.info(f"\tSHAP sample: {len(X_sample)} rows")

    # 3. Feature names after preprocessor + selector
    feature_names = _get_feature_names(pipeline)
    logger.info(f"\tFeatures after selection: {len(feature_names)}")

    # 4. Transform X through all steps except clf
    logger.info("=== Transforming features ===")
    transform_pipe = Pipeline(pipeline.steps[:-1])
    X_transformed = transform_pipe.transform(X_sample)

    # 5. SHAP LinearExplainer
    logger.info("=== Running SHAP ===")
    clf = _patch_nb_as_linear(pipeline.named_steps["clf"])
    explainer = shap.LinearExplainer(clf, X_transformed)
    shap_values = explainer(X_transformed)
    shap_values.feature_names = feature_names

    # 6a. Beeswarm summary plot (task a)
    logger.info("=== 6a. Saving beeswarm summary plot ===")
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title(f"SHAP summary — {model_name} / {method_name}", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "summary_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 6b. Positive / negative feature importance bar chart (task b)
    logger.info("=== 6b. Saving feature importance plot ===")
    sv = shap_values.values
    if sv.ndim == 3:
        sv = sv[:, :, 1]  # positive class for multiclass fallback
    mean_shap = sv.mean(axis=0)
    top_idx = np.argsort(np.abs(mean_shap))[-20:]  # top 20 by |mean SHAP|
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_shap[top_idx]
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in top_vals]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_idx)), top_vals, color=colors)
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Mean SHAP value  (positive → pushes toward positive class)")
    ax.set_title(f"Top 20 features by mean SHAP — {model_name} / {method_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # 6c. Misclassified examples — waterfall plots (task c)
    logger.info("=== 6c. Saving misclassified example plots ===")
    y_pred_sample = pipeline.predict(X_sample)
    misclassified_pos = np.where(y_pred_sample != y_sample.values)[0]
    logger.info(f"\tMisclassified in sample: {len(misclassified_pos)}")

    text_col = tune_cfg["text_column"]
    for i, pos in enumerate(misclassified_pos[:n_misclassified]):
        true_label = y_sample.values[pos]
        pred_label = y_pred_sample[pos]

        orig_row = original_df.loc[X_sample.index[pos]]
        orig_text = (str(orig_row.get("review_text", "")) + " " + str(orig_row.get("review_title", ""))).strip()
        clean_text = str(X_sample.iloc[pos][text_col]).strip()

        orig_wrapped  = textwrap.fill(orig_text,  width=110)
        clean_wrapped = textwrap.fill(clean_text, width=110)
        text_block = f"Original:     {orig_wrapped}\n\nPreprocessed: {clean_wrapped}"

        shap.plots.waterfall(shap_values[pos], max_display=15, show=False)
        fig = plt.gcf()
        fig.suptitle(
            f"True: {true_label}  |  Pred: {pred_label}",
            fontsize=11, fontweight="bold", y=1.01,
        )
        fig.text(
            0.5, -0.04, text_block,
            ha="center", va="top", fontsize=7.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff9c4", alpha=0.85),
            transform=fig.transFigure,
        )
        plt.tight_layout()
        out_path = os.path.join(misclassified_dir, f"example_{i:02d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    # 7. Log all artifacts to the MLflow run
    logger.info("=== 7. Logging artifacts to MLflow ===")
    client.log_artifact(run_id, os.path.join(output_dir, "summary_beeswarm.png"), artifact_path="shap")
    client.log_artifact(run_id, os.path.join(output_dir, "feature_importance.png"), artifact_path="shap")
    for i in range(min(n_misclassified, len(misclassified_pos))):
        client.log_artifact(
            run_id,
            os.path.join(misclassified_dir, f"example_{i:02d}.png"),
            artifact_path="shap/misclassified",
        )

    logger.info(f"=== Done → {output_dir} ===")


if __name__ == "__main__":
    main()
