set -e

VEC=tfidf
EXP=all_features
MODEL=linear_svc
TAG_KEY=scalers_comparison

run_exp() {
    local name=$1
    shift
    dvc exp remove "${name}" 2>/dev/null || true
    dvc exp run --temp --name "${name}" "$@"
}

run_exp "tfidf__scaler_minmax" \
    --set-param train.vectorizer=${VEC} \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.active_model=${MODEL} \
    --set-param train.num_scaler=minmax \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=minmax

run_exp "tfidf__scaler_standard" \
    --set-param train.vectorizer=${VEC} \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.active_model=${MODEL} \
    --set-param train.num_scaler=standard \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=standard

run_exp "tfidf__scaler_robust" \
    --set-param train.vectorizer=${VEC} \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.active_model=${MODEL} \
    --set-param train.num_scaler=robust \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=robust
