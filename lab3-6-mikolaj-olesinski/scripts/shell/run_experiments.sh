set -e

VEC=tfidf
EXP=all_features
TAG_KEY=feature_engineering

run_exp() {
    local name=$1
    shift
    dvc exp remove "${name}" 2>/dev/null || true
    dvc exp run --temp --name "${name}" "$@"
}

# --- Baseline ---
run_exp "tfidf__none" \
    --set-param train.vectorizer=${VEC} \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=none

# --- SelectKBest(chi2) ---
for k in 5000 15000 30000; do
    run_exp "tfidf__selectk_${k}" \
        --set-param train.vectorizer=${VEC} \
        --set-param train.feature_selection=select_k_best \
        --set-param train.n_top_features=${k} \
        --set-param train.experiment=${EXP} \
        --set-param train.tag_key=${TAG_KEY} \
        --set-param train.tag_value=select_k_best
done

# --- TruncatedSVD (LSA) ---
for n in 100 200 300; do
    run_exp "tfidf__svd_${n}" \
        --set-param train.vectorizer=${VEC} \
        --set-param train.feature_selection=truncated_svd \
        --set-param train.n_components=${n} \
        --set-param train.experiment=${EXP} \
        --set-param train.tag_key=${TAG_KEY} \
        --set-param train.tag_value=truncated_svd
done

# --- Both: SelectKBest → TruncatedSVD ---
run_exp "tfidf__both_20k_100" \
    --set-param train.vectorizer=${VEC} \
    --set-param train.feature_selection=both \
    --set-param train.n_top_features=20000 \
    --set-param train.n_components=100 \
    --set-param train.experiment=${EXP} \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=both

run_exp "tfidf__both_30k_200" \
    --set-param train.vectorizer=${VEC} \
    --set-param train.feature_selection=both \
    --set-param train.n_top_features=30000 \
    --set-param train.n_components=200 \
    --set-param train.experiment=${EXP} \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=both
