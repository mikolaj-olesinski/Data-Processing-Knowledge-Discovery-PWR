set -e

EXP=all_features
MODEL=linear_svc
TAG_KEY=vectorizer_comparison

run_exp() {
    local name=$1
    shift
    dvc exp remove "${name}" 2>/dev/null || true
    dvc exp run --temp --name "${name}" "$@"
}

# Bag-of-Words
run_exp "bow__baseline" \
    --set-param train.vectorizer=bow \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.active_model=${MODEL} \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=bow

# TF-IDF
run_exp "tfidf__baseline" \
    --set-param train.vectorizer=tfidf \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.active_model=${MODEL} \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=tfidf

# Word2Vec
run_exp "w2v__baseline" \
    --set-param train.vectorizer=w2v \
    --set-param train.feature_selection=none \
    --set-param train.experiment=${EXP} \
    --set-param train.active_model=${MODEL} \
    --set-param train.tag_key=${TAG_KEY} \
    --set-param train.tag_value=word2vec
