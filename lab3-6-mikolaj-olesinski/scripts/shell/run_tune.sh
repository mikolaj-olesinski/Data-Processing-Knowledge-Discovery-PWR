set -e

TAG_KEY=hyperparameter_tuning
MODELS=(linear_svc logistic_regression multinomial_nb)
METHODS=(grid_search random_search halving_grid_search halving_random_search)

run_exp() {
    local name=$1
    shift
    dvc exp remove "${name}" 2>/dev/null || true
    dvc exp run --temp --name "${name}" "$@"
}

for model in "${MODELS[@]}"; do
    for method in "${METHODS[@]}"; do
        run_exp "tune__${model}__${method}" \
            --set-param tune.active_models=[${model}] \
            --set-param tune.search_methods=[${method}] \
            --set-param tune.tag_key=${TAG_KEY} \
            --set-param tune.tag_value=${model} \
            -- tune
    done
done
