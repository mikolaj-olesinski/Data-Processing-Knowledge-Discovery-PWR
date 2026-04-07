TAG_KEY=models_comparison

for exp in text_only non_text all_features; do
    for model in dummy linear_svc random_forest; do
        dvc exp run --temp --name "${exp}__${model}" \
            --set-param train.experiment=$exp \
            --set-param train.active_model=$model \
            --set-param train.tag_key=${TAG_KEY} \
            --set-param train.tag_value=${exp}__${model}
    done
done
