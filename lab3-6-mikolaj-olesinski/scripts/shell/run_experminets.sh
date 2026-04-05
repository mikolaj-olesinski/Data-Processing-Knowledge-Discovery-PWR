for exp in text_only non_text all_features; do
    for model in dummy linear_svc random_forest; do
        dvc exp run --set-param train.experiment=$exp --set-param train.active_model=$model
    done
done