# MLflow Experiment Comparison

Generated from the local MLflow tracking backend.

## Comparison policy

- Runs are grouped by dataset, modality, and task.
- Regression and classification metrics are not ranked against each other.
- Only existing tracked runs are included.
- This report does not retrain or modify any model.
- TECNALIA and RaptorMaps remain independent datasets.

Total tracked runs: **3**
Comparison groups: **2**

## TECNALIA — telemetry — regression

| model_family | run_name | validation_mae | validation_rmse | validation_r2 | test_mae | test_rmse | test_r2 | test_mape_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gradient_boosting | tecnalia_gradient_boosting_final_retrospective | 0.029539 | 0.061526 | 0.910609 | 0.029967015745371 | 0.0659600795768725 | 0.915242823273993 | 13.261537784758684 |

## RaptorMaps — thermal — classification

| model_family | run_name | best_validation_macro_f1 | test_accuracy | test_balanced_accuracy | test_macro_f1 | test_weighted_f1 | test_roc_auc_ovr_macro | test_pr_auc_macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet18 | raptormaps_resnet18_finetuned_retrospective | 0.6507379837195436 | 0.784594864954985 | 0.6160580192563342 | 0.6385971332082477 | 0.7790920175787409 | 0.9424316268041923 | 0.6583795510819005 |
| efficientnet_b0 | raptormaps_efficientnet_b0_finetuned_retrospective | 0.5832767374402924 | 0.6988996332110704 | 0.6128985783269729 | 0.5596800360480753 | 0.7123531799012272 | 0.9322413140249924 | 0.5870737371430415 |
