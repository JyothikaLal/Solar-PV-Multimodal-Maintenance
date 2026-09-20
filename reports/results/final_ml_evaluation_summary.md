# Final ML Evaluation Summary

## 1. Evaluation Scope

This document records the final ML evaluation results for the Solar PV Multimodal Predictive Maintenance system.

The two primary datasets remain independent:

- TECNALIA PV Performance Dataset — telemetry regression
- RaptorMaps InfraredSolarModules — thermal anomaly classification

No asset-level, timestamp-level, or module-level pairing is performed between the datasets.

---

## 2. TECNALIA Regression

### Target

`normalized_pmpp = Pmpp / rated_power`

Operating subset:

`Front GPOA >= 200 W/m²`

### Final Model Selection

The final model was selected using validation MAE.

The test set was reserved exclusively for final evaluation.

Locked model:

- Gradient Boosting Regressor
- n_estimators: 400
- learning_rate: 0.03
- max_depth: 3
- min_samples_leaf: 5
- random_state: 42

### Final Test Performance

| Metric | Value |
|---|---:|
| MAE | 0.029967 |
| RMSE | 0.065960 |
| R² | 0.915243 |
| MAPE | 13.261538% |

### Alternative Tuned Benchmark

XGBoost was retained as a valid alternative benchmark.

| Metric | XGBoost |
|---|---:|
| MAE | 0.031234 |
| RMSE | 0.064184 |
| R² | 0.919746 |
| MAPE | 12.975683% |

XGBoost was not selected as the final model because model selection was predeclared to use validation MAE rather than test performance.

### Final Model Error Analysis

Test observations: 12,925

Overall:

- MAE: 0.029967
- RMSE: 0.065960
- Mean residual: -0.009154
- Median residual: -0.007510
- Residual standard deviation: 0.065322
- 95th percentile absolute residual: 0.134556
- Maximum absolute residual: 0.763911

### Module-Level Error

| Module | MAE | RMSE | Mean Residual |
|---|---:|---:|---:|
| Atersa | 0.033896 | 0.079032 | -0.007639 |
| JaSolar3 | 0.028319 | 0.064908 | -0.003289 |
| NingboSolar | 0.028539 | 0.062914 | -0.005587 |
| Photowatt | 0.028376 | 0.054633 | -0.014133 |
| TrinaSolar | 0.030706 | 0.065968 | -0.015124 |

### GPOA-Level Error

| GPOA Bin | N | MAE | RMSE |
|---|---:|---:|---:|
| 200-400 | 4370 | 0.022410 | 0.049236 |
| 400-600 | 1955 | 0.054242 | 0.098570 |
| 600-800 | 1600 | 0.037268 | 0.078486 |
| 800-1000 | 4420 | 0.021671 | 0.050641 |
| 1000-1200 | 570 | 0.047395 | 0.096604 |
| 1200+ | 10 | 0.091690 | 0.115718 |

The 1200+ group contains only 10 observations and should not be used for strong conclusions.

The 400-600 GPOA group shows substantially higher error than the 200-400 and 800-1000 groups.

### Regression Limitations

- TECNALIA does not provide scientifically defensible supervised failure/degradation labels.
- Regression performance should therefore not be interpreted as fault-classification performance.
- Large residuals indicate prediction errors, not confirmed physical failures.
- No RUL target is defined.
- Target-derived telemetry features were excluded from the monitoring-safe regression contract.
- Test data was not used for model selection.

---

## 3. RaptorMaps Thermal Classification

### Task

12-class infrared anomaly classification.

Primary evaluation metric:

**Macro F1**

Macro F1 is emphasized because of substantial class imbalance.

### Model Comparison

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Random Forest | 0.600200 | 0.439587 | 0.434217 | 0.600867 |
| Logistic Regression | 0.305435 | 0.348717 | 0.241965 | 0.341678 |
| Decision Tree | 0.462154 | 0.235167 | 0.235828 | 0.458608 |
| Dummy | 0.500167 | 0.083333 | 0.055568 | 0.333519 |
| Custom CNN | 0.543515 | 0.491673 | 0.436323 | 0.577189 |

### Custom CNN

Test Macro F1:

0.436323

Test balanced accuracy:

0.491673

Test ROC-AUC OVR macro:

0.899304

Test PR-AUC macro:

0.468486

### Minority Classes

The minority classes include:

- Diode-Multi
- Hot-Spot
- Hot-Spot-Multi
- Soiling

These classes have low test support and remain difficult to classify reliably.

### Error Patterns

The classical Random Forest baseline shows substantial confusion among:

- No-Anomaly and Cell
- No-Anomaly and Vegetation
- No-Anomaly and Shadowing
- No-Anomaly and Offline-Module
- Cell and Vegetation
- Cell-Multi and Cell

The CNN also shows difficulty on several low-support anomaly classes.

### Classification Limitations

- The RaptorMaps images are extremely low resolution.
- Classes are strongly imbalanced.
- Minority-class metrics are sensitive to small test supports.
- Classification performance should be interpreted per class rather than through accuracy alone.
- RaptorMaps and TECNALIA are independent datasets and are not paired.

---

## 4. Evaluation Artifacts

### TECNALIA

- `reports/results/tecnalia/final_models/final_model_metrics.csv`
- `reports/results/tecnalia/final_models/final_model_config.json`
- `reports/results/tecnalia/final_models/gradient_boosting_test_predictions.csv`
- `reports/results/tecnalia/final_models/error_analysis/overall_metrics.csv`
- `reports/results/tecnalia/final_models/error_analysis/module_error_metrics.csv`
- `reports/results/tecnalia/final_models/error_analysis/gpoa_error_metrics.csv`
- `reports/results/tecnalia/final_models/error_analysis/largest_prediction_errors.csv`
- `reports/results/tecnalia/final_models/error_analysis/residual_quantiles.csv`
- `reports/results/tecnalia/final_models/error_analysis/merged_test_predictions_with_conditions.csv`

### RaptorMaps

- `reports/results/raptormaps/classification_baselines/`
- `reports/results/raptormaps/custom_cnn/`
- `reports/results/raptormaps/classification_model_comparison.csv`

---

## 5. Overall Interpretation

The TECNALIA branch provides a strong regression benchmark for expected normalized PV performance under the defined operating conditions.

The RaptorMaps branch demonstrates multiclass thermal anomaly classification, but performance varies substantially by class because of image resolution and class imbalance.

The two branches remain scientifically independent and are intended to be combined only at decision level in later fusion tasks.

