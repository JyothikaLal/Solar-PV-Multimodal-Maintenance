# Project Scope

## Project

**Solar PV Multimodal Predictive Maintenance & Inspection System**

## 1. Problem Statement

Photovoltaic systems can experience electrical, performance, and physical anomalies that require timely inspection and maintenance.

This project develops an end-to-end AI/ML decision-support system that independently analyzes:

1. PV telemetry/performance data using classical machine learning.
2. Thermal infrared PV imagery using deep learning/computer vision.

The outputs from these independent models are integrated through decision-level fusion to produce a maintenance-oriented health assessment and maintenance priority.

## 2. Primary Datasets

### TECNALIA PV Performance Dataset

Purpose:

Telemetry/performance-based PV failure, degradation, and performance analysis.

Official source:

https://zenodo.org/records/18306410

### RaptorMaps InfraredSolarModules

Purpose:

Thermal infrared PV anomaly classification.

Official source:

https://github.com/RaptorMaps/InfraredSolarModules

## 3. Dataset Relationship

The TECNALIA and RaptorMaps datasets are independent.

There is no known:

- Asset-level pairing
- Module-level pairing
- Timestamp-level pairing
- Physical-equipment correspondence

Therefore:

> TECNALIA and RaptorMaps will be trained and evaluated independently. The project makes no paired-data claim and performs no cross-dataset supervised training.

The independent model outputs may be combined only through a decision-level fusion layer.

## 4. Classical ML Scope

The TECNALIA branch will include:

- Data ingestion
- Data validation
- Exploratory data analysis
- Data cleaning
- Preprocessing
- Feature engineering
- Feature selection
- Leakage-safe splitting
- Classification
- Regression
- Baseline models
- Advanced models
- Hyperparameter tuning
- Evaluation
- SHAP explainability
- Model serialization
- Inference

The final classification target and regression target will be determined only after auditing the actual TECNALIA data and metadata.

## 5. RUL Constraint

Remaining Useful Life (RUL) prediction is not automatically part of the project.

RUL will only be implemented if the actual TECNALIA data provides sufficient temporal and failure information to construct a scientifically defensible RUL target.

If this condition is not satisfied, the project will use a defensible continuous performance/degradation regression problem instead.

## 6. Deep Learning Scope

The RaptorMaps branch will include:

- Image validation
- Label verification
- Leakage-safe dataset splitting
- Image preprocessing
- Data augmentation
- Custom CNN baseline
- Transfer learning
- Fine-tuning
- Model evaluation
- Class-imbalance analysis
- Grad-CAM
- Image embedding extraction
- Model serialization
- Inference

PyTorch is the preferred deep-learning framework.

## 7. Multimodal Fusion

Fusion will occur at the decision/output level.

Potential telemetry outputs:

- Failure/degradation probability
- Regression prediction
- Confidence
- Health indicators

Potential thermal outputs:

- Anomaly class
- Anomaly confidence
- Severity-related indicators

No feature-level fusion between unpaired TECNALIA observations and RaptorMaps images will be performed.

## 8. Explainability

### Telemetry

SHAP will be used to explain:

- Important features
- Positive and negative feature contributions
- Individual predictions
- Global model behaviour

### Thermal

Grad-CAM or an appropriate equivalent will be used to investigate which image regions influence model predictions.

## 9. MLOps

MLflow will be used for:

- Experiment tracking
- Hyperparameter tracking
- Metric tracking
- Artifact tracking
- Model versioning
- Model registry

Monitoring will address relevant:

- Feature drift
- Missingness
- Prediction drift
- Image quality
- Embedding drift
- Prediction confidence changes

Retraining will require validation against the current production model before promotion.

## 10. Deployment

FastAPI will provide model inference APIs.

Docker will be used for containerization.

PostgreSQL will be introduced only if persistent prediction or model history requires it.

## 11. Evaluation Requirements

### Classification

Evaluation may include:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- ROC-AUC where appropriate
- PR-AUC where appropriate
- Balanced accuracy where appropriate
- Confusion matrix
- Per-class metrics

### Regression

Evaluation may include:

- MAE
- MSE
- RMSE
- R²
- MAPE only where appropriate for the selected target

The final test set will be reserved for final model evaluation and will not be repeatedly used for model tuning.

## 12. Data Leakage

The project must explicitly prevent data leakage.

For TECNALIA, individual timestamped rows will not automatically be randomly split.

The splitting strategy will be selected after examining:

- Module structure
- Timestamp structure
- Failure episodes
- Temporal dependencies

Possible approaches include chronological, grouped temporal, module-aware, or forward validation.

## 13. Engineering Principles

- Raw datasets remain immutable.
- Actual files and official documentation are the source of truth.
- Dataset contents will never be fabricated.
- Labels will never be fabricated.
- Evaluation metrics will never be fabricated.
- RUL ground truth will never be fabricated.
- Baselines precede advanced models.
- Every engineered feature requires technical justification.
- Configuration should be separated from implementation.
- Experiments and model versions must be reproducible.
- Tests should accompany implemented components.
- Existing working code should not be rewritten unnecessarily.
- Project limitations must be documented honestly.
- Uncertain technical decisions must be investigated before implementation.

## 14. Project Boundaries

This project is an AI-assisted predictive maintenance and inspection decision-support system.

It does not claim:

- Real-time paired TECNALIA/RaptorMaps sensing
- Asset-level correspondence between the datasets
- Scientifically validated RUL without supporting data
- A universally calibrated PV health score
- Autonomous maintenance execution

## 15. Success Criteria

The completed project should demonstrate:

### Classical ML

- Defensible classification problem
- Defensible regression problem
- Leakage-safe evaluation
- Baseline and advanced model comparison
- Hyperparameter tuning
- Classification metrics
- Regression metrics
- SHAP explainability

### Deep Learning

- Image preprocessing
- Data augmentation
- Custom CNN baseline
- Transfer learning
- Fine-tuning
- Explicit PyTorch training loop
- Class-imbalance evaluation
- Grad-CAM
- Image embeddings

### Multimodal AI

- Independently trained telemetry model
- Independently trained thermal model
- Decision-level fusion
- Maintenance-oriented health assessment
- Sensitivity analysis of fusion assumptions

### MLOps / Engineering

- Experiment tracking
- Model versioning
- Monitoring
- Retraining workflow
- FastAPI inference
- Docker deployment
- Automated tests
- Reproducible project structure
- Technical documentation

## 16. Current Status

Task 1 — Requirements and project scope: **Complete**

Task 2 — Git repository and workflow: **In progress**