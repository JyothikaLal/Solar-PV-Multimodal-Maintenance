# Solar PV Multimodal Predictive Maintenance & Inspection System

An end-to-end AI/ML system for photovoltaic predictive maintenance that independently analyzes PV telemetry/performance data and thermal infrared imagery.

## Project Overview

The system contains two independent AI pipelines:

### 1. TECNALIA PV Performance Dataset

- Classical machine learning
- Failure/degradation classification
- Continuous performance/degradation regression
- SHAP explainability

### 2. RaptorMaps InfraredSolarModules

- Deep learning / computer vision
- Thermal anomaly classification
- Transfer learning and fine-tuning
- Grad-CAM explainability
- Image embeddings

The outputs of the two independent pipelines are combined through **decision-level fusion**.

## Important Dataset Constraint

The TECNALIA and RaptorMaps datasets are independent.

They are **not paired** by:

- Asset ID
- Module ID
- Timestamp
- Physical equipment

The project makes **no paired-data claim** and does not perform cross-dataset supervised training.

The two datasets are trained and evaluated independently. Their outputs are combined only at the decision/fusion layer.

## Planned System

```text
                    SOLAR PV AI SYSTEM
                           |
              +------------+------------+
              |                         |
              v                         v
       TECNALIA TELEMETRY        RAPTOR MAPS IR
              |                         |
              v                         v
       Classical ML Pipeline       Deep Learning
              |                         |
       +------+------+           +------+------+
       |             |           |             |
       v             v           v             v
 Classification  Regression  Anomaly       Embedding
                             Classification
       |             |           |
       +------+------+           |
              |                   |
              +--------+----------+
                       |
                       v
                DECISION FUSION
                       |
                       v
                PV HEALTH ASSESSMENT
                       |
                       v
              MAINTENANCE PRIORITY
                       |
                       v
               API / DEPLOYMENT
                       |
                       v
                  MLOps / Monitoring