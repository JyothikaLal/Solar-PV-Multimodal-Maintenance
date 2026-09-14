# RaptorMaps Vision Task Definition

## 1. Vision Task

RaptorMaps is treated as a 12-class thermal/infrared image anomaly classification problem.

Each image receives one `anomaly_class` label from the dataset metadata.

## 2. Input

- Image type: grayscale infrared image
- Native width: 24 pixels
- Native height: 40 pixels
- Channels: 1
- Initial tensor shape: `[1, 40, 24]`
- Initial pixel scaling: divide by 255
- Resulting range: approximately `[0, 1]`

## 3. Target Classes

The fixed class mapping is:

| Index | Class |
|---:|---|
| 0 | Cell |
| 1 | Cell-Multi |
| 2 | Cracking |
| 3 | Diode |
| 4 | Diode-Multi |
| 5 | Hot-Spot |
| 6 | Hot-Spot-Multi |
| 7 | No-Anomaly |
| 8 | Offline-Module |
| 9 | Shadowing |
| 10 | Soiling |
| 11 | Vegetation |

This mapping must remain fixed across training, evaluation, inference, and deployment.

## 4. Model Output

The classifier will produce 12 logits.

These are converted to 12 class probabilities using softmax.

The predicted class is the class with the highest probability.

## 5. Baseline Training Objective

The baseline is a multiclass classification model using standard cross-entropy loss.

Class weighting, oversampling, and other imbalance-mitigation techniques are not frozen at this stage.

They will only be considered after baseline validation and error analysis.

## 6. Evaluation

### Primary Metric

- Macro F1

### Required Metrics

- Per-class Precision
- Per-class Recall
- Per-class F1
- Balanced Accuracy
- Confusion Matrix

### Additional Metrics

- Accuracy
- Weighted F1

Per-class support must also be reported, especially for minority classes.

## 7. Minority Classes

Special attention will be given to:

- Hot-Spot
- Hot-Spot-Multi
- Soiling
- Diode-Multi

These four classes together represent approximately 4.37% of the dataset.

## 8. Dataset Split

The validated `split_manifest.csv` is the authoritative supervised split.

- Train: 13,992
- Validation: 2,997
- Test: 2,999
- Excluded conflicting duplicates: 12

The split will not be recreated during model development.

## 9. Duplicate Handling

The 12 records belonging to conflicting duplicate groups are excluded from supervised classification.

Raw images are not modified or deleted.

Consistent duplicate groups are retained but remain within a single split.

Cross-split duplicate groups validated in Task 10: 0.

## 10. What the Model Predicts

The vision model predicts the anomaly class represented by the RaptorMaps dataset label.

It does not directly predict:

- physical temperature in degrees Celsius
- remaining useful life
- degradation percentage
- maintenance date
- real-world asset failure probability
- TECNALIA telemetry outcomes

## 11. Relationship to TECNALIA

The RaptorMaps vision model and TECNALIA telemetry model are trained and evaluated independently.

No asset-level, module-level, timestamp-level, or physical-equipment pairing is performed.

The two model outputs may later be combined only through decision-level fusion.

## 12. Current Vision Pipeline Specification

```text
RaptorMaps image
      |
      v
Grayscale 1 × 40 × 24
      |
      v
Float32 / 255
      |
      v
CNN classifier
      |
      v
12 logits
      |
      v
12 class probabilities
      |
      v
Predicted anomaly class