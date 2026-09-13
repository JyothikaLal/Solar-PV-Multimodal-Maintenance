# Target Definition

## Purpose

This document freezes the defensible machine-learning targets for the Solar PV Multimodal Predictive Maintenance & Inspection System based on the completed dataset audit.

The project uses two independent datasets:

- TECNALIA PV Performance Dataset — telemetry/electrical and environmental measurements.
- RaptorMaps InfraredSolarModules — thermal/infrared images with explicit anomaly labels.

The datasets must not be paired by asset, module, timestamp, physical equipment, image, or telemetry record. They will be trained and evaluated independently and combined only at the decision-level fusion stage.

---

## 1. TECNALIA Targets

### 1.1 Supervised Classification

Decision: No supervised failure/fault classification target.

The audited TECNALIA telemetry contains no explicit labels for failure, fault, alarm, maintenance, degradation, health, error, loss, or availability.

Therefore, artificial labels will not be created from low power, thresholds, elapsed time, or similar heuristics.

### 1.2 Regression

Decision: Expected-performance regression.

The TECNALIA branch will investigate regression for expected PV performance/power under observed operating conditions.

Candidate explanatory variables include:

- Front GPOA
- Module temperature
- Ambient temperature
- Wind speed
- Module identity and specifications
- Justified temporal/context features

The exact mathematical representation of the regression target will be finalized during preprocessing and regression-task design.

### 1.3 Performance-Deviation Indicator

After expected-performance regression:

Performance Deviation = Actual Pmpp - Expected Pmpp

This residual is a continuous performance-deviation indicator.

It is NOT a ground-truth degradation label.

### 1.4 RUL

Decision: Do not implement supervised RUL.

The audited TECNALIA data does not provide sufficient evidence for a defensible Remaining Useful Life target.

No failure timestamps, maintenance events, end-of-life information, or explicit degradation ground truth were identified.

RUL will therefore not be fabricated from elapsed time or observed power decline.

---

## 2. RaptorMaps Target

### 2.1 Supervised Thermal Classification

Decision: 12-class anomaly classification.

Input:

Thermal / infrared image

Target:

anomaly_class

The 12 classes are:

1. No-Anomaly
2. Cell
3. Vegetation
4. Diode
5. Cell-Multi
6. Shadowing
7. Cracking
8. Offline-Module
9. Hot-Spot
10. Hot-Spot-Multi
11. Soiling
12. Diode-Multi

The dataset is substantially imbalanced. Accuracy alone will not be sufficient for model assessment.

Particular attention will be given to:

- Hot-Spot
- Hot-Spot-Multi
- Soiling
- Diode-Multi

### 2.2 Duplicate and Label-Conflict Handling

The audit identified 22 exact duplicate groups.

Six groups contained identical image content with conflicting labels, involving 12 records.

These 12 records are excluded from the supervised classification split but remain untouched in the raw dataset.

The remaining consistent duplicate groups are kept within a single split to prevent exact-duplicate leakage.

---

## 3. Multimodal Fusion Target

TECNALIA and RaptorMaps will not share a supervised target or be trained as paired records.

Fusion will occur at the decision level using independently generated model outputs.

Potential fusion inputs include:

- TECNALIA expected-performance prediction
- TECNALIA performance-deviation indicator
- RaptorMaps anomaly class
- RaptorMaps classification confidence/probabilities

No health-score formula is frozen at this stage.

Any combined score or meta-model must be justified and evaluated after the independent modality models are established.

---

## 4. Final Target Definition

| Branch | Task | Target | Decision |
|---|---|---|---|
| TECNALIA | Supervised classification | None | Not supported by available labels |
| TECNALIA | Regression | Expected PV performance/power | Defensible |
| TECNALIA | Monitoring indicator | Actual - Expected | Derived continuous indicator |
| TECNALIA | RUL | Remaining Useful Life | Not defensible |
| RaptorMaps | Classification | 12-class anomaly_class | Defensible |
| Fusion | Multimodal decision | Independent model outputs | Deferred until modality models exist |

---

## 5. Modeling Constraints

1. Do not fabricate TECNALIA failure, degradation, or RUL labels.
2. Avoid trivial Pmpp reconstruction from Vmpp and Impp when defining the regression feature set because Pmpp = Vmpp x Impp.
3. Preserve module identity/specification information where required.
4. Account for temporal structure in TECNALIA validation.
5. Do not interpret seasonal/environmental variation as degradation without defensible evidence.
6. Evaluate RaptorMaps using imbalance-aware metrics and per-class performance.
7. Maintain dataset independence until decision-level fusion.
8. Do not freeze an unsupported health-score formulation.

---

## Status

Task 9 - Define Defensible Targets: COMPLETE
