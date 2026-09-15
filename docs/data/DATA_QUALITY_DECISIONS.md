# Data Quality Decisions

## Purpose

This document converts the verified TECNALIA and RaptorMaps audit/EDA findings into explicit data-quality, preprocessing, leakage-control, split, and dataset-independence rules

Raw datasets remain immutable. These rules apply to future intermediate/processed data pipelines.

---

## 1. TECNALIA Data-Quality Decisions

### 1.1 Missing values

The following columns were found to be 100% missing across the five module datasets:

- DHI
- DNI
- Vac
- Iac
- Pac
- Back GPOA
- SR
- Pressure
- Humidity
- Rain Accumulation
- Intensity
- Direction of wind

**Decision:** Do not impute these columns. Exclude them from modeling unless a valid data source or reconstruction method is established later.

`Isc`, `Voc`, and `FF` have substantial systematic missingness.

**Decision:** Preserve their missing values and do not blindly impute them. Their usefulness for future modeling must be established empirically.

Photowatt has a small number of missing values in `Vmpp`, `Impp`, and `Pmpp`.

**Decision:** Handle these observations explicitly in the processed dataset according to the future modeling pipeline; never alter the raw CSV.

### 1.2 Sentinel values

JaSolar3 contains 783 occurrences of `Temp. Mod (°C) = -9999`.

**Decision:** Convert `-9999` to `NaN` only in intermediate/processed representations. Never modify the raw source file.

### 1.3 Timestamp integrity

Verified:

- No duplicate timestamps.
- Timestamps are sorted.
- Coverage: 2024-11-01 to 2025-07-30.
- Sampling intervals are 5 or 10 minutes.
- No gaps larger than 10 minutes were observed.

**Decision:** Do not assume a fixed sampling frequency. Do not blindly resample raw telemetry. Any future resampling must be an explicit, documented transformation.

### 1.4 Low-light observations

A substantial portion of the data occurs under low irradiance.

**Decision:** Do not treat low-light observations as failures or degraded operation automatically.

For the initial regression task, use:

`Front GPOA >= 200 W/m²`

This is an operating-condition inclusion criterion, not a health or failure threshold.

### 1.5 Negative GHI

Small negative GHI values occur predominantly during night/low-light periods.

**Decision:** Do not automatically delete or overwrite these observations. Any correction/transformation must be explicitly justified and applied only to processed features.

### 1.6 Derived electrical variables and leakage

The audit established:

`Pmpp = Vmpp × Impp`

exactly for the available data.

**Decision:** Do not use `Pmpp` or `normalized_pmpp` as predictors when predicting normalized Pmpp. `Vmpp` and `Impp` also require careful treatment because they directly reconstruct the target through Pmpp.

---

## 2. TECNALIA Target and Feature Decisions

### 2.1 Regression target

Initial target:

`normalized_pmpp = Pmpp / rated_power`

Initial operating filter:

`Front GPOA >= 200 W/m²`

Objective: predict expected module performance under observed operating conditions.

Residual:

`actual_normalized_pmpp - expected_normalized_pmpp`

**Decision:** Use residuals as a performance-deviation monitoring indicator, not as ground-truth degradation labels.

### 2.2 Initial regression features

Initial candidate features:

- Front GPOA
- Module temperature
- Ambient temperature
- GHI
- Wind speed
- Module identity

Potential contextual features such as hour, month, or day-of-year require later validation before inclusion.

Exclude target-derived/reconstructive variables that create leakage.

### 2.3 Module-specific normalization

Verified rated powers:

| Module | Rated Power |
|---|---:|
| Atersa | 330 W |
| JaSolar3 | 315 W |
| NingboSolar | 175 W |
| Photowatt | 155 W |
| TrinaSolar | 185 W |

**Decision:** Account for module-specific rated power and module identity when comparing or modeling performance.

Lower observed normalized performance for a module must not automatically be interpreted as degradation.

---

## 3. RaptorMaps Data-Quality Decisions

### 3.1 Image integrity

Verified:

- 20,000 metadata records
- 20,000 referenced images
- 0 missing images
- 0 corrupt/unreadable images
- Native dimensions: 40 × 24 pixels
- Grayscale mode `L`

Initial tensor representation:

`[1, 40, 24]`

### 3.2 Pixel preprocessing

Observed pixel range: `0–255`.

Initial preprocessing:

- Convert to `float32`
- Divide pixel values by `255`
- Resulting range: `[0, 1]`

**Decision:** Do not interpret the 0–255 values as calibrated physical temperature.

### 3.3 Image normalization

**Decision:** Do not perform per-image normalization initially because absolute intensity may contain useful discriminative information.

Any dataset-level normalization required for a later transfer-learning architecture will be evaluated separately.

### 3.4 Image resolution

**Decision:** Preserve native resolution for the custom CNN baseline.

Do not blindly stretch 40 × 24 images to 224 × 224. Any resizing/padding required for transfer learning must be an explicit architecture-specific decision.

---

## 4. RaptorMaps Duplicate and Split Decisions

### 4.1 Duplicate audit

Exact image hashing found:

- 19,978 unique hashes
- 22 duplicate groups
- 44 images involved
- 6 conflicting duplicate groups
- 12 contradictory records

**Decision:** Exclude the 12 contradictory records from supervised classification splits. Raw images and metadata remain unchanged.

Consistent duplicate groups may remain, but all members must remain within the same split.

### 4.2 Frozen split

Authoritative manifest:

`data/processed/raptormaps/split_manifest.csv`

Current split:

- Train: 13,992
- Validation: 2,997
- Test: 2,999
- Excluded conflicting duplicates: 12

Verified cross-split duplicate groups: `0`.

**Decision:** Future loaders must use the split manifest rather than independently recreating the split from the raw image directory.

---

## 5. RaptorMaps Class-Imbalance Decisions

RaptorMaps contains 12 classes with substantial imbalance.

Largest class:

- No-Anomaly: 10,000 (50%)

Smallest class:

- Diode-Multi: 175 (0.875%)

Minority classes include:

- Hot-Spot
- Hot-Spot-Multi
- Soiling
- Diode-Multi

**Decision:** Primary classification evaluation must emphasize:

- Macro F1
- Per-class precision
- Per-class recall
- Per-class F1
- Balanced accuracy
- Confusion matrix

Accuracy and weighted F1 are secondary metrics.

Do not automatically apply class weighting, oversampling, or aggressive augmentation before baseline error analysis.

---

## 6. Dataset Independence

TECNALIA and RaptorMaps are independent datasets.

They must not be artificially paired by:

- asset
- module
- timestamp
- physical equipment
- image
- telemetry record

Each dataset is trained and evaluated independently.

Only model outputs may later participate in decision-level fusion.

---

## 7. Raw Data Preservation

Files under:

- `data/raw/tecnalia/`
- `data/raw/raptormaps/`

are treated as immutable source data.

Operations such as:

- sentinel conversion
- scaling
- filtering
- feature generation
- duplicate exclusion
- image normalization

must produce intermediate or processed representations and must never overwrite raw source files.

---

## 8. Assumptions and Unresolved Decisions

The following are not established facts and must remain explicitly marked as unresolved:

### TECNALIA

- Exact cause of systematic `Isc`, `Voc`, and `FF` missingness is unknown.
- The exact sensor cause of small negative GHI values is not conclusively established.
- The usefulness of temporal/context features requires validation.
- The initial normalized Pmpp representation may be revisited if later evidence supports a more appropriate physically motivated target.
- Residual behavior requires validation before it can support degradation monitoring claims.
- RUL is unsupported by the current dataset because defensible failure/maintenance/end-of-life labels are unavailable.

### RaptorMaps

- Pixel intensity is not assumed to be calibrated temperature.
- Intensity statistics alone do not establish class separability.
- Transfer-learning resizing/padding strategy remains to be evaluated.
- The best class-imbalance strategy must be selected from baseline error analysis and validation results.

---

## 9. Rule Hierarchy for Future Pipelines

When implementing future preprocessing:

1. Preserve raw data.
2. Apply only documented corrections.
3. Record exclusions and their reasons.
4. Prevent target leakage.
5. Respect the frozen RaptorMaps split manifest.
6. Preserve dataset independence.
7. Validate every transformation.
8. Revisit unresolved assumptions only with supporting evidence.

This document is the reference for Tasks involving actual preprocessing and dataset construction.
