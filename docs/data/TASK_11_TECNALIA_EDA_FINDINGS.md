# Task 11 — TECNALIA EDA Findings

## Objective

Understand TECNALIA telemetry behavior and establish a defensible regression target, operating-condition filter, and initial feature set.

---

## 11.1 Basic Operating Behavior

### Analyzed

- GHI
- Front GPOA
- Pmpp
- Vmpp
- Impp
- Module temperature
- Ambient temperature
- Wind speed
- Day/night behavior
- Per-module distributions

### Observed

The five modules share the same environmental measurements and time coverage but have different electrical performance distributions.

Approximately 53% of observations are low-light observations when using the exploratory GHI >= 20 daytime definition.

Module temperature contained invalid extreme values in the raw telemetry. For analysis, known sentinel values and physically invalid values below -50°C were treated as missing. Raw source files were not modified.

### Inferred

Front GPOA is more directly useful than GHI for explaining module output because it represents irradiance incident on the module plane.

Low-light observations produce near-zero power and can dominate an unrestricted regression target.

### Modeling impact

- Front GPOA is a primary feature.
- Module temperature is retained as an operating-state feature.
- Low-light observations require filtering for the regression task.
- Raw telemetry remains unchanged.

---

## 11.2 Environmental → Electrical Relationships

### Analyzed

Pearson and Spearman correlations between environmental variables and Pmpp, plus Pmpp behavior across GPOA ranges.

### Observed

Front GPOA has the strongest relationship with Pmpp across all five modules.

Pmpp increases nonlinearly with GPOA.

### Inferred

Expected-performance regression must model the relationship between irradiance and power.

### Modeling impact

- Front GPOA is the primary predictor.
- Nonlinear regression models should be evaluated.
- GHI, module temperature, ambient temperature and wind remain candidate contextual features.
- Vmpp and Impp are excluded because Pmpp is exactly their product.

---

## 11.3 Module-Specific Behavior

### Analyzed

Module metadata, rated power, normalized Pmpp, matched-GPOA comparisons and temperature-conditioned behavior.

### Observed

The five modules have different rated powers and different normalized performance.

At Front GPOA 800–1000 W/m², normalized mean performance differs substantially between modules.

### Inferred

Module identity and/or module specifications explain systematic differences that cannot be removed by rated-power normalization alone.

### Modeling impact

Use one common module-aware regression model initially.

Include module identity and evaluate selected module specifications as candidate features.

Do not assume one identical performance curve for all modules.

---

## 11.4 Temporal Behavior

### Analyzed

Monthly Pmpp, normalized Pmpp, matched-GPOA normalized performance and monthly environmental conditions.

### Observed

Environmental conditions vary strongly by month.

Normalized performance also changes across months even when restricting GPOA to 800–1000 W/m².

The pattern is not a simple monotonic decline.

### Inferred

Temporal variation exists, but the dataset does not provide sufficient evidence to interpret it as degradation.

### Modeling impact

- Temporal features may be evaluated as contextual predictors.
- Time must not be used as a degradation label.
- Chronological/time-aware validation is required.
- RUL remains unsupported.

---

## 11.5 Regression Target

### Candidate targets considered

1. Raw Pmpp
2. Pmpp / rated power
3. Expected-performance formulation

### Decision

Use:

`normalized_pmpp = Pmpp / rated_power`

as the regression target.

The regression objective is to predict expected normalized performance under observed operating conditions.

### Reason

Raw Pmpp is strongly dependent on irradiance and module rating.

Normalized Pmpp improves cross-module comparability while preserving the actual observed performance relationship.

It is not treated as a health score.

---

## 11.5.2 Residual Formulation

After regression:

`performance_residual = actual_normalized_pmpp - expected_normalized_pmpp`

The residual represents performance deviation relative to model expectation.

It is a monitoring/anomaly indicator, not a degradation or failure ground-truth label.

Arbitrary residual thresholds will not be used as supervised fault labels.

---

## 11.5.3 Operating-Condition Filter

Candidate GPOA thresholds were evaluated:

- >= 20 W/m²
- >= 100 W/m²
- >= 200 W/m²
- >= 400 W/m²
- >= 600 W/m²

The selected initial threshold is:

`Front GPOA >= 200 W/m²`

This retains 11,977 observations per module, approximately 23.07% of the complete module telemetry.

### Interpretation

This is a regression-data inclusion criterion.

It does not mean:

- healthy operation,
- fault-free operation,
- a universal engineering threshold,
- or a degradation boundary.

A higher threshold may be evaluated later through model sensitivity analysis.

---

## 11.6 Feature-Engineering Decisions

### Initial features

- Front GPOA
- Module temperature
- Ambient temperature
- GHI
- Wind speed
- Module identity

### Candidate additions

- Selected module specifications
- Month
- Day-of-year
- Hour-of-day
- Other justified contextual features

### Excluded

- Vmpp
- Impp
- Pmpp
- normalized Pmpp as an input feature

The excluded electrical variables would create target leakage or trivial target reconstruction.

---

## Final TECNALIA Regression Specification

### Target

`normalized_pmpp = Pmpp / rated_power`

### Filter

`Front GPOA >= 200 W/m²`

### Initial features

`Front GPOA, module temperature, ambient temperature, GHI, wind speed, module identity`

### Objective

Predict expected normalized module performance.

### Downstream monitoring quantity

`actual_normalized_pmpp - expected_normalized_pmpp`

### Validation

Chronological/time-aware splitting rather than random row splitting.

### Unsupported tasks

- supervised failure classification
- supervised degradation classification
- RUL prediction

because TECNALIA does not provide defensible labels for these tasks.

---

## Overall Task 11 Conclusion

TECNALIA is suitable for a condition-aware expected-performance regression problem.

It is not suitable, based on the available evidence, for directly supervised failure/degradation/RUL prediction.

The regression model should learn normal expected performance from environmental operating conditions and module characteristics, with deviations from that expectation later available as a monitoring signal.

Raw source data remains unchanged.
