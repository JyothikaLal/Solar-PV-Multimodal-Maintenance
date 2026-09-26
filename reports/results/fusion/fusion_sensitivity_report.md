# Task 34 — Fusion Evaluation and Sensitivity Analysis

## Purpose

Evaluate whether the evidence-based fusion assessment behaves consistently when telemetry and thermal evidence are varied independently and in combination.

This evaluation uses controlled synthetic evidence values only for testing the decision logic. These values are not paired TECNALIA/RaptorMaps observations and are not physical labels.

## Frozen Thresholds

- Telemetry elevated: `-0.02075990540437562`
- Telemetry strong: `-0.06827571496472343`
- Telemetry extreme: `-0.23838252371512167`
- Thermal anomaly evidence: `0.5`

## Controlled State Matrix

| Telemetry | Thermal | Combined evidence state | Maintenance priority |
|---|---|---|---|
| none | none | `no_elevated_evidence` | `routine` |
| none | present | `thermal_evidence` | `review` |
| elevated | none | `telemetry_evidence` | `monitor` |
| elevated | present | `multimodal_evidence` | `review` |
| strong | none | `telemetry_evidence` | `review` |
| strong | present | `multimodal_evidence` | `priority_review` |
| extreme | none | `telemetry_evidence` | `review` |
| extreme | present | `multimodal_evidence` | `priority_review` |

## Monotonicity Findings

Telemetry evidence was increased from none → elevated → strong → extreme.

- none → routine
- elevated → monitor
- strong → review
- extreme → review

The assessment never decreases as negative telemetry evidence increases.

Thermal evidence was changed from none → present:

- none → routine
- present → review

## Cross-Modal Dominance Findings

| Scenario | Result |
|---|---|
| Weak telemetry + thermal | `review` |
| Strong telemetry + no thermal | `review` |
| No telemetry + thermal | `review` |
| Strong telemetry + thermal | `priority_review` |

The results show that a single weak modality does not bypass the evidence rules, while strong evidence from both independent modalities can produce the highest defined maintenance-priority state.

## Important Interpretation Constraint

The evaluation does not establish a calibrated physical PV-health score, physical degradation percentage, failure probability, remaining useful life, or maintenance date.

The telemetry thresholds are validation-derived evidence bands, while the thermal threshold is a validation-supported binary anomaly-evidence operating point.

The two source datasets remain independent. No asset, image, module, or timestamp pairing is performed.

## Threshold Constants

- `TELEMETRY_ELEVATED_THRESHOLD = -0.02075990540437562`
- `TELEMETRY_STRONG_THRESHOLD = -0.06827571496472343`
- `TELEMETRY_EXTREME_THRESHOLD = -0.23838252371512167`
- `THERMAL_ANOMALY_THRESHOLD = 0.5`
