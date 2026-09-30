# API Documentation

## Overview

This API exposes the implemented inference surface for the Solar PV multimodal maintenance system.

It covers three primary runtime capabilities:

1. TECNALIA telemetry prediction
2. RaptorMaps thermal anomaly classification
3. Decision-level fusion and health assessment

The API follows the repository's actual architecture: the TECNALIA and RaptorMaps branches remain independent and are only merged via decision-level fusion outputs.

---

## Supported endpoints

### 1. Health check

Endpoint:

- GET /api/v1/health

Purpose:

- verify the service is reachable

Response:

```json
{
  "status": "ok",
  "service": "solar-pv-multimodal-maintenance",
  "version": "0.1.0"
}
```

---

### 2. TECNALIA telemetry prediction

Endpoint:

- POST /api/v1/tecnalia/predict

Purpose:

- run a telemetry-based PV performance prediction using the TECNALIA regression inference contract.

Request body:

```json
{
  "model_name": "gradient_boosting_tuned",
  "model_version": "latest",
  "module_name": "Atersa",
  "features": {
    "Front GPOA (W/m²)": 650.0,
    "GHI (W/m²)": 540.0,
    "Temp. Mod (°C)": 42.5,
    "Amb. Temp. (°C)": 28.0,
    "Wind Speed (m/s)": 3.2
  }
}
```

Validation rules:

- all five required telemetry features must be present
- feature values must be numeric and finite
- the model name and version are required

Typical response:

```json
{
  "predicted_normalized_pmpp": 0.7624,
  "target_name": "normalized_pmpp",
  "provenance": {
    "model_name": "gradient_boosting_tuned",
    "model_version": "latest",
    "checkpoint_path": "local_fallback",
    "checkpoint_epoch": null,
    "validation_metric_name": null,
    "validation_metric_value": null
  }
}
```

Outcome:

- returns a continuous normalized Pmpp prediction suitable for downstream performance evidence evaluation.

---

### 3. RaptorMaps thermal prediction

Endpoint:

- POST /api/v1/raptormaps/predict

Purpose:

- classify a thermal PV image into one of the project's 12 anomaly classes.

Accepted inputs:

- JSON body with `image_base64`
- JSON body with `image_path`
- multipart upload with `file`

Example JSON request:

```json
{
  "model_name": "resnet18_finetuned",
  "model_version": "latest",
  "image_base64": "<base64-encoded-24x40-grayscale-image>"
}
```

Validation rules:

- either `image_base64` or `image_path` must be present
- file or image data must decode successfully
- the image must be valid and readable

Typical response:

```json
{
  "anomaly_class": "Hot-Spot",
  "anomaly_confidence": 0.63,
  "class_probabilities": {
    "Cell": 0.02,
    "Cell-Multi": 0.01,
    "Cracking": 0.02,
    "Diode": 0.01,
    "Diode-Multi": 0.01,
    "Hot-Spot": 0.63,
    "Hot-Spot-Multi": 0.06,
    "No-Anomaly": 0.12,
    "Offline-Module": 0.02,
    "Shadowing": 0.04,
    "Soiling": 0.03,
    "Vegetation": 0.03
  },
  "provenance": {
    "model_name": "resnet18_finetuned",
    "model_version": "latest",
    "checkpoint_path": "local_fallback",
    "checkpoint_epoch": null,
    "validation_metric_name": null,
    "validation_metric_value": null
  }
}
```

Outcome:

- returns the anomaly class, confidence, and complete 12-class probability vector.

---

### 4. Decision-level fusion

Endpoint:

- POST /api/v1/fusion

Purpose:

- combine the independent telemetry and thermal outputs at the decision layer without shared feature-level training.

Request body:

```json
{
  "telemetry": {
    "predicted_normalized_pmpp": 0.76,
    "target_name": "normalized_pmpp",
    "provenance": {
      "model_name": "gradient_boosting_tuned",
      "model_version": "latest",
      "checkpoint_path": "local_fallback",
      "checkpoint_epoch": null,
      "validation_metric_name": null,
      "validation_metric_value": null
    }
  },
  "thermal": {
    "anomaly_class": "Hot-Spot",
    "anomaly_confidence": 0.63,
    "class_probabilities": {
      "Cell": 0.02,
      "Cell-Multi": 0.01,
      "Cracking": 0.02,
      "Diode": 0.01,
      "Diode-Multi": 0.01,
      "Hot-Spot": 0.63,
      "Hot-Spot-Multi": 0.06,
      "No-Anomaly": 0.12,
      "Offline-Module": 0.02,
      "Shadowing": 0.04,
      "Soiling": 0.03,
      "Vegetation": 0.03
    },
    "provenance": {
      "model_name": "resnet18_finetuned",
      "model_version": "latest",
      "checkpoint_path": "local_fallback",
      "checkpoint_epoch": null,
      "validation_metric_name": null,
      "validation_metric_value": null
    }
  },
  "embedding_indicators": {
    "thermal_embedding_score": 0.72
  }
}
```

Outcome:

- returns the fused evidence output with `thermal_anomaly_detected` and `evidence_state`.

---

### 5. Combined health assessment

Endpoint:

- POST /api/v1/fusion/health

Purpose:

- translate independent telemetry and thermal outputs into a maintenance-oriented assessment.

Request body:

```json
{
  "telemetry_prediction": {
    "predicted_normalized_pmpp": 0.76,
    "target_name": "normalized_pmpp",
    "provenance": {
      "model_name": "gradient_boosting_tuned",
      "model_version": "latest",
      "checkpoint_path": "local_fallback",
      "checkpoint_epoch": null,
      "validation_metric_name": null,
      "validation_metric_value": null
    }
  },
  "thermal_prediction": {
    "anomaly_class": "Hot-Spot",
    "anomaly_confidence": 0.63,
    "class_probabilities": {
      "Cell": 0.02,
      "Cell-Multi": 0.01,
      "Cracking": 0.02,
      "Diode": 0.01,
      "Diode-Multi": 0.01,
      "Hot-Spot": 0.63,
      "Hot-Spot-Multi": 0.06,
      "No-Anomaly": 0.12,
      "Offline-Module": 0.02,
      "Shadowing": 0.04,
      "Soiling": 0.03,
      "Vegetation": 0.03
    },
    "provenance": {
      "model_name": "resnet18_finetuned",
      "model_version": "latest",
      "checkpoint_path": "local_fallback",
      "checkpoint_epoch": null,
      "validation_metric_name": null,
      "validation_metric_value": null
    }
  },
  "actual_normalized_pmpp": 0.68,
  "embedding_indicators": {
    "thermal_embedding_score": 0.72
  }
}
```

Typical response:

```json
{
  "telemetry": {
    "actual_normalized_pmpp": 0.68,
    "predicted_normalized_pmpp": 0.76,
    "performance_deviation": -0.08,
    "evidence_level": "strong"
  },
  "thermal": {
    "anomaly_class": "Hot-Spot",
    "no_anomaly_probability": 0.12,
    "anomaly_evidence": 0.88,
    "evidence_level": "present"
  },
  "combined_evidence_state": "multimodal_evidence",
  "maintenance_priority": "priority_review"
}
```

Outcome:

- returns the evidence state and maintenance priority derived from independent telemetry and thermal evidence.

---

## Validation and error handling

The API validates and returns consistent errors for invalid inputs.

### Error response format

```json
{
  "detail": "Request validation failed.",
  "error_code": "validation_error",
  "errors": [
    {
      "type": "missing",
      "loc": ["body", "features"],
      "msg": "Field required"
    }
  ]
}
```

Common failure cases:

- missing required telemetry feature fields
- non-numeric feature values
- invalid or missing image payloads
- malformed fusion input contracts
- impossible evidence-state combinations

---

## Execution model

This API is currently designed as an inference service layer atop the repository's backend logic. It intentionally follows the project architecture rather than creating a new unsupported data pipeline.

The execution flow is:

1. request validation
2. feature or image decoding
3. branch-specific inference
4. branch output contract validation
5. decision-level fusion
6. evidence-based health assessment
7. consistent response serialization

---

## Notes on model loading

The current implementation includes a project-aware inference layer that follows the repository's existing model contracts and local artifact layout. It uses the known TECNALIA reference model family and the RaptorMaps model families defined in the project.

The API is structured so that future production deployment can swap in versioned model artifacts and registry-backed model resolution without changing the request/response contracts.
