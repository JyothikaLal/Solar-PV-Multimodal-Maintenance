from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from src.inference.services import (
    build_health_assessment_for_predictions,
    infer_telemetry_prediction,
    infer_thermal_prediction,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


TELEMETRY_FEATURES = {
    "Front GPOA (W/m²)": 650.0,
    "GHI (W/m²)": 540.0,
    "Temp. Mod (°C)": 42.5,
    "Amb. Temp. (°C)": 28.0,
    "Wind Speed (m/s)": 3.2,
}

THERMAL_IMAGE_CANDIDATES = [
    PROJECT_ROOT / "reports/figures/raptormaps/grad_cam/custom_cnn/pilot/sample_0011_Hot-Spot_pred-Hot-Spot_correct.png",
    PROJECT_ROOT / "reports/figures/raptormaps/grad_cam/custom_cnn/pilot/sample_0010_Hot-Spot_pred-Hot-Spot_correct.png",
    PROJECT_ROOT / "reports/figures/raptormaps/grad_cam/custom_cnn/pilot/sample_0000_No-Anomaly_pred-No-Anomaly_correct.png",
]

SHAP_IMAGE = PROJECT_ROOT / "reports/results/tecnalia/shap/shap_summary_bar.png"
THERMAL_EXPLANATION_IMAGE = PROJECT_ROOT / "reports/figures/raptormaps/grad_cam/finetuned_minority_overlay_contact_sheet.png"


def pick_image() -> Path:
    for candidate in THERMAL_IMAGE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No representative thermal image was found in the repository.")


def build_dashboard_data() -> tuple[dict, dict, dict]:
    thermal_image = pick_image()
    telemetry_prediction = infer_telemetry_prediction(
        model_name="gradient_boosting_tuned",
        model_version="latest",
        module_name="Atersa",
        features=TELEMETRY_FEATURES,
    )
    thermal_prediction = infer_thermal_prediction(
        model_name="resnet18_finetuned",
        model_version="latest",
        image_path=str(thermal_image),
    )
    health = build_health_assessment_for_predictions(
        telemetry_prediction=telemetry_prediction,
        thermal_prediction=thermal_prediction,
        actual_normalized_pmpp=0.68,
    )
    return {
        "telemetry": telemetry_prediction,
        "thermal": thermal_prediction,
        "health": health,
        "image_path": thermal_image,
    }, TELEMETRY_FEATURES, {
        "thermal_explanation": THERMAL_EXPLANATION_IMAGE,
        "telemetry_explanation": SHAP_IMAGE,
    }


def main() -> None:
    st.set_page_config(
        page_title="Solar PV Maintenance Dashboard",
        page_icon="☀️",
        layout="wide",
    )

    data, features, explanation_paths = build_dashboard_data()
    telemetry_prediction = data["telemetry"]
    thermal_prediction = data["thermal"]
    health = data["health"]
    thermal_path = data["image_path"]

    st.title("Solar PV Multimodal Maintenance Dashboard")
    st.caption("Decision-support view for telemetry performance, thermal anomaly classification, and maintenance assessment.")

    st.sidebar.header("Model Controls")
    st.sidebar.text_input("Telemetry model", value=telemetry_prediction.provenance.model_name)
    st.sidebar.text_input("Telemetry version", value=telemetry_prediction.provenance.model_version)
    st.sidebar.text_input("Thermal model", value=thermal_prediction.provenance.model_name)
    st.sidebar.text_input("Thermal version", value=thermal_prediction.provenance.model_version)

    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Thermal prediction")
        thermal_image = Image.open(thermal_path)
        st.image(thermal_image, caption=f"Input thermal image: {thermal_path.name}", use_container_width=True)
        st.markdown(f"**Predicted anomaly:** {thermal_prediction.anomaly_class}")
        st.markdown(f"**Confidence:** {thermal_prediction.anomaly_confidence:.2%}")

        top_probs = sorted(thermal_prediction.class_probabilities.items(), key=lambda item: item[1], reverse=True)[:5]
        st.bar_chart({label: score for label, score in top_probs})

    with right:
        st.subheader("Telemetry prediction")
        st.metric("Predicted normalized Pmpp", f"{telemetry_prediction.predicted_normalized_pmpp:.3f}")
        telemetry_df = {
            "Feature": list(features.keys()),
            "Value": list(features.values()),
        }
        st.dataframe(telemetry_df, use_container_width=True, hide_index=True)

        st.subheader("Health assessment")
        st.metric("Maintenance priority", health["maintenance_priority"])
        st.metric("Combined evidence state", health["combined_evidence_state"])

        st.json(
            {
                "actual_normalized_pmpp": health["telemetry"]["actual_normalized_pmpp"],
                "predicted_normalized_pmpp": health["telemetry"]["predicted_normalized_pmpp"],
                "performance_deviation": health["telemetry"]["performance_deviation"],
                "telemetry_evidence_level": health["telemetry"]["evidence_level"],
                "thermal_anomaly_class": health["thermal"]["anomaly_class"],
                "thermal_evidence_level": health["thermal"]["evidence_level"],
            }
        )

    st.divider()

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Explanation: thermal model")
        st.image(explanation_paths["thermal_explanation"], use_container_width=True)
        st.markdown(
            "- The thermal branch identifies the active defect class from the heatmap signature.\n"
            "- The model concentrates on high-contrast hot regions and thermal gradients to justify the class selection.\n"
            "- The probability distribution indicates the certainty of the selected anomaly label."
        )

    with cols[1]:
        st.subheader("Explanation: telemetry model")
        st.image(explanation_paths["telemetry_explanation"], use_container_width=True)
        st.markdown(
            "- The telemetry explanation highlights the feature contribution to expected performance.\n"
            "- The model-weighted features point to module temperature and irradiance behavior rather than isolated by-chance fluctuations.\n"
            "- Any sustained negative deviation can support a maintenance review workflow."
        )

    st.divider()
    st.subheader("Model version and provenance")
    model_provenance = {
        "telemetry": telemetry_prediction.provenance.model_dump(),
        "thermal": thermal_prediction.provenance.model_dump(),
    }
    st.json(model_provenance)


if __name__ == "__main__":
    main()
