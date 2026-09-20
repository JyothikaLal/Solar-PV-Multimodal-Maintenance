from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingRegressor

from src.data.tecnalia_regression import (
    MODULE_PATHS,
    RAW_ROOT,
    SPLIT_MANIFEST,
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    engineer_tecnalia_regression_modules,
    get_regression_feature_columns,
    load_tecnalia_regression_modules,
)
from src.models.regression.preprocessing import (
    create_regression_preprocessor,
)


RANDOM_STATE = 42

OUTPUT_ROOT = Path(
    "reports/results/tecnalia/shap"
)

LOCAL_OUTPUT_ROOT = (
    OUTPUT_ROOT / "local"
)

IMPORTANCE_PATH = (
    OUTPUT_ROOT / "global_feature_importance.csv"
)

ENGINEERING_IMPORTANCE_PATH = (
    OUTPUT_ROOT / "engineering_feature_importance.csv"
)

PLOT_PATH = (
    OUTPUT_ROOT / "shap_summary_bar.png"
)

BEESWARM_PATH = (
    OUTPUT_ROOT / "shap_summary_beeswarm.png"
)

LOCAL_EXPLANATION_PATH = (
    LOCAL_OUTPUT_ROOT
    / "local_shap_explanations.csv"
)


def prepare_tecnalia_data() -> pd.DataFrame:
    modules = load_tecnalia_regression_modules(
        RAW_ROOT,
        MODULE_PATHS,
    )

    modules = engineer_tecnalia_regression_modules(
        modules
    )

    frame = build_tecnalia_regression_frame(
        modules
    )

    split_manifest = pd.read_csv(
        SPLIT_MANIFEST
    )

    return attach_frozen_tecnalia_split(
        frame,
        split_manifest,
    )


def fit_final_gradient_boosting(
    frame: pd.DataFrame,
):
    numeric_features, categorical_features = (
        get_regression_feature_columns()
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    development = frame[
        frame["split"].isin(
            ["train", "validation"]
        )
    ].copy()

    preprocessor = create_regression_preprocessor(
        numeric_features,
        categorical_features,
    )

    X_development = development[
        feature_columns
    ]

    y_development = development[
        "normalized_pmpp"
    ]

    X_processed = (
        preprocessor.fit_transform(
            X_development
        )
    )

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_processed,
        y_development,
    )

    return (
        model,
        preprocessor,
        feature_columns,
    )


def aggregate_feature_names(
    transformed_names: list[str],
) -> list[str]:

    aggregated = []

    for name in transformed_names:

        if name.startswith(
            "categorical__module_name_"
        ):
            aggregated.append(
                "module_name"
            )

        elif name.startswith(
            "cat__module_name_"
        ):
            aggregated.append(
                "module_name"
            )

        elif name.startswith(
            "numeric__"
        ):
            aggregated.append(
                name.replace(
                    "numeric__",
                    "",
                    1,
                )
            )

        elif name.startswith(
            "num__"
        ):
            aggregated.append(
                name.replace(
                    "num__",
                    "",
                    1,
                )
            )

        else:
            aggregated.append(name)

    return aggregated


def calculate_global_importance(
    explanation,
    transformed_names: list[str],
) -> pd.DataFrame:

    shap_values = explanation.values

    transformed_importance = (
        np.mean(
            np.abs(shap_values),
            axis=0,
        )
    )

    transformed_df = pd.DataFrame(
        {
            "transformed_feature":
                transformed_names,
            "mean_abs_shap":
                transformed_importance,
        }
    )

    transformed_df["feature"] = (
        aggregate_feature_names(
            transformed_names
        )
    )

    aggregated_df = (
        transformed_df
        .groupby(
            "feature",
            as_index=False,
        )["mean_abs_shap"]
        .sum()
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    aggregated_df[
        "importance_rank"
    ] = np.arange(
        1,
        len(aggregated_df) + 1,
    )

    return aggregated_df[
        [
            "importance_rank",
            "feature",
            "mean_abs_shap",
        ]
    ]


def create_engineering_importance(
    explanation,
    transformed_names: list[str],
) -> pd.DataFrame:

    shap_values = explanation.values

    rows = []

    for index, name in enumerate(
        transformed_names
    ):

        if (
            name.startswith(
                "categorical__module_name_"
            )
            or name.startswith(
                "cat__module_name_"
            )
        ):
            feature = "module_name"
            feature_type = "module_identity"

        elif name.startswith(
            "numeric__"
        ):
            feature = name.replace(
                "numeric__",
                "",
                1,
            )
            feature_type = (
                "telemetry_environmental"
            )

        elif name.startswith(
            "num__"
        ):
            feature = name.replace(
                "num__",
                "",
                1,
            )
            feature_type = (
                "telemetry_environmental"
            )

        else:
            feature = name
            feature_type = "other"

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "mean_abs_shap_component": float(
                    np.mean(
                        np.abs(
                            shap_values[
                                :,
                                index,
                            ]
                        )
                    )
                ),
            }
        )

    component_df = pd.DataFrame(
        rows
    )

    grouped = (
        component_df
        .groupby(
            [
                "feature",
                "feature_type",
            ],
            as_index=False,
        )["mean_abs_shap_component"]
        .sum()
        .rename(
            columns={
                "mean_abs_shap_component":
                    "mean_abs_shap"
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    total = grouped[
        "mean_abs_shap"
    ].sum()

    grouped[
        "relative_importance_percent"
    ] = (
        grouped["mean_abs_shap"]
        / total
        * 100
    )

    grouped[
        "importance_rank"
    ] = np.arange(
        1,
        len(grouped) + 1,
    )

    return grouped[
        [
            "importance_rank",
            "feature",
            "feature_type",
            "mean_abs_shap",
            "relative_importance_percent",
        ]
    ]


def validate_reconstruction(
    explanation,
    predictions: np.ndarray,
    tolerance: float = 1e-6,
) -> dict:

    reconstructed = (
        explanation.base_values
        + explanation.values.sum(axis=1)
    )

    errors = np.abs(
        reconstructed - predictions
    )

    return {
        "max_error": float(
            errors.max()
        ),
        "mean_error": float(
            errors.mean()
        ),
        "passed": bool(
            errors.max() <= tolerance
        ),
    }


def create_local_explanations(
    test: pd.DataFrame,
    explanation,
    predictions: np.ndarray,
    transformed_names: list[str],
) -> pd.DataFrame:

    local = test[
        [
            "module_name",
            "Fecha",
            "normalized_pmpp",
        ]
    ].copy()

    local["prediction"] = predictions

    local["residual"] = (
        local["normalized_pmpp"]
        - local["prediction"]
    )

    local["absolute_error"] = np.abs(
        local["residual"]
    )

    # Observation closest to the median
    # absolute prediction error.
    typical_position = (
        local["absolute_error"]
        .sub(
            local["absolute_error"]
            .median()
        )
        .abs()
        .idxmin()
    )

    # Observation with the largest
    # absolute prediction error.
    largest_error_position = (
        local["absolute_error"]
        .idxmax()
    )

    selected_cases = [
        (
            "typical_prediction",
            typical_position,
        ),
        (
            "largest_error",
            largest_error_position,
        ),
    ]

    rows = []

    for case_name, row_position in selected_cases:

        shap_index = int(
            row_position
        )

        shap_values = (
            explanation.values[
                shap_index
            ]
        )

        ranking = np.argsort(
            np.abs(shap_values)
        )[::-1]

        for rank, feature_index in enumerate(
            ranking[:10],
            start=1,
        ):

            rows.append(
                {
                    "case": case_name,
                    "rank": rank,
                    "module_name":
                        local.loc[
                            row_position,
                            "module_name",
                        ],
                    "Fecha":
                        local.loc[
                            row_position,
                            "Fecha",
                        ],
                    "actual_normalized_pmpp":
                        local.loc[
                            row_position,
                            "normalized_pmpp",
                        ],
                    "predicted_normalized_pmpp":
                        local.loc[
                            row_position,
                            "prediction",
                        ],
                    "residual":
                        local.loc[
                            row_position,
                            "residual",
                        ],
                    "absolute_error":
                        local.loc[
                            row_position,
                            "absolute_error",
                        ],
                    "feature":
                        transformed_names[
                            feature_index
                        ],
                    "shap_value":
                        float(
                            shap_values[
                                feature_index
                            ]
                        ),
                }
            )

    return pd.DataFrame(rows)


def save_local_waterfall_plots(
    explanation,
    test: pd.DataFrame,
    predictions: np.ndarray,
    transformed_names: list[str],
    X_test_processed,
):
    local = test[
        [
            "module_name",
            "Fecha",
            "normalized_pmpp",
        ]
    ].copy()

    local["prediction"] = predictions

    local["residual"] = (
        local["normalized_pmpp"]
        - local["prediction"]
    )

    local["absolute_error"] = np.abs(
        local["residual"]
    )

    # Same case-selection logic as the CSV
    # so the plots and CSV refer to exactly
    # the same observations.
    typical_position = (
        local["absolute_error"]
        .sub(
            local["absolute_error"]
            .median()
        )
        .abs()
        .idxmin()
    )

    largest_error_position = (
        local["absolute_error"]
        .idxmax()
    )

    cases = [
        (
            "typical_prediction",
            typical_position,
        ),
        (
            "largest_error",
            largest_error_position,
        ),
    ]

    for case_name, row_position in cases:

        shap_index = int(
            row_position
        )

        case_explanation = shap.Explanation(
            values=explanation.values[
                shap_index
            ],
            base_values=explanation.base_values[
                shap_index
            ],
            data=X_test_processed[
                shap_index
            ],
            feature_names=transformed_names,
        )

        shap.plots.waterfall(
            case_explanation,
            max_display=10,
            show=False,
        )

        plt.title(
            (
                f"{case_name} | "
                f"{local.loc[row_position, 'module_name']} | "
                f"{local.loc[row_position, 'Fecha']}"
            )
        )

        plt.tight_layout()

        output_path = (
            LOCAL_OUTPUT_ROOT
            / f"{case_name}_waterfall.png"
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

        print(
            "Saved:",
            output_path,
        )


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOCAL_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Load and prepare TECNALIA data
    # -------------------------------------------------

    frame = prepare_tecnalia_data()

    # -------------------------------------------------
    # Train locked final Gradient Boosting model
    # -------------------------------------------------

    (
        model,
        preprocessor,
        feature_columns,
    ) = fit_final_gradient_boosting(
        frame
    )

    # -------------------------------------------------
    # Prepare untouched test set
    # -------------------------------------------------

    test = frame[
        frame["split"] == "test"
    ].copy()

    test = test.sort_values(
        ["Fecha", "module_name"]
    ).reset_index(
        drop=True
    )

    X_test = test[
        feature_columns
    ]

    X_test_processed = (
        preprocessor.transform(
            X_test
        )
    )

    transformed_names = list(
        preprocessor.get_feature_names_out()
    )

    # -------------------------------------------------
    # SHAP TreeExplainer
    # -------------------------------------------------

    explainer = shap.TreeExplainer(
        model
    )

    explanation = explainer(
        X_test_processed
    )

    predictions = model.predict(
        X_test_processed
    )

    # -------------------------------------------------
    # SHAP reconstruction validation
    # -------------------------------------------------

    reconstruction = validate_reconstruction(
        explanation,
        predictions,
    )

    # -------------------------------------------------
    # Global SHAP importance
    # -------------------------------------------------

    importance = calculate_global_importance(
        explanation,
        transformed_names,
    )

    importance.to_csv(
        IMPORTANCE_PATH,
        index=False,
    )

    engineering_importance = (
        create_engineering_importance(
            explanation,
            transformed_names,
        )
    )

    engineering_importance.to_csv(
        ENGINEERING_IMPORTANCE_PATH,
        index=False,
    )

    # -------------------------------------------------
    # Global SHAP bar plot
    # -------------------------------------------------

    shap.summary_plot(
        explanation.values,
        X_test_processed,
        feature_names=transformed_names,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------
    # Global SHAP beeswarm plot
    # -------------------------------------------------

    shap.summary_plot(
        explanation.values,
        X_test_processed,
        feature_names=transformed_names,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        BEESWARM_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------
    # Local SHAP explanations
    # -------------------------------------------------

    local_df = create_local_explanations(
        test,
        explanation,
        predictions,
        transformed_names,
    )

    local_df.to_csv(
        LOCAL_EXPLANATION_PATH,
        index=False,
    )

    # -------------------------------------------------
    # Local SHAP waterfall plots
    # -------------------------------------------------

    save_local_waterfall_plots(
        explanation,
        test,
        predictions,
        transformed_names,
        X_test_processed,
    )

    # -------------------------------------------------
    # Console output
    # -------------------------------------------------

    print(
        "Test rows:",
        len(test),
    )

    print(
        "SHAP value shape:",
        explanation.values.shape,
    )

    print(
        "Reconstruction:",
        reconstruction,
    )

    print(
        "\nTransformed SHAP importance:"
    )

    print(
        importance.to_string(
            index=False
        )
    )

    print(
        "\nEngineering-level SHAP importance:"
    )

    print(
        engineering_importance.to_string(
            index=False
        )
    )

    print(
        "\nLocal SHAP explanations:"
    )

    print(
        local_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved:",
        IMPORTANCE_PATH,
    )

    print(
        "Saved:",
        ENGINEERING_IMPORTANCE_PATH,
    )

    print(
        "Saved:",
        PLOT_PATH,
    )

    print(
        "Saved:",
        BEESWARM_PATH,
    )

    print(
        "Saved:",
        LOCAL_EXPLANATION_PATH,
    )


if __name__ == "__main__":
    main()