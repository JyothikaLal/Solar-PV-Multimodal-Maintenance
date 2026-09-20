from __future__ import annotations

from xgboost import XGBRegressor


def create_xgboost_regressor() -> XGBRegressor:
    """Create the advanced TECNALIA regression model."""

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
