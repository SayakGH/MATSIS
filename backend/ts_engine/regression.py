import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


def run_regression(df: pd.DataFrame) -> ToolResult:
    """
    Fit an OLS trend line over the time index and return slope, intercept,
    R², and the fitted series for chart rendering.
    """
    try:
        df = df.copy().reset_index(drop=True)

        # Encode time as an integer index (days since first record)
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.isna().all():
            raise ValueError("No valid timestamps found")

        t0 = ts.min()
        X = ((ts - t0).dt.total_seconds() / 86400).values.reshape(-1, 1)  # days
        y = df["value"].to_numpy()

        valid = ~np.isnan(X.ravel()) & ~np.isnan(y)
        X_fit, y_fit = X[valid], y[valid]

        model = LinearRegression()
        model.fit(X_fit, y_fit)

        trend = model.predict(X)
        r2 = float(model.score(X_fit, y_fit))

        df["timestamp"] = ts.astype(str)
        df["trend"] = trend.tolist()

        chart_cols = ["timestamp", "value", "trend"]
        chart_data = df[chart_cols].to_dict("records")
        for row in chart_data:
            for k in ("value", "trend"):
                if row[k] is not None:
                    try:
                        row[k] = float(row[k])
                    except (TypeError, ValueError):
                        pass

        slope_per_day = float(model.coef_[0])
        direction = "upward" if slope_per_day > 0 else "downward"

        return ToolResult(
            task="regression",
            raw_output=chart_data,
            chart_data=chart_data,
            metrics={
                "slope_per_day": slope_per_day,
                "intercept": float(model.intercept_),
                "r_squared": r2,
                "direction": direction,
                "total_change": float(trend[-1] - trend[0]),
            },
        )

    except Exception as e:
        logger.error(f"Regression error: {e}", exc_info=True)
        return ToolResult(task="regression", raw_output={}, chart_data=[], metrics={}, error=str(e))
