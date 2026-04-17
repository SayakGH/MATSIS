import pandas as pd
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


def run_correlation(df: pd.DataFrame, method: str = "pearson") -> ToolResult:
    """
    Compute a correlation matrix for all numeric columns.
    Falls back to autocorrelation (lag 1–20) when there is only one numeric column.
    """
    try:
        numeric = df.select_dtypes(include="number")

        if numeric.shape[1] >= 2:
            corr = numeric.corr(method=method)

            # Flatten the matrix into chart-friendly records
            chart_data = []
            for col_a in corr.columns:
                for col_b in corr.columns:
                    chart_data.append(
                        {"col_a": col_a, "col_b": col_b, "correlation": float(corr.loc[col_a, col_b])}
                    )

            # Keep only the top off-diagonal pairs for metrics
            pairs = []
            for i, col_a in enumerate(corr.columns):
                for j, col_b in enumerate(corr.columns):
                    if j > i:
                        pairs.append((col_a, col_b, float(corr.loc[col_a, col_b])))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)

            metrics = {
                "method": method,
                "n_columns": int(numeric.shape[1]),
                "top_pair": f"{pairs[0][0]} × {pairs[0][1]}" if pairs else "n/a",
                "top_correlation": pairs[0][2] if pairs else None,
            }

        else:
            # Single column — compute autocorrelation at lags 1‥20
            col = numeric.columns[0] if len(numeric.columns) else "value"
            series = df[col] if col in df.columns else df["value"]
            max_lag = min(20, len(series) - 1)
            lags = list(range(1, max_lag + 1))
            autocorrs = [float(series.autocorr(lag)) for lag in lags]

            chart_data = [{"lag": lag, "autocorrelation": ac} for lag, ac in zip(lags, autocorrs)]
            peak_lag = lags[autocorrs.index(max(autocorrs, key=abs))]

            metrics = {
                "method": "autocorrelation",
                "n_lags": max_lag,
                "peak_lag": peak_lag,
                "peak_value": autocorrs[peak_lag - 1],
            }

        return ToolResult(
            task="correlation",
            raw_output=chart_data,
            chart_data=chart_data,
            metrics=metrics,
        )

    except Exception as e:
        logger.error(f"Correlation error: {e}", exc_info=True)
        return ToolResult(task="correlation", raw_output={}, chart_data=[], metrics={}, error=str(e))
