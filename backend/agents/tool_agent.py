import pandas as pd
import numpy as np
from ts_engine.forecast import prophet_forecast, arima_forecast, lstm_forecast
from ts_engine.anomaly import run_isolation_forest, run_zscore
from ts_engine.decompose import run_stl_decomposition
from ts_engine.correlation import run_correlation
from ts_engine.rolling_stats import run_rolling_stats
from ts_engine.peak_detection import run_peak_detection
from ts_engine.regression import run_regression
from ts_engine.clustering import run_clustering
from utils.statistical_tools import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    is_stationary,
    correlation_test as stats_correlation,
    one_sample_ttest,
    two_sample_ttest,
    entropy,
    runs_test,
)
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


class ToolAgent:
    def run(self, task: str, params: dict, df: pd.DataFrame) -> ToolResult:
        dispatch = {
            "forecast": self._forecast,
            "anomaly_detection": self._anomaly,
            "decomposition": self._decompose,
            "summary": self._summary,
            "correlation": self._correlation,
            "rolling_stats": self._rolling_stats,
            "peak_detection": self._peak_detection,
            "regression": self._regression,
            "clustering": self._clustering,
            "outlier_detection": self._outlier_detection,
            "stationarity_test": self._stationarity_test,
            "statistical_correlation": self._statistical_correlation,
            "one_sample_ttest": self._one_sample_ttest,
            "two_sample_ttest": self._two_sample_ttest,
            "runs_test": self._runs_test,
            "entropy": self._entropy,
        }
        handler = dispatch.get(task, self._summary)
        try:
            return handler(params, df)
        except Exception as e:
            logger.error(f"ToolAgent error in task '{task}': {e}", exc_info=True)
            return ToolResult(task=task, raw_output={}, chart_data=[], metrics={}, error=str(e))

    def _forecast(self, params, df):
        method = params.get("method", "prophet")
        horizon = int(params.get("horizon", 10))
        if method == "arima":  return arima_forecast(df, horizon)
        if method == "lstm":   return lstm_forecast(df, horizon)
        return prophet_forecast(df, horizon)

    def _anomaly(self, params, df):
        method = params.get("method", "isolation_forest")
        if method == "zscore": return run_zscore(df)
        return run_isolation_forest(df)

    def _decompose(self, params, df):
        return run_stl_decomposition(df, int(params.get("period", 7)))

    def _summary(self, params, df):
        summary = df['value'].describe().to_dict()
        chart = df[['timestamp', 'value']].copy()
        # Safe timestamp serialisation — works for both datetime and string dtypes
        chart['timestamp'] = pd.to_datetime(chart['timestamp'], errors='coerce').astype(str)
        return ToolResult(
            task="summary",
            raw_output={k: float(v) for k, v in summary.items()},
            chart_data=chart.to_dict('records'),
            metrics={
                "row_count": int(len(df)),
                "mean": float(df['value'].mean()),
                "std": float(df['value'].std()),
                "min": float(df['value'].min()),
                "max": float(df['value'].max()),
            }
        )

    def _correlation(self, params, df):
        method = params.get("method", "pearson")
        return run_correlation(df, method)

    def _rolling_stats(self, params, df):
        window = int(params.get("window", 7))
        return run_rolling_stats(df, window)

    def _peak_detection(self, params, df):
        order = int(params.get("order", 3))
        return run_peak_detection(df, order)

    def _regression(self, params, df):
        return run_regression(df)

    def _clustering(self, params, df):
        n_clusters = int(params.get("n_clusters", 3))
        return run_clustering(df, n_clusters)

    def _outlier_detection(self, params, df):
        """Detect outliers using IQR or Z-score method."""
        column = params.get("column", "value")
        method = params.get("method", "iqr")
        series = df[column].dropna()

        if method == "zscore":
            mask = detect_outliers_zscore(series)
        else:
            mask = detect_outliers_iqr(series)

        outlier_indices = mask[mask].index.tolist()
        return ToolResult(
            task="outlier_detection",
            raw_output={"method": method, "outlier_count": len(outlier_indices)},
            chart_data=[],
            metrics={
                "outlier_count": len(outlier_indices),
                "outlier_percentage": round(len(outlier_indices) / len(series) * 100, 2),
            },
        )

    def _stationarity_test(self, params, df):
        """Test for stationarity using ADF test."""
        column = params.get("column", "value")
        series = df[column]
        is_stat, results = is_stationary(series)

        return ToolResult(
            task="stationarity_test",
            raw_output=results,
            chart_data=[],
            metrics={
                "is_stationary": int(is_stat),
                "adf_statistic": results["adf_statistic"],
                "p_value": results["p_value"],
            },
        )

    def _statistical_correlation(self, params, df):
        """Test correlation between two columns."""
        col1 = params.get("col1")
        col2 = params.get("col2")
        method = params.get("method", "pearson")

        if not col1 or not col2:
            return ToolResult(
                task="correlation",
                raw_output={},
                chart_data=[],
                metrics={"error": "col1 and col2 must be specified"},
                error="Missing columns"
            )

        result = stats_correlation(df, col1, col2, method)
        scatter_data = df[[col1, col2]].dropna().head(500).to_dict('records')

        return ToolResult(
            task="correlation",
            raw_output=result,
            chart_data=scatter_data,
            metrics={
                "correlation": round(result["correlation"], 4),
                "p_value": round(result["p_value"], 6),
                "method": result["method"],
            },
        )

    def _one_sample_ttest(self, params, df):
        """One-sample t-test."""
        column = params.get("column", "value")
        pop_mean = float(params.get("pop_mean", 0))
        series = df[column]

        result = one_sample_ttest(series, pop_mean)

        return ToolResult(
            task="one_sample_ttest",
            raw_output=result,
            chart_data=[],
            metrics={
                "t_statistic": round(result["t_statistic"], 4),
                "p_value": round(result["p_value"], 6),
                "sample_mean": round(series.mean(), 4),
                "pop_mean": pop_mean,
            },
        )

    def _two_sample_ttest(self, params, df):
        """Two-sample t-test for independent samples."""
        col1 = params.get("col1")
        col2 = params.get("col2")
        equal_var = params.get("equal_var", False)

        if not col1 or not col2:
            return ToolResult(
                task="two_sample_ttest",
                raw_output={},
                chart_data=[],
                metrics={"error": "col1 and col2 must be specified"},
                error="Missing columns"
            )

        series1 = df[col1].dropna()
        series2 = df[col2].dropna()

        result = two_sample_ttest(series1, series2, equal_var)

        return ToolResult(
            task="two_sample_ttest",
            raw_output=result,
            chart_data=[],
            metrics={
                "t_statistic": round(result["t_statistic"], 4),
                "p_value": round(result["p_value"], 6),
                "sample1_mean": round(result["sample1_mean"], 4),
                "sample2_mean": round(result["sample2_mean"], 4),
            },
        )

    def _runs_test(self, params, df):
        """Wald-Wolfowitz runs test for randomness."""
        column = params.get("column", "value")
        series = df[column]

        result = runs_test(series)

        return ToolResult(
            task="runs_test",
            raw_output=result,
            chart_data=[],
            metrics={
                "runs": result["runs"],
                "expected_runs": round(result["expected_runs"], 2),
                "z_statistic": round(result["z_statistic"], 4),
                "p_value": round(result["p_value"], 6),
                "is_random": result["is_random"],
            },
        )

    def _entropy(self, params, df):
        """Calculate Shannon entropy."""
        column = params.get("column", "value")
        bins = int(params.get("bins", 10))
        series = df[column]

        entropy_val = entropy(series, bins)

        # Histogram data for chart
        hist, bin_edges = np.histogram(series.dropna(), bins=bins)
        hist_data = [
            {"bin_start": float(bin_edges[i]),
             "bin_end": float(bin_edges[i+1]),
             "count": int(hist[i])}
            for i in range(len(hist))
        ]

        return ToolResult(
            task="entropy",
            raw_output={"entropy": entropy_val, "bins": bins},
            chart_data=hist_data,
            metrics={
                "entropy": round(entropy_val, 4),
                "max_entropy": round(np.log2(bins), 4) if bins > 1 else 0,
            },
        )


tool_agent = ToolAgent()
