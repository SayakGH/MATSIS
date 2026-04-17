"""Statistical analysis agent for hypothesis testing and diagnostics."""
import json
from scipy import stats
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from llm.ollama_client import ollama_client
from config import settings
from utils.statistical_tools import (
    is_stationary,
    detect_outliers_iqr,
    detect_outliers_zscore,
    confidence_interval,
    correlation_test,
    one_sample_ttest,
    two_sample_ttest,
    anova_test,
    runs_test,
    entropy,
    mutual_information,
)
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


class StatisticalAgent:
    def __init__(self):
        self.model = settings.ANALYST_MODEL

    async def test_stationarity(self, df: pd.DataFrame, column: str) -> ToolResult:
        """Test for stationarity using ADF test."""
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

    async def detect_outliers(self, df: pd.DataFrame, column: str,
                              method: str = "iqr") -> ToolResult:
        """Detect outliers in a series."""
        series = df[column].dropna()

        if method == "zscore":
            mask = detect_outliers_zscore(series)
        else:  # iqr
            mask = detect_outliers_iqr(series)

        outlier_indices = mask[mask].index.tolist()
        outlier_values = df.loc[outlier_indices, column].tolist()

        return ToolResult(
            task="outlier_detection",
            raw_output={
                "method": method,
                "total_outliers": len(outlier_indices),
                "outlier_percent": round(len(outlier_indices) / len(series) * 100, 2),
            },
            chart_data=[],
            metrics={
                "outlier_count": len(outlier_indices),
                "outlier_indices": outlier_indices[:100],  # Limit for serialization
                "outlier_values": outlier_values[:100],
            },
        )

    async def test_correlation(self, df: pd.DataFrame, col1: str, col2: str,
                                method: str = "pearson") -> ToolResult:
        """Test correlation between two variables."""
        result = correlation_test(df, col1, col2, method)

        # Create scatter plot data
        scatter_data = df[[col1, col2]].dropna().head(500).to_dict('records')

        return ToolResult(
            task="correlation_test",
            raw_output=result,
            chart_data=scatter_data,
            metrics={
                "correlation": round(result["correlation"], 4),
                "p_value": round(result["p_value"], 6),
                "method": result["method"],
            },
        )

    async def t_test(self, df: pd.DataFrame, column: str,
                     pop_mean: float = 0.0) -> ToolResult:
        """One-sample t-test."""
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

    async def two_sample_ttest(self, df: pd.DataFrame, col1: str, col2: str,
                                equal_var: bool = False) -> ToolResult:
        """Two-sample t-test for independent samples."""
        series1 = df[col1].dropna()
        series2 = df[col2].dropna()

        # Create boxplot data
        boxplot_data = [
            {"series": col1, "values": series1.head(100).tolist()},
            {"series": col2, "values": series2.head(100).tolist()},
        ]

        result = two_sample_ttest(series1, series2, equal_var)

        return ToolResult(
            task="two_sample_ttest",
            raw_output=result,
            chart_data=boxplot_data,
            metrics={
                "t_statistic": round(result["t_statistic"], 4),
                "p_value": round(result["p_value"], 6),
                "sample1_mean": round(result["sample1_mean"], 4),
                "sample2_mean": round(result["sample2_mean"], 4),
                "equal_var": equal_var,
            },
        )

    async def anova_test(self, df: pd.DataFrame, value_col: str,
                         group_col: str) -> ToolResult:
        """One-way ANOVA for multiple groups."""
        groups = df[group_col].unique()
        series = [df[df[group_col] == g][value_col] for g in groups]

        result = anova_test(*series)

        return ToolResult(
            task="anova_test",
            raw_output=result,
            chart_data=[],
            metrics={
                "f_statistic": round(result["f_statistic"], 4),
                "p_value": round(result["p_value"], 6),
                "n_groups": result["n_groups"],
                "significant": result["p_value"] < 0.05,
            },
        )

    async def test_randomness(self, df: pd.DataFrame, column: str) -> ToolResult:
        """Wald-Wolfowitz runs test for randomness."""
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

    async def calculate_entropy(self, df: pd.DataFrame, column: str,
                                bins: int = 10) -> ToolResult:
        """Calculate Shannon entropy of a series."""
        series = df[column]
        entropy_val = entropy(series, bins)

        # Histogram data
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
                "max_entropy": round(np.log2(bins), 4),
                "entropy_ratio": round(entropy_val / np.log2(bins), 4) if bins > 1 else 0,
            },
        )

    async def interpret(self, query: str, tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Interpret statistical test results using LLM."""
        results_text = json.dumps(tool_results, indent=2)

        prompt = f"""
You are a data analyst. Interpret these statistical test results for the query: "{query}"

Results:
{results_text}

Provide:
1. Key findings with statistical significance
2. Probable causes or explanations
3. Confidence level (0-1)
4. Recommendations

Output as JSON:
{{
    "key_findings": ["finding 1", "finding 2"],
    "probable_causes": ["cause 1", "cause 2"],
    "confidence": 0.85,
    "recommendations": ["action 1", "action 2"]
}}
"""

        try:
            response_text = await ollama_client.generate(self.model, prompt)
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
            return {
                "key_findings": ["Analysis completed"],
                "probable_causes": ["N/A"],
                "confidence": 0.5,
                "recommendations": ["Review results manually"],
            }
        except Exception as e:
            logger.error(f"Statistical interpretation error: {e}")
            return {
                "error": str(e),
                "key_findings": [],
                "probable_causes": [],
                "confidence": 0.0,
                "recommendations": [],
            }


statistical_agent = StatisticalAgent()
