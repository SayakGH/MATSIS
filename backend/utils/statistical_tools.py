"""Statistical tools for time series analysis."""
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


def is_stationary(series: pd.Series, alpha: float = 0.05) -> Tuple[bool, Dict[str, float]]:
    """ADF test for stationarity.

    Returns:
        Tuple of (is_stationary, test_results)
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    return p_value < alpha, {
        "adf_statistic": result[0],
        "p_value": result[1],
        "critical_values": result[4],
        "used_lag": result[2],
        "nobs": result[3],
    }


def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """IQR-based outlier detection.

    Returns boolean mask where True indicates an outlier.
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score based outlier detection.

    Returns boolean mask where True indicates an outlier.
    """
    z_scores = np.abs(stats.zscore(series.dropna()))
    return pd.Series(z_scores > threshold, index=series.index)


def confidence_interval(series: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
    """Compute confidence interval for mean."""
    n = len(series)
    mean = series.mean()
    std_err = series.std() / np.sqrt(n)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h


def correlation_test(df: pd.DataFrame, col1: str, col2: str,
                     method: str = "pearson") -> Dict[str, Any]:
    """Correlation test with p-value.

    Args:
        df: DataFrame
        col1: First column name
        col2: Second column name
        method: "pearson", "spearman", or "kendall"

    Returns:
        Dictionary with correlation coefficient and p-value
    """
    valid_methods = ["pearson", "spearman", "kendall"]
    if method not in valid_methods:
        method = "pearson"

    corr, p_value = df[[col1, col2]].corr(method=method).iloc[0, 1], 0.0

    # Get p-value using scipy
    if method == "pearson":
        corr, p_value = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
    elif method == "spearman":
        corr, p_value = stats.spearmanr(df[col1].dropna(), df[col2].dropna())
    else:  # kendall
        corr, p_value = stats.kendalltau(df[col1].dropna(), df[col2].dropna())

    return {"correlation": corr, "p_value": p_value, "method": method}


def one_sample_ttest(series: pd.Series, pop_mean: float) -> Dict[str, Any]:
    """One-sample t-test.

    Tests if sample mean differs from population mean.
    """
    t_stat, p_value = stats.ttest_1samp(series.dropna(), pop_mean)
    return {"t_statistic": t_stat, "p_value": p_value, "pop_mean": pop_mean}


def two_sample_ttest(series1: pd.Series, series2: pd.Series,
                     equal_var: bool = False) -> Dict[str, Any]:
    """Two-sample t-test (independent samples).

    Args:
        series1: First sample
        series2: Second sample
        equal_var: Assume equal variances
    """
    t_stat, p_value = stats.ttest_ind(series1.dropna(), series2.dropna(), equal_var=equal_var)
    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "equal_var": equal_var,
        "sample1_mean": series1.mean(),
        "sample2_mean": series2.mean(),
    }


def anova_test(*series: pd.Series) -> Dict[str, Any]:
    """One-way ANOVA for comparing multiple groups."""
    f_stat, p_value = stats.f_oneway(*[s.dropna() for s in series])
    return {
        "f_statistic": f_stat,
        "p_value": p_value,
        "n_groups": len(series),
        "group_means": [s.mean() for s in series],
    }


def seasonal_decomposition_stats(df: pd.DataFrame, period: int = 7) -> Dict[str, Any]:
    """Statistics for seasonal decomposition.

    Calculates decomposition components and their properties.
    """
    values = df['value'].values
    n = len(values)

    # Trend via moving average
    trend = pd.Series(values).rolling(window=period, center=True).mean()

    # Detrend
    detrended = values - trend.fillna(0).values

    # Seasonal component (average by position in period)
    seasonal = np.zeros(n)
    for i in range(period):
        indices = [j for j in range(i, n, period)]
        seasonal[indices] = np.mean(detrended[indices])

    # Residual
    residual = values - trend.fillna(0).values - seasonal

    return {
        "trend_std": trend.std(),
        "seasonal_std": pd.Series(seasonal).std(),
        "residual_std": pd.Series(residual).std(),
        "signal_to_noise": trend.var() / (residual.var() + 1e-8),
        "seasonality_strength": max(0, 1 - (np.var(residual) / np.var(values))),
    }


def entropy(series: pd.Series, bins: int = 10) -> float:
    """Calculate Shannon entropy of a series."""
    hist, _ = np.histogram(series.dropna(), bins=bins)
    probs = hist / hist.sum()
    # Filter out zero probabilities for log
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def mutual_information(df: pd.DataFrame, col1: str, col2: str,
                       bins: int = 10) -> float:
    """Calculate mutual information between two variables."""
    hist, xedges, yedges = np.histogram2d(
        df[col1].dropna(),
        df[col2].dropna(),
        bins=bins
    )
    probs = hist / hist.sum()
    prob_x = probs.sum(axis=1)
    prob_y = probs.sum(axis=0)
    # Filter zeros
    probs = probs[probs > 0]
    prob_x = prob_x[prob_x > 0]
    prob_y = prob_y[prob_y > 0]
    return np.sum(probs * np.log2(probs / (np.outer(prob_x, prob_y) + 1e-8) + 1e-8))


def runs_test(series: pd.Series) -> Dict[str, Any]:
    """Wald-Wolfowitz runs test for randomness."""
    median = series.median()
    runs = 1
    n = len(series)
    for i in range(1, n):
        if (series.iloc[i] > median) != (series.iloc[i-1] > median):
            runs += 1

    # Expected runs and variance
    above = (series > median).sum()
    below = n - above
    expected_runs = (2 * above * below) / n + 1
    variance_runs = (2 * above * below * (2 * above * below - n)) / (n ** 2 * (n - 1))

    if variance_runs > 0:
        z_stat = (runs - expected_runs) / np.sqrt(variance_runs)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    else:
        z_stat = 0
        p_value = 1.0

    return {
        "runs": runs,
        "expected_runs": expected_runs,
        "z_statistic": z_stat,
        "p_value": p_value,
        "is_random": p_value > 0.05,
    }
