import pandas as pd
from models.schemas import ToolResult
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

def _df_to_chart(df: pd.DataFrame, extra_cols: list = None) -> list:
    """Convert a dataframe with a 'timestamp' and 'value' column to chart-friendly dicts."""
    cols = ['timestamp', 'value'] + (extra_cols or [])
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out['timestamp'] = pd.to_datetime(out['timestamp'], errors='coerce').astype(str)
    return out.to_dict('records')


def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05) -> ToolResult:
    try:
        clf = IsolationForest(contamination=contamination, random_state=42)
        df = df.copy()
        df['anomaly_score'] = clf.fit_predict(df[['value']].values)
        df['is_anomaly'] = df['anomaly_score'] == -1
        return ToolResult(
            task="anomaly_detection",
            raw_output=df[df['is_anomaly']][['timestamp', 'value']].to_dict('records'),
            chart_data=_df_to_chart(df, ['is_anomaly']),
            metrics={"method": "isolation_forest", "n_anomalies": int(df['is_anomaly'].sum())}
        )
    except Exception as e:
        logger.error(f"Isolation Forest error: {e}")
        return ToolResult(task="anomaly_detection", raw_output={}, chart_data=[], metrics={}, error=str(e))


def run_zscore(df: pd.DataFrame, threshold: float = 3.0) -> ToolResult:
    try:
        df = df.copy()
        df['zscore'] = (df['value'] - df['value'].mean()) / df['value'].std()
        df['is_anomaly'] = df['zscore'].abs() > threshold
        return ToolResult(
            task="anomaly_detection",
            raw_output=df[df['is_anomaly']][['timestamp', 'value']].to_dict('records'),
            chart_data=_df_to_chart(df, ['is_anomaly']),
            metrics={"method": "zscore", "n_anomalies": int(df['is_anomaly'].sum()), "threshold": threshold}
        )
    except Exception as e:
        logger.error(f"Z-score error: {e}")
        return ToolResult(task="anomaly_detection", raw_output={}, chart_data=[], metrics={}, error=str(e))
