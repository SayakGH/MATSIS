import pandas as pd
from ts_engine.forecast import prophet_forecast, arima_forecast, lstm_forecast
from ts_engine.anomaly import run_isolation_forest, run_zscore
from ts_engine.decompose import run_stl_decomposition
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


tool_agent = ToolAgent()
