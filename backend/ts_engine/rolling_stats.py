import pandas as pd
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


def run_rolling_stats(df: pd.DataFrame, window: int = 7) -> ToolResult:
    """
    Compute sliding-window statistics (mean, std, min, max) over the value column.
    """
    try:
        df = df.copy()
        window = max(2, min(window, len(df) // 2))

        roll = df["value"].rolling(window=window, min_periods=1)
        df["rolling_mean"] = roll.mean()
        df["rolling_std"] = roll.std().fillna(0)
        df["rolling_min"] = roll.min()
        df["rolling_max"] = roll.max()

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype(str)

        chart_cols = ["timestamp", "value", "rolling_mean", "rolling_std", "rolling_min", "rolling_max"]
        chart_data = df[chart_cols].to_dict("records")

        # Cast numpy floats to Python floats
        for row in chart_data:
            for k, v in row.items():
                if k != "timestamp" and v is not None:
                    try:
                        row[k] = float(v)
                    except (TypeError, ValueError):
                        pass

        return ToolResult(
            task="rolling_stats",
            raw_output=chart_data,
            chart_data=chart_data,
            metrics={
                "window": window,
                "overall_mean": float(df["rolling_mean"].iloc[-1]),
                "overall_std": float(df["rolling_std"].iloc[-1]),
                "row_count": int(len(df)),
            },
        )

    except Exception as e:
        logger.error(f"Rolling stats error: {e}", exc_info=True)
        return ToolResult(task="rolling_stats", raw_output={}, chart_data=[], metrics={}, error=str(e))
