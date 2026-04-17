import pandas as pd
from statsmodels.tsa.seasonal import STL
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)

def run_stl_decomposition(df: pd.DataFrame, period: int = 7) -> ToolResult:
    try:
        # STL requires period >= 2 and at least 2*period observations
        period = max(2, int(period))
        min_rows = period * 2
        if len(df) < min_rows:
            # Auto-shrink period so it always works on small datasets
            period = max(2, len(df) // 2)
            logger.warning(
                f"STL: not enough rows ({len(df)}) for requested period — "
                f"auto-adjusted to period={period}"
            )

        res = STL(df['value'], period=period).fit()
        decomp_df = pd.DataFrame({
            'timestamp': pd.to_datetime(df['timestamp'], errors='coerce').astype(str),
            'observed': df['value'].values,
            'trend': res.trend,
            'seasonal': res.seasonal,
            'resid': res.resid,
        })
        return ToolResult(
            task="decomposition",
            raw_output=decomp_df.to_dict('records'),
            chart_data=decomp_df.to_dict('records'),
            metrics={"method": "stl", "period": period, "row_count": len(df)}
        )
    except Exception as e:
        logger.error(f"STL decomposition error: {e}")
        return ToolResult(task="decomposition", raw_output={}, chart_data=[], metrics={}, error=str(e))
