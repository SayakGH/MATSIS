import pandas as pd
from statsmodels.tsa.seasonal import STL
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)

def run_stl_decomposition(df: pd.DataFrame, period: int = 7) -> ToolResult:
    try:
        res = STL(df['value'], period=period).fit()
        decomp_df = pd.DataFrame({
            'timestamp': df['timestamp'].astype(str),
            'observed': df['value'].values,
            'trend': res.trend,
            'seasonal': res.seasonal,
            'resid': res.resid,
        })
        return ToolResult(
            task="decomposition",
            raw_output=decomp_df.to_dict('records'),
            chart_data=decomp_df.to_dict('records'),
            metrics={"method": "stl", "period": period}
        )
    except Exception as e:
        logger.error(f"STL decomposition error: {e}")
        return ToolResult(task="decomposition", raw_output={}, chart_data=[], metrics={}, error=str(e))
