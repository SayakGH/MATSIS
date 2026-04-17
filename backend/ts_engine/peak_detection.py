import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


def run_peak_detection(df: pd.DataFrame, order: int = 3) -> ToolResult:
    """
    Detect local maxima (peaks) and minima (troughs) in the value column.

    Parameters
    ----------
    order : int
        How many surrounding points must be lower/higher for a point to be
        considered a peak/trough (sensitivity). Default 3.
    """
    try:
        df = df.copy().reset_index(drop=True)
        values = df["value"].to_numpy()

        order = max(1, min(order, len(values) // 4))

        peak_idx = argrelextrema(values, np.greater, order=order)[0]
        trough_idx = argrelextrema(values, np.less, order=order)[0]

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype(str)

        # Build enriched chart data
        peak_set = set(peak_idx.tolist())
        trough_set = set(trough_idx.tolist())
        chart_data = []
        for i, row in df.iterrows():
            entry = {
                "timestamp": row["timestamp"],
                "value": float(row["value"]),
                "is_peak": i in peak_set,
                "is_trough": i in trough_set,
            }
            chart_data.append(entry)

        peaks_data = [
            {"timestamp": df.loc[i, "timestamp"], "value": float(df.loc[i, "value"]), "type": "peak"}
            for i in peak_idx
        ]
        troughs_data = [
            {"timestamp": df.loc[i, "timestamp"], "value": float(df.loc[i, "value"]), "type": "trough"}
            for i in trough_idx
        ]

        return ToolResult(
            task="peak_detection",
            raw_output={"peaks": peaks_data, "troughs": troughs_data},
            chart_data=chart_data,
            metrics={
                "n_peaks": int(len(peak_idx)),
                "n_troughs": int(len(trough_idx)),
                "order": order,
                "highest_peak": float(values[peak_idx].max()) if len(peak_idx) else None,
                "lowest_trough": float(values[trough_idx].min()) if len(trough_idx) else None,
            },
        )

    except Exception as e:
        logger.error(f"Peak detection error: {e}", exc_info=True)
        return ToolResult(task="peak_detection", raw_output={}, chart_data=[], metrics={}, error=str(e))
