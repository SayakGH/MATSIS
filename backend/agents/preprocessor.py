import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PreprocessorAgent:
    """
    Cleans a DataFrame before it is passed to analysis tools.

    Operations (all configurable via params):
    - fill_missing  : "forward" | "backward" | "interpolate" | "mean" (default "forward")
    - clip_outliers : bool — remove IQR-based outliers by clipping to [Q1 − 1.5·IQR, Q3 + 1.5·IQR]
    """

    def clean(self, df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
        params = params or {}
        df = df.copy()

        # ── 1. Ensure timestamp is datetime ──────────────────────────────────
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        n_bad_ts = df["timestamp"].isna().sum()
        if n_bad_ts:
            logger.warning(f"Preprocessor: dropped {n_bad_ts} rows with unparseable timestamps")
            df = df.dropna(subset=["timestamp"])

        # ── 2. Sort chronologically ───────────────────────────────────────────
        df = df.sort_values("timestamp").reset_index(drop=True)

        # ── 3. Fill missing values ────────────────────────────────────────────
        fill_method = params.get("fill_missing", "forward")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if fill_method == "interpolate":
            df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
        elif fill_method == "mean":
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())
        elif fill_method == "backward":
            df[numeric_cols] = df[numeric_cols].bfill()
        else:  # default: forward fill
            df[numeric_cols] = df[numeric_cols].ffill().bfill()  # bfill covers leading NaNs

        # ── 4. IQR outlier clipping (optional) ───────────────────────────────
        if params.get("clip_outliers", False):
            for col in numeric_cols:
                q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_clipped = int(((df[col] < lower) | (df[col] > upper)).sum())
                df[col] = df[col].clip(lower=lower, upper=upper)
                if n_clipped:
                    logger.info(f"Preprocessor: clipped {n_clipped} outliers in '{col}'")

        logger.info(f"Preprocessor: cleaned DataFrame → {len(df)} rows, {len(df.columns)} columns")
        return df


preprocessor_agent = PreprocessorAgent()
