import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)


def run_clustering(df: pd.DataFrame, n_clusters: int = 3) -> ToolResult:
    """
    Apply K-Means clustering on [time_index, value] to find behavioural segments.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (default 3). Capped to min(n_clusters, n_rows // 5).
    """
    try:
        df = df.copy().reset_index(drop=True)

        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        t0 = ts.min()
        t_index = ((ts - t0).dt.total_seconds() / 86400).fillna(0).values

        y = df["value"].fillna(df["value"].median()).to_numpy()
        X = np.column_stack([t_index, y])

        # Scale features so time and value are comparable
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        k = max(2, min(n_clusters, len(df) // 5))
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)

        # Centroids in original space
        centroids_scaled = model.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)

        df["timestamp"] = ts.astype(str)
        df["cluster"] = labels.tolist()

        chart_data = df[["timestamp", "value", "cluster"]].copy()
        chart_data["value"] = chart_data["value"].apply(
            lambda v: float(v) if v is not None and not pd.isna(v) else None
        )
        chart_data = chart_data.to_dict("records")

        centroid_records = [
            {"cluster": int(i), "center_time_days": float(c[0]), "center_value": float(c[1])}
            for i, c in enumerate(centroids)
        ]

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

        return ToolResult(
            task="clustering",
            raw_output={"centroids": centroid_records, "cluster_sizes": cluster_sizes},
            chart_data=chart_data,
            metrics={
                "n_clusters": int(k),
                "inertia": float(model.inertia_),
                "cluster_sizes": cluster_sizes,
            },
        )

    except Exception as e:
        logger.error(f"Clustering error: {e}", exc_info=True)
        return ToolResult(task="clustering", raw_output={}, chart_data=[], metrics={}, error=str(e))
