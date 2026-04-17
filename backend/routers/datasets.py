from fastapi import APIRouter, HTTPException
from db.connections import db_manager
import pandas as pd
import os
from config import settings

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.get("")
async def list_datasets():
    """Return all uploaded datasets (metadata only)."""
    cursor = db_manager.db.datasets.find({}, {"_id": 0})
    datasets = await cursor.to_list(length=200)
    return datasets


@router.get("/{dataset_id}/preview")
async def dataset_preview(dataset_id: str, rows: int = 200):
    """Return the first N rows of a dataset as chart-ready JSON."""
    meta = await db_manager.db.datasets.find_one({"dataset_id": dataset_id}, {"_id": 0})
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = os.path.join(settings.DATA_DIR, f"{dataset_id}.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="CSV file not found on disk")

    df = pd.read_csv(file_path, nrows=rows)
    ts_col = meta["timestamp_col"]
    val_col = meta["value_cols"][0] if meta["value_cols"] else None

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col])

    chart = pd.DataFrame()
    chart["timestamp"] = df[ts_col].astype(str)
    if val_col and val_col in df.columns:
        chart["value"] = pd.to_numeric(df[val_col], errors="coerce")

    return {
        "dataset_id": dataset_id,
        "meta": meta,
        "chart_data": chart.dropna().to_dict("records"),
    }
