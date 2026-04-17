from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import uuid
import os
from datetime import datetime
from models.schemas import DatasetMeta
from config import settings
from db.connections import db_manager

router = APIRouter(prefix="/api/upload", tags=["upload"])

@router.post("")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    dataset_id = str(uuid.uuid4())
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    file_path = os.path.join(settings.DATA_DIR, f"{dataset_id}.csv")

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        df = pd.read_csv(file_path)
        cols = df.columns.tolist()

        # Auto-detect timestamp column (first parseable datetime column)
        timestamp_col = cols[0]
        for col in cols:
            try:
                pd.to_datetime(df[col])
                timestamp_col = col
                break
            except Exception:
                continue

        value_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

        parsed_dates = pd.to_datetime(df[timestamp_col], errors='coerce').dropna()
        date_range = [str(parsed_dates.min()), str(parsed_dates.max())]

        meta = DatasetMeta(
            dataset_id=dataset_id,
            filename=file.filename,
            timestamp_col=timestamp_col,
            value_cols=value_cols,
            row_count=len(df),
            date_range=date_range,
        )

        await db_manager.db.datasets.insert_one(meta.model_dump(mode="json"))

        return {
            "dataset_id": dataset_id,
            "columns": cols,
            "detected_timestamp_col": timestamp_col,
            "detected_value_cols": value_cols,
            "row_count": len(df),
            "preview": df.head(5).to_dict('records'),
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {e}")
