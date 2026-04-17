from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import pandas as pd
import numpy as np
import traceback
import math
import os
import json
from models.schemas import DatasetMeta
from agents.orchestrator import orchestrator
from agents.query_interpreter import resolve_target_column
from db.connections import db_manager
from config import settings

# HTTP router
router = APIRouter(prefix="/api/query", tags=["query"])

# WebSocket router — no prefix so path is exactly /ws/query/{session_id}
ws_router = APIRouter(tags=["websocket"])


def sanitise(obj):
    """Recursively convert non-JSON-serialisable types to plain Python types."""
    if isinstance(obj, dict):
        return {k: sanitise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitise(i) for i in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
    if isinstance(obj, np.ndarray):
        return sanitise(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@router.post("")
async def submit_query():
    return {"message": "Use WebSocket at /ws/query/{session_id} for streaming results"}


@ws_router.websocket("/ws/query/{session_id}")
async def query_websocket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)
            dataset_id = payload.get("dataset_id")
            query = payload.get("query")

            if not dataset_id or not query:
                await websocket.send_json({"event": "error", "message": "Missing dataset_id or query"})
                continue

            # 1. Load metadata from MongoDB
            meta_dict = await db_manager.db.datasets.find_one({"dataset_id": dataset_id})
            if not meta_dict:
                await websocket.send_json({"event": "error", "message": f"Dataset {dataset_id} not found in DB"})
                continue

            meta_dict.pop("_id", None)   # remove ObjectId — not JSON-serialisable
            meta = DatasetMeta(**meta_dict)

            # 2. Load CSV from disk
            file_path = os.path.join(settings.DATA_DIR, f"{dataset_id}.csv")
            if not os.path.exists(file_path):
                await websocket.send_json({"event": "error", "message": f"CSV file not found on disk: {file_path}"})
                continue

            df = pd.read_csv(file_path)
            df[meta.timestamp_col] = pd.to_datetime(df[meta.timestamp_col], errors="coerce")

            # ── Smart column selection ──────────────────────────────────────
            # Resolve which value column the user is asking about,
            # then rename it to canonical "value" for the pipeline.
            # All other numeric columns stay in df so correlation/clustering
            # tools can access them via df.select_dtypes(include='number').
            resolved_col = await resolve_target_column(query, meta.value_cols)
            rename_map = {meta.timestamp_col: "timestamp", resolved_col: "value"}
            df_ready = df.rename(columns=rename_map)

            # Ensure the resolved column is sent as event so the frontend can show it
            await websocket.send_json({
                "event": "column_resolved",
                "column": resolved_col,
                "all_columns": meta.value_cols,
            })

            # 3. Stream agent pipeline events
            try:
                async for event in orchestrator.execute_query(query, meta, session_id, df_ready):
                    await websocket.send_json(sanitise(event))
            except Exception as pipeline_err:
                tb = traceback.format_exc()
                print(f"❌ Pipeline error: {pipeline_err}\n{tb}")
                await websocket.send_json({"event": "error", "message": str(pipeline_err)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        tb = traceback.format_exc()
        print(f"❌ WebSocket handler error: {e}\n{tb}")
        try:
            await websocket.send_json({"event": "error", "message": str(e)})
        except Exception:
            pass
