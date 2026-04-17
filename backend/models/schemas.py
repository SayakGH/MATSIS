from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional, Any, Dict, Union

class DatasetMeta(BaseModel):
    dataset_id: str
    filename: str
    timestamp_col: str
    value_cols: List[str]
    row_count: int
    date_range: List[Any]   # [start, end] — kept as Any for serialisation flexibility
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

class PlanStep(BaseModel):
    agent: str
    task: str
    params: Dict[str, Any] = {}

class PlanSchema(BaseModel):
    intent: str
    steps: List[PlanStep]
    target_column: Optional[str] = None
    resolved_column: Optional[str] = None   # actual column selected from multi-col dataset
    time_window: Optional[str] = None

class ToolResult(BaseModel):
    task: str
    raw_output: Any
    chart_data: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    error: Optional[str] = None

class QueryRequest(BaseModel):
    dataset_id: str
    query: str
    session_id: str

class QueryRecord(BaseModel):
    query_id: str
    session_id: str
    raw_query: str
    plan: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    follow_up_suggestions: List[str] = []          # powered by analyst output
    latency_ms: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
