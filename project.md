MATSIS — Technical Requirements Document (v1.0)

1. System Overview
   MATSIS is a fully local, offline-capable multi-agent AI system for time series analysis. It exposes a conversational interface where natural language queries are decomposed into structured execution plans, fulfilled by specialized agents, and returned as human-readable explanations with interactive visualizations. Zero cloud dependencies. All LLM inference runs via Ollama.

2. Tech Stack
   Frontend: React 18 + Vite, TailwindCSS, Recharts (visualizations), React Query (server state), Zustand (client state), Socket.IO client (streaming responses)
   Backend: FastAPI (Python 3.11+), Uvicorn ASGI server, Pydantic v2 for schema validation, Motor (async MongoDB driver), aioredis for queue management, WebSockets for streaming
   LLM Runtime: Ollama (local), models: llama3, mistral, phi3 (selectable per agent)
   Time Series: Prophet, statsmodels (ARIMA), scikit-learn (Isolation Forest), NeuralProphet (LSTM wrapper), pandas, numpy
   Storage: MongoDB 7.x (documents), Redis 7.x (inter-agent message queue + result cache), local filesystem for raw CSV datasets
   Infra (local): Docker Compose orchestrating all services — FastAPI, MongoDB, Redis, Ollama container

3. Directory Structure
   matsis/
   ├── frontend/ # React app
   │ ├── src/
   │ │ ├── pages/ # Upload, Chat, Dashboard
   │ │ ├── components/ # Chart, ChatBubble, AgentTrace
   │ │ ├── stores/ # Zustand slices
   │ │ └── api/ # Axios + WS hooks
   ├── backend/
   │ ├── main.py # FastAPI entry point
   │ ├── routers/
   │ │ ├── upload.py # CSV ingestion
   │ │ ├── query.py # /api/query endpoint
   │ │ └── history.py # Query history
   │ ├── agents/
   │ │ ├── orchestrator.py # Pipeline runner
   │ │ ├── planner.py
   │ │ ├── tool_agent.py
   │ │ ├── analyst.py
   │ │ ├── explainer.py
   │ │ └── memory.py
   │ ├── ts_engine/
   │ │ ├── forecast.py # Prophet, ARIMA, LSTM
   │ │ ├── anomaly.py # Isolation Forest, Z-score
   │ │ └── decompose.py # STL
   │ ├── llm/
   │ │ └── ollama_client.py # Ollama HTTP wrapper
   │ ├── models/ # Pydantic schemas
   │ ├── db/ # MongoDB + Redis connections
   │ └── config.py # Settings, env vars
   ├── docker-compose.yml
   └── .env

4. API Contract
   4.1 Upload Dataset
   POST /api/upload
   Content-Type: multipart/form-data

Body: file (CSV)
Response: {
"dataset_id": "uuid",
"columns": ["date", "sales", "revenue"],
"detected_timestamp_col": "date",
"detected_value_cols": ["sales", "revenue"],
"row_count": 365,
"preview": [...]
}
4.2 Submit Query (streaming)
POST /api/query
Body: {
"dataset_id": "uuid",
"query": "Why did sales spike last month?",
"session_id": "uuid"
}

WebSocket: ws://localhost:8000/ws/query/{session_id}

Streaming events:
{ "event": "agent_start", "agent": "planner" }
{ "event": "agent_done", "agent": "planner", "output": { plan } }
{ "event": "agent_start", "agent": "tool" }
{ "event": "agent_done", "agent": "tool", "output": { anomalies, chart_data } }
{ "event": "agent_start", "agent": "analyst" }
{ "event": "agent_done", "agent": "analyst", "output": { interpretation } }
{ "event": "agent_start", "agent": "explainer" }
{ "event": "stream_token", "token": "The spike..." } ← LLM tokens
{ "event": "complete", "final": { full_response } }
4.3 Query History
GET /api/history?session_id=uuid&limit=20
Response: [{ query, plan, result, timestamp }]

5.  Agent Specifications
    5.1 Planner Agent
    Input: raw query string + dataset metadata (columns, date range, row count)
    LLM prompt pattern:
    You are a time series planning agent. Given the user query and dataset info,
    output ONLY valid JSON with this schema:
    {
    "intent": "anomaly_detection | forecast | decomposition | summary | comparison",
    "steps": [
    { "agent": "tool", "task": "anomaly_detection", "params": { "method": "isolation_forest" } },
    { "agent": "analyst", "task": "interpret" },
    { "agent": "explainer", "task": "explain" }
    ],
    "target_column": "sales",
    "time_window": "last_30_days"
    }
    Output: validated PlanSchema Pydantic object. If JSON parse fails, retry once with an error correction prompt. On second failure, fallback to a default summary plan.
    Model: phi3 (fast, small — planning doesn't need a large model)
    5.2 Tool Agent
    Executes the model task determined by the planner. All execution is pure Python, no LLM involved.
    pythonclass ToolAgent:
    def run(self, task: str, params: dict, df: pd.DataFrame) -> ToolResult:
    dispatch = {
    "forecast": self.\_forecast,
    "anomaly_detection": self.\_anomaly,
    "decomposition": self.\_decompose,
    "summary": self.\_summary,
    }
    return dispatch[task](params, df)

        def _forecast(self, params, df):
            method = params.get("method", "prophet")
            horizon = params.get("horizon", 10)
            if method == "prophet":   return prophet_forecast(df, horizon)
            if method == "arima":     return arima_forecast(df, horizon)
            if method == "lstm":      return lstm_forecast(df, horizon)

        def _anomaly(self, params, df):
            method = params.get("method", "isolation_forest")
            if method == "isolation_forest": return run_isolation_forest(df)
            if method == "zscore":           return run_zscore(df)

    Output schema:
    pythonclass ToolResult(BaseModel):
    task: str
    raw_output: dict # model predictions, anomaly indices, etc.
    chart_data: list[dict] # ready for Recharts on frontend
    metrics: dict # MAPE, RMSE, n_anomalies, etc.
    error: str | None
    5.3 Analyst Agent
    Interprets the ToolResult using the LLM to identify patterns, correlations, and causes. Receives both the tool output and the original data statistics.
    Prompt pattern:
    You are a data analyst. Given anomaly detection results on a sales time series:

- 3 anomalies found on: [2024-01-15, 2024-02-03, 2024-03-22]
- Overall trend: upward (+12% MoM)
- Seasonality: weekly pattern detected

Identify possible causes and patterns. Output JSON:
{ "key_findings": [...], "probable_causes": [...], "confidence": 0.0-1.0 }
Model: llama3 or mistral
5.4 Explainer Agent
Converts the analyst JSON into a flowing natural language response. This is the only agent whose output streams token-by-token to the frontend via WebSocket.
Prompt pattern:
You are a friendly data expert. Convert this structured analysis into a clear,
concise explanation for a non-technical user. Use plain language. 2-3 paragraphs max.

Analysis: { ...analyst output... }
Streaming: Uses stream: True in the Ollama API call. Tokens are forwarded to the WebSocket as stream_token events in real time.
Model: llama3 (best output quality for prose)
5.5 Memory Agent
Persists and retrieves session state so users can reference previous analyses.
pythonclass MemoryAgent:
async def store(self, session_id, query, plan, result): ...
async def retrieve(self, session_id, n=5) -> list[MemoryEntry]: ...
async def get_context_prompt(self, session_id) -> str: # Returns: "Previous queries in this session: ..." # Injected into Analyst + Explainer prompts for continuity
Storage: MongoDB collection sessions, indexed on session_id + timestamp. TTL index: 7-day auto-expiry.

6. Time Series Engine
   6.1 Forecasting
   Prophet (primary, recommended for most cases):
   pythondef prophet_forecast(df: pd.DataFrame, horizon: int, freq: str = "D") -> ToolResult:
   model = Prophet(seasonality_mode="multiplicative", yearly_seasonality=True)
   model.fit(df.rename(columns={"timestamp": "ds", "value": "y"}))
   future = model.make_future_dataframe(periods=horizon, freq=freq)
   forecast = model.predict(future) # returns yhat, yhat_lower, yhat_upper for uncertainty intervals
   ARIMA (for stationary series with clear autocorrelation):
   pythonfrom statsmodels.tsa.arima.model import ARIMA

# Auto-selection of (p,d,q) via AIC minimization across a grid

# Fallback: (1,1,1) if grid search times out

LSTM via NeuralProphet (for multivariate + complex nonlinear patterns):
pythonfrom neuralprophet import NeuralProphet

# Only invoked when the planner explicitly selects method="lstm"

# Requires more data (>200 points) — validated before invocation

6.2 Anomaly Detection
Isolation Forest (default, fast, no distributional assumption):
pythonfrom sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05, random_state=42)
df["anomaly"] = clf.fit_predict(df[["value"]])

# -1 = anomaly, 1 = normal

Z-score (simple, interpretable, good for normally distributed residuals):
pythondf["zscore"] = (df["value"] - df["value"].mean()) / df["value"].std()
df["anomaly"] = df["zscore"].abs() > threshold # default: 3σ
6.3 Decomposition (STL)
pythonfrom statsmodels.tsa.seasonal import STL
stl = STL(df["value"], period=7) # period auto-detected from data freq
result = stl.fit()

# Returns: trend, seasonal, residual components → sent to frontend as chart series

7.  Ollama Integration
    pythonclass OllamaClient:
    BASE_URL = "http://localhost:11434"

        async def generate(self, model: str, prompt: str, stream: bool = False):
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(
                    f"{self.BASE_URL}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": stream}
                )
                if not stream:
                    return response.json()["response"]
                # For streaming: yield tokens as they arrive
                async for line in response.aiter_lines():
                    chunk = json.loads(line)
                    if not chunk.get("done"):
                        yield chunk["response"]

        async def health_check(self) -> bool:
            try:
                r = await httpx.AsyncClient().get(f"{self.BASE_URL}/api/tags")
                return r.status_code == 200
            except: return False

    Model selection per agent:
    AgentModelReasonPlannerphi3Fast JSON generation, small contextAnalystmistralGood at structured reasoningExplainerllama3Best prose quality
    Models are configurable via .env. The system validates model availability at startup via health_check.

8.  Frontend Architecture
    8.1 Pages
    Upload Page (/upload): Drag-and-drop CSV zone. On upload, calls POST /api/upload, displays column detection results, lets user confirm timestamp and value columns, stores dataset_id in Zustand global state.
    Chat Page (/chat): Primary interface. Left panel shows conversation history. Right panel shows the active agent trace (which agent is running, in real time). Input bar at bottom. Each AI response includes an expandable "Agent trace" accordion showing the plan JSON and intermediate outputs.
    Dashboard (/dashboard): Persistent visualization space. All chart_data from previous queries is rendered here using Recharts. Supports: line charts (forecast + actuals), scatter overlays (anomalies highlighted in red), multi-series decomposition chart (trend / seasonal / residual stacked).
    8.2 Agent Trace Component
    Shows a live step-by-step view of which agent is active, with animated pulse indicator:
    [●] Planner → Detected: anomaly_detection
    [●] Tool Agent → Running: Isolation Forest (365 rows)
    [●] Analyst → Interpreting 3 anomalies found
    [●] Explainer → Generating explanation...
    8.3 State Management (Zustand)
    typescriptinterface AppStore {
    datasetId: string | null;
    sessionId: string;
    messages: Message[];
    agentTrace: AgentStep[];
    charts: ChartData[];
    isStreaming: boolean;
    }

9.  Data Models (Pydantic / MongoDB)
    pythonclass DatasetMeta(BaseModel):
    dataset_id: str
    filename: str
    timestamp_col: str
    value_cols: list[str]
    row_count: int
    date_range: tuple[datetime, datetime]
    uploaded_at: datetime

class QueryRecord(BaseModel):
query_id: str
session_id: str
raw_query: str
plan: dict
tool_result: dict
analysis: dict
explanation: str
latency_ms: int
created_at: datetime

class SessionMemory(BaseModel):
session_id: str
entries: list[QueryRecord]
created_at: datetime
expires_at: datetime # TTL: 7 days

10. Error Handling Strategy
    Every agent wraps its execution in a try/except with typed AgentError. The orchestrator catches errors at each step and either retries (LLM calls, max 2 retries), falls back to a simpler model/method, or short-circuits to a graceful degradation response — never a raw traceback to the user.
    LLM JSON parse failure: If Planner returns non-JSON, the orchestrator sends a corrective follow-up prompt: "Your previous response was not valid JSON. Here is the error: {e}. Please respond with valid JSON only." If it fails again, fallback to intent: "summary".
    Ollama unavailable: On startup, the system checks Ollama health. If unavailable, it returns a clear error in the API response: "Local LLM is not running. Please start Ollama with: ollama serve".
    Dataset too small for LSTM: Validated before invocation. If fewer than 200 data points, auto-downgrades to Prophet and notifies the user in the explanation.

11. Performance Targets
    OperationTarget LatencyCSV upload + column detection< 500msPlanner agent (phi3)< 2sTool agent (Prophet forecast)< 3sTool agent (Isolation Forest)< 1sAnalyst agent (mistral)< 4sExplainer first token (streaming)< 2sEnd-to-end (simple query)< 10s
    Optimizations: Redis caching for repeated queries on the same dataset (key: hash(dataset_id + query)). Prophet model objects cached in-process after first fit. Smaller Ollama models used for planning to minimize wait on the first response.

12. Docker Compose Setup
    yamlservices:
    backend:
    build: ./backend
    ports: ["8000:8000"]
    environment: - MONGO_URL=mongodb://mongo:27017 - REDIS_URL=redis://redis:6379 - OLLAMA_URL=http://host.docker.internal:11434
    volumes: - ./data:/app/data

frontend:
build: ./frontend
ports: ["3000:3000"]

mongo:
image: mongo:7
volumes: ["mongo_data:/data/db"]

redis:
image: redis:7-alpine

volumes:
mongo_data:
Ollama runs on the host (not in Docker) to access GPU acceleration directly. The backend connects via host.docker.internal.

13. Security Notes (Local Deployment)
    Since this is a local-only system, security is minimal by design. However: CSV uploads are validated for file size (max 50MB), MIME type, and sanitized column names before processing. No user auth required for local mode, but a session_id UUID is generated per browser session to isolate memory. MongoDB and Redis bind to 127.0.0.1 only, not exposed externally.

This TRD is complete and production-ready for a local deployment. The architecture cleanly separates every concern — inference, execution, reasoning, and presentation — making it straightforward to swap out individual agents, models, or TS methods independently without touching the rest of the system.
