# AI Agents Architecture - Deep Dive

## Overview

The MATSIS backend uses a **multi-agent pipeline** for time series analysis. Each agent has a single responsibility and communicates through well-defined interfaces (dictionaries and Pydantic models).

---

## System Flow

```
User Query → Orchestrator → Preprocessor → Planner
                                      ↓
                      ┌───────────────┼───────────────┐
                      ↓               ↓               ↓
                  Tool Agent      Analyst Agent   Explainer Agent
                      ↓               ↓
                  Memory Agent ←───┘
```

---

## Agent Breakdown

### 1. Preprocessor Agent (`agents/preprocessor.py`)

**Purpose**: Data cleaning before analysis

**Functionality**:
- Handles missing values (fill, interpolate, or drop)
- Removes duplicates
- Standardizes column names
- Type conversion for timestamps and numeric columns

**Key Methods**:
```python
async def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Returns cleaned DataFrame
```

**Why Separate?**: Keeps analysis agents focused on their core tasks without data quality concerns.

---

### 2. Planner Agent (`agents/planner.py`)

**Purpose**: Query classification and workflow routing

**How It Works**:
1. Receives user query + dataset metadata
2. LLM (Phi-3) classifies intent into one of 14 categories:
   - `forecast`, `anomaly_detection`, `decomposition`, `summary`
   - `correlation`, `rolling_stats`, `peak_detection`, `regression`, `clustering`
   - **Statistical tests**: `outlier_detection`, `stationarity_test`, `statistical_correlation`, `one_sample_ttest`, `two_sample_ttest`, `runs_test`, `entropy`
3. Returns a `PlanSchema` with ordered steps

**Hardcoded Templates**:
```python
_PLANS = {
    "forecast": [
        PlanStep(agent="tool", task="forecast", params={"method": "prophet", "horizon": 14}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    # ... more templates
}
```

**Keyword Fallback**:
```python
_INTENT_KEYWORDS = {
    "outlier_detection": ["outlier", "anomaly", "spike", "extreme", "deviation"],
    "stationarity_test": ["stationar", "trend stationary", "diffuse", "random walk"],
    # ...
}
```

**Query Interpreter Integration**:
- After intent detection, `query_interpreter.extract_params()` fine-tunes parameters from natural language
- Example: "forecast next 30 days" → `horizon: 30`

---

### 3. Tool Agent (`agents/tool_agent.py`)

**Purpose**: Execute domain-specific analysis tasks

**Dispatch Pattern**:
```python
dispatch = {
    "forecast": self._forecast,
    "anomaly_detection": self._anomaly,
    "stationarity_test": self._stationarity_test,
    "statistical_correlation": self._statistical_correlation,
    # ... 20+ handlers
}
```

**Available Tools**:

| Task | Method | Output |
|------|--------|--------|
| `forecast` | Prophet/ARIMA/LSTM | Future predictions + confidence intervals |
| `outlier_detection` | IQR or Z-score | Anomaly indices + percentages |
| `stationarity_test` | ADF (from statsmodels) | `is_stationary` boolean + test stats |
| `statistical_correlation` | Pearson/Spearman/Kendall | Correlation coefficient + p-value |
| `one_sample_ttest` | scipy.stats.ttest_1samp | t-statistic + p-value |
| `two_sample_ttest` | scipy.stats.ttest_ind | Comparison of two groups |
| `runs_test` | Wald-Wolfowitz | Randomness assessment |
| `entropy` | Shannon entropy | Information content metric |

**Returns**: `ToolResult` Pydantic model:
```python
class ToolResult(BaseModel):
    task: str
    raw_output: Any
    chart_data: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    error: Optional[str] = None
```

---

### 4. Statistical Agent (`agents/statistical_agent.py`)

**Purpose**: Advanced statistical hypothesis testing

**Key Methods**:
```python
async def test_stationarity(df, column) -> ToolResult
async def detect_outliers(df, column, method) -> ToolResult
async def t_test(df, column, pop_mean) -> ToolResult
async def two_sample_ttest(df, col1, col2) -> ToolResult
async def interpret(query, tool_results) -> Dict
```

**Difference from Tool Agent**:
- Uses **LLM (Mistral)** to interpret complex results
- Returns structured JSON with key findings
- Can process multiple tool results for holistic analysis

---

### 5. Analyst Agent (`agents/analyst.py`)

**Purpose**: Interpret tool results in context

**Workflow**:
1. Receives query + ToolResult
2. LLM (Phi-3) generates:
   - Key findings
   - Probable causes
   - Confidence score (0-1)

**Prompt Template**:
```
You are a data analyst. Given the results of a time series analysis task "{task}", 
interpret the findings for the user query: "{query}".

Tool Results:
- Metrics: {metrics}
- Sample raw output: {raw_output}

Identify key findings, probable causes, and your confidence.
```

**Output**:
```json
{
    "key_findings": ["Sales increased 15% this month"],
    "probable_causes": ["New marketing campaign", "Seasonal effect"],
    "confidence": 0.85,
    "follow_up_suggestions": ["Investigate correlation with marketing spend"]
}
```

---

### 6. Explainer Agent (`agents/explainer.py`)

**Purpose**: Convert technical analysis to natural language

**Features**:
- **Streaming**: Tokens generated incrementally for responsive UX
- **LLM**: Llama3 for high-quality explanations
- **Context-aware**: Uses session history from Memory Agent

**Streaming Pipeline**:
```python
async for token in ollama_client.generate_stream(model, prompt):
    yield token  # Sent via WebSocket in real-time
```

**What It Explains**:
- Business implications of findings
- Actionable recommendations
- Limitations/caveats

---

### 7. Memory Agent (`agents/memory.py`)

**Purpose**: Conversation history and context persistence

**Storage**: MongoDB (no Redis required)

**Operations**:
```python
async def store_query(session_id, record)  # Save full analysis
async def get_history(session_id, limit)   # Retrieve past 10 queries
async def get_context_prompt(session_id)   # Build context for current query
```

**Context Build-up**:
```
Previous conversation:
User: What are sales trends?
Assistant: Sales increased 12%...

User: Any outliers?
Assistant: Found 3 spikes in November...
```

**TTL**: 7-day expiry for sessions

---

### 8. Orchestrator (`agents/orchestrator.py`)

**Purpose**: Coordinate agent pipeline and events

**Execution Stages**:

#### Stage 1: Preprocess + Plan (Parallel)
```python
df_clean, plan = await asyncio.gather(
    asyncio.to_thread(preprocessor_agent.clean, df),  # CPU-bound
    planner_agent.plan(query, dataset_meta),          # IO-bound
)
```
- Preprocessor runs in thread (CPU-bound task)
- Planner runs async (LLM call)

#### Stage 2: Tool Execution (Sequential)
```python
for step in plan.steps:
    if step.agent == "tool":
        tool_result = tool_agent.run(step.task, step.params, df_clean)
    elif step.agent == "analyst":
        analysis = await analyst_agent.analyze(query, tool_result)
    elif step.agent == "explainer":
        # Streaming generation
```

#### Stage 3: Parallel Analyst + Context Warming
```python
analysis, _ = await asyncio.gather(
    analyst_agent.analyze(query, tool_result, session_id),
    memory_agent.get_context_prompt(session_id),  # Warm cache in background
)
```

---

## Agent Communication Protocol

### Data Flow

```
Query → Orchestrator
    ↓
[Agent Start Event] → Client
    ↓
[Tool/Analysis Result] → Client (via WebSocket)
    ↓
[Agent Done Event] → Client
    ↓
[Stream Tokens] → Client (for explainer)
    ↓
[Complete Event] → Client
```

### Event Types
```json
{"event": "agent_start", "agent": "planner"}
{"event": "agent_done", "agent": "planner", "output": {...}}
{"event": "agent_start", "agent": "tool", "task": "forecast"}
{"event": "agent_done", "agent": "tool", "output": {...}}
{"event": "stream_token", "token": "The"}
{"event": "complete", "final": {...}}
{"event": "error", "message": "..."}
```

---

## Statistical Tools Deep Dive

### Outlier Detection

**IQR Method**:
```
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
Outliers = values outside [Lower, Upper]
```

**Z-Score Method**:
```
z = (x - mean) / std
Outliers = |z| > 3.0
```

### Stationarity Test (ADF)

**Null Hypothesis**: Series has a unit root (non-stationary)

**Interpretation**:
- `p-value < 0.05` → Reject H₀ → **Stationary**
- `p-value ≥ 0.05` → Fail to reject H₀ → **Non-stationary**

**Metrics**:
- `adf_statistic`: Test statistic (more negative = more stationary)
- `p_value`: Significance level
- `critical_values`: 1%, 5%, 10% thresholds

### Correlation Test

**Pearson**: Linear relationship (parametric)
**Spearman**: Monotonic relationship (non-parametric/rank-based)
**Kendall**: Rank-based, smaller samples

**Output**:
```json
{
    "correlation": 0.75,
    "p_value": 0.001,
    "method": "pearson"
}
```

**Interpretation**:
- `|r| > 0.7`: Strong
- `|r| > 0.5`: Moderate
- `|r| > 0.3`: Weak
- `p < 0.05`: Statistically significant

### T-Tests

**One-Sample**:
```
H₀: μ = μ₀ (sample mean = population mean)
H₁: μ ≠ μ₀
```

**Two-Sample**:
```
H₀: μ₁ = μ₂ (two groups same mean)
H₁: μ₁ ≠ μ₂
```

### Runs Test (Wald-Wolfowitz)

**Purpose**: Test if data is randomly distributed

**Method**:
1. Find median
2. Count "runs" (sequences above/below median)
3. Compare to expected runs: `E = (2*n₁*n₂)/(n₁+n₂) + 1`
4. Z-score: `(runs - E) / sqrt(variance)`

**Interpretation**:
- `is_random = True`: No pattern detected
- `is_random = False`: Sequential pattern exists

### Shannon Entropy

```
H(X) = -Σ p(x) * log₂(p(x))
```

**Interpretation**:
- `H ≈ 0`: Highly predictable (one value dominates)
- `H ≈ log₂(n)`: Maximum uncertainty (uniform distribution)

---

## Performance Optimizations

### 1. Parallel Stage 1
Preprocessing (CPU) + Planning (IO) run concurrently.

### 2. Context Prefetching
Memory agent prefetches session history while analyst runs.

### 3. Thread Pool for CPU Tasks
```python
asyncio.to_thread(preprocessor_agent.clean, df)
```

### 4. LLM Timeout Protection
```python
await asyncio.wait_for(llm_call(), timeout=20)
```

### 5. Streaming Responses
Explainer streams tokens → client shows partial results immediately.

---

## Error Handling

### LLM Timeout
```python
except asyncio.TimeoutError:
    intent = _keyword_intent(query)
    logger.warning("LLM timed out — using keyword fallback")
```

### Agent Errors
```python
if tool_result.error:
    yield {"event": "error", "message": tool_result.error}
    return  # Short-circuit pipeline
```

### Safe Defaults
- Unrecognized intent → `"summary"`
- Empty results → Empty lists/dicts
- Missing columns → Error message in ToolResult

---

## Adding New Agents

### Step 1: Create Agent File
```python
# agents/new_agent.py
class NewAgent:
    async def run(self, params, df):
        return {"result": "data"}

new_agent = NewAgent()
```

### Step 2: Add to Orchestrator
```python
from agents.new_agent import new_agent

# In orchestrator.py execute_query():
elif step.agent == "new_agent":
    result = new_agent.run(step.params, df)
    yield {"event": "agent_done", "agent": "new_agent", "output": result}
```

### Step 3: Add to Planner
```python
# In planner.py
_PLANS["new_task"] = [
    PlanStep(agent="new_agent", task="new_task", params={...}),
]

_INTENT_KEYWORDS["new_task"] = ["keyword1", "keyword2"]
```

---

## Configuration

```python
# config.py
settings.PLANNER_MODEL = "phi3"       # Intent classifier
settings.ANALYST_MODEL = "mistral"    # Result interpretation
settings.EXPLAINER_MODEL = "llama3"   # Natural language generation
```

All models served via **Ollama** (local LLM server).

---

## Monitoring

### Key Metrics
- **Latency**: Time from query to complete
- **Agent timings**: Time per agent (planner, tool, analyst, explainer)
- **LLM failures**: Count of fallbacks due to timeouts/errors
- **Memory usage**: MongoDB document count per session

### Event Stream
Clients receive real-time events via WebSocket for progress tracking.

---

## Current Limitations

1. **Sequential Tool Execution**: Cannot parallelize independent tool tasks
2. **No Caching**: Same queries re-run analysis (Redis could help)
3. **Single Thread for Preprocessing**: Large datasets block other requests
4. **No Agent Pool**: Each request spawns new agent instances

## Future Improvements

- Parallel tool execution for independent tasks
- Query result caching
- Agent pools for concurrent requests
- Dynamic plan generation (LLM-controlled workflow)
- Auto-correction for common query patterns
