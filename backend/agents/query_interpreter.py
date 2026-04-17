import asyncio
import json
import re
import difflib
import logging
from typing import List
from llm.ollama_client import ollama_client
from config import settings

logger = logging.getLogger(__name__)

# ── Param extraction regexes ───────────────────────────────────────────────────
_HORIZON_RE = re.compile(r"\b(\d+)\s*(day|week|month)s?\b", re.IGNORECASE)
_WINDOW_RE  = re.compile(r"\b(\d+)[- ]?day\b", re.IGNORECASE)
_K_RE       = re.compile(r"\b(\d+)\s*cluster", re.IGNORECASE)
_METHOD_RE  = re.compile(r"\b(arima|prophet|lstm|pearson|spearman|kendall|zscore|z.score|isolation.forest)\b", re.IGNORECASE)

_METHOD_MAP = {
    "z.score": "zscore",
    "z-score": "zscore",
    "isolation forest": "isolation_forest",
    "isolation_forest": "isolation_forest",
}


def _regex_extract(query: str) -> dict:
    """Fast regex-based param extraction, no LLM required."""
    params: dict = {}
    q = query.lower()

    m = _HORIZON_RE.search(q)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit.startswith("week"):  n *= 7
        elif unit.startswith("month"): n *= 30
        params["horizon"] = n

    m = _WINDOW_RE.search(q)
    if m:
        params["window"] = int(m.group(1))

    m = _K_RE.search(q)
    if m:
        params["n_clusters"] = int(m.group(1))

    m = _METHOD_RE.search(q)
    if m:
        raw = m.group(1).lower().replace(" ", "_")
        params["method"] = _METHOD_MAP.get(raw, raw)

    return params


# ── Column resolver ───────────────────────────────────────────────────────────

async def resolve_target_column(
    query: str,
    value_cols: List[str],
    model: str | None = None,
) -> str:
    """
    Pick the most relevant value column from `value_cols` based on the query.

    Strategy (fastest wins):
      1. Single column  → skip resolution, return immediately
      2. Exact match    → column name appears literally in query
      3. Fuzzy match    → difflib cutoff ≥ 0.6
      4. LLM semantic   → ask the fast planner model
      5. Default        → value_cols[0]
    """
    if not value_cols:
        return "value"
    if len(value_cols) == 1:
        return value_cols[0]

    q_lower = query.lower()

    # ── 1. Exact substring match (case-insensitive) ───────────────────────
    for col in value_cols:
        if col.lower() in q_lower:
            logger.info(f"Column resolver: exact match → '{col}'")
            return col

    # ── 2. Fuzzy match via difflib ────────────────────────────────────────
    # Split query into words and check each against column names
    words = re.findall(r"[a-z_]+", q_lower)
    best_col, best_score = None, 0.0
    for word in words:
        matches = difflib.get_close_matches(word, [c.lower() for c in value_cols], n=1, cutoff=0.6)
        if matches:
            score = difflib.SequenceMatcher(None, word, matches[0]).ratio()
            if score > best_score:
                best_score = score
                # Map back to original-case column name
                best_col = value_cols[[c.lower() for c in value_cols].index(matches[0])]

    if best_col is not None:
        logger.info(f"Column resolver: fuzzy match → '{best_col}' (score={best_score:.2f})")
        return best_col

    # ── 3. LLM semantic pick ──────────────────────────────────────────────
    _model = model or settings.PLANNER_MODEL
    cols_str = ", ".join(value_cols)
    prompt = (
        f"A dataset has these numeric columns: {cols_str}.\n"
        f"The user asked: \"{query}\"\n"
        f"Which single column is the user most likely asking about?\n"
        f"Reply with ONLY the column name, exactly as written above. "
        f"If unclear, reply with: {value_cols[0]}\n"
        f"Column:"
    )
    try:
        raw = await asyncio.wait_for(
            ollama_client.generate(_model, prompt, use_cache=True),
            timeout=8,
        )
        raw = raw.strip().strip('"').strip("'")
        # Validate the LLM didn't hallucinate a column name
        for col in value_cols:
            if col.lower() == raw.lower():
                logger.info(f"Column resolver: LLM semantic → '{col}'")
                return col
    except Exception as e:
        logger.debug(f"Column resolver LLM error: {e}")

    # ── 4. Default ────────────────────────────────────────────────────────
    logger.info(f"Column resolver: defaulting to '{value_cols[0]}'")
    return value_cols[0]


class QueryInterpreterAgent:
    """
    Extracts tool params from a natural language query.

    Strategy:
    1. Fast regex pass (always runs)
    2. LLM refinement (only when regex finds nothing meaningful)
    3. Merge: LLM result wins over regex for overlapping keys
    """

    def __init__(self):
        self.model = settings.PLANNER_MODEL

    async def extract_params(self, query: str, intent: str) -> dict:
        """Return a dict of params to override the plan defaults."""
        regex_params = _regex_extract(query)

        if len(regex_params) >= 1:
            logger.debug(f"Interpreter regex hit: {regex_params}")
            return regex_params

        prompt = (
            f"Extract analysis parameters from this query as compact JSON.\n"
            f"Intent: {intent}\n"
            f"Query: \"{query}\"\n\n"
            f"Possible keys (only include if clearly mentioned):\n"
            f"  horizon (int, days), window (int, days), n_clusters (int),\n"
            f"  method (string: prophet|arima|lstm|pearson|spearman|isolation_forest|zscore),\n"
            f"  period (int, seasonality days)\n\n"
            f"Return ONLY a JSON object, e.g. {{\"horizon\": 30}} or {{}} if nothing found.\n"
            f"JSON:"
        )
        try:
            raw = await asyncio.wait_for(
                ollama_client.generate(self.model, prompt),
                timeout=10,
            )
            raw = raw.strip()
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > start:
                llm_params = json.loads(raw[start:end])
                if isinstance(llm_params, dict):
                    merged = {**regex_params, **llm_params}
                    logger.debug(f"Interpreter LLM+regex merged: {merged}")
                    return merged
        except asyncio.TimeoutError:
            logger.debug("QueryInterpreter LLM timed out — using regex only")
        except Exception as e:
            logger.debug(f"QueryInterpreter LLM error: {e} — using regex only")

        return regex_params


query_interpreter = QueryInterpreterAgent()
