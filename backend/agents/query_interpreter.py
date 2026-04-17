"""
QueryInterpreterAgent — extracts structured params from free-form NL queries.

Uses the fast planner model to return a JSON dict of recognised params.
Falls back silently to an empty dict if the LLM is unavailable or returns garbage
so the pipeline always uses safe hardcoded defaults from _PLANS.
"""
import asyncio
import json
import re
import logging
from llm.ollama_client import ollama_client
from config import settings

logger = logging.getLogger(__name__)

# Regexes used as a fast, zero-LLM fallback
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

    # Horizon: "next 30 days", "next 2 weeks"
    m = _HORIZON_RE.search(q)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower()
        if unit.startswith("week"):
            n *= 7
        elif unit.startswith("month"):
            n *= 30
        params["horizon"] = n

    # Rolling window: "14-day rolling"
    m = _WINDOW_RE.search(q)
    if m:
        params["window"] = int(m.group(1))

    # Cluster count: "3 clusters"
    m = _K_RE.search(q)
    if m:
        params["n_clusters"] = int(m.group(1))

    # Method name
    m = _METHOD_RE.search(q)
    if m:
        raw = m.group(1).lower().replace(" ", "_")
        params["method"] = _METHOD_MAP.get(raw, raw)

    return params


class QueryInterpreterAgent:
    """
    Extracts tool params from a natural language query.

    Strategy:
    1. Fast regex pass (always runs)
    2. LLM refinement (only when regex finds nothing meaningful)
    3. Merge: LLM result wins over regex for overlapping keys
    """

    def __init__(self):
        self.model = settings.PLANNER_MODEL  # reuse fast model

    async def extract_params(self, query: str, intent: str) -> dict:
        """Return a dict of params to override the plan defaults."""
        regex_params = _regex_extract(query)

        # Only call the LLM if regex didn't capture everything we expect
        if len(regex_params) >= 1:
            logger.debug(f"Interpreter regex hit: {regex_params}")
            return regex_params

        # LLM fallback
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
                    # Regex result is already validated — let LLM fill missing keys
                    merged = {**regex_params, **llm_params}
                    logger.debug(f"Interpreter LLM+regex merged: {merged}")
                    return merged
        except asyncio.TimeoutError:
            logger.debug("QueryInterpreter LLM timed out — using regex only")
        except Exception as e:
            logger.debug(f"QueryInterpreter LLM error: {e} — using regex only")

        return regex_params


query_interpreter = QueryInterpreterAgent()
