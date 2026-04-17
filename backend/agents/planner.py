import asyncio
from llm.ollama_client import ollama_client
from config import settings
from models.schemas import PlanSchema, DatasetMeta, PlanStep
import logging

logger = logging.getLogger(__name__)

# ── Hardcoded plan templates — never trust the LLM to invent agent names ──────
_PLANS = {
    "forecast": [
        PlanStep(agent="tool", task="forecast", params={"method": "prophet", "horizon": 14}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "anomaly_detection": [
        PlanStep(agent="tool", task="anomaly_detection", params={"method": "isolation_forest"}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "decomposition": [
        PlanStep(agent="tool", task="decomposition", params={"period": 7}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "summary": [
        PlanStep(agent="tool", task="summary", params={}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
}

_INTENT_KEYWORDS = {
    "forecast": ["forecast", "predict", "future", "next", "upcoming", "trend"],
    "anomaly_detection": ["anomal", "outlier", "unusual", "spike", "detect", "abnormal"],
    "decomposition": ["decompos", "seasonal", "trend", "component", "period"],
    "summary": ["summar", "describe", "max", "min", "mean", "average", "statistic", "about", "what"],
}


def _keyword_intent(query: str) -> str:
    """Fast keyword-based fallback intent detector."""
    q = query.lower()
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(k in q for k in keywords):
            return intent
    return "summary"


class PlannerAgent:
    def __init__(self):
        self.model = settings.PLANNER_MODEL

    async def plan(self, query: str, dataset_meta: DatasetMeta) -> PlanSchema:
        # Ask the LLM only for a single intent word — much more reliable with small models
        prompt = (
            "Classify the following data analysis query into exactly ONE of these intents: "
            "forecast, anomaly_detection, decomposition, summary.\n\n"
            f"Query: \"{query}\"\n\n"
            "Reply with only the intent word, nothing else:"
        )

        intent = "summary"  # safe default
        try:
            raw = await asyncio.wait_for(
                ollama_client.generate(self.model, prompt),
                timeout=20,
            )
            raw = raw.strip().lower().split()[0] if raw.strip() else ""
            if raw in _PLANS:
                intent = raw
            else:
                # keyword fallback when the model returns something unexpected
                intent = _keyword_intent(query)
                logger.info(f"Planner LLM returned '{raw}', falling back to keyword intent '{intent}'")
        except asyncio.TimeoutError:
            intent = _keyword_intent(query)
            logger.warning(f"Planner LLM timed out — using keyword intent '{intent}'")
        except Exception as e:
            intent = _keyword_intent(query)
            logger.error(f"Planner error: {e} — using keyword intent '{intent}'")

        target_col = dataset_meta.value_cols[0] if dataset_meta.value_cols else None
        return PlanSchema(
            intent=intent,
            steps=_PLANS[intent],
            target_column=target_col,
        )


planner_agent = PlannerAgent()
