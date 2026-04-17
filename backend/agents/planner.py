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
    "correlation": [
        PlanStep(agent="tool", task="correlation", params={"method": "pearson"}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "rolling_stats": [
        PlanStep(agent="tool", task="rolling_stats", params={"window": 7}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "peak_detection": [
        PlanStep(agent="tool", task="peak_detection", params={"order": 3}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "regression": [
        PlanStep(agent="tool", task="regression", params={}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "clustering": [
        PlanStep(agent="tool", task="clustering", params={"n_clusters": 3}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    # Statistical tests
    "outlier_detection": [
        PlanStep(agent="tool", task="outlier_detection", params={"method": "iqr"}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "stationarity_test": [
        PlanStep(agent="tool", task="stationarity_test", params={}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "statistical_correlation": [
        PlanStep(agent="tool", task="statistical_correlation", params={"method": "pearson"}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "one_sample_ttest": [
        PlanStep(agent="tool", task="one_sample_ttest", params={"pop_mean": 0}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "two_sample_ttest": [
        PlanStep(agent="tool", task="two_sample_ttest", params={"equal_var": False}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "runs_test": [
        PlanStep(agent="tool", task="runs_test", params={}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
    "entropy": [
        PlanStep(agent="tool", task="entropy", params={"bins": 10}),
        PlanStep(agent="analyst", task="interpret", params={}),
        PlanStep(agent="explainer", task="explain", params={}),
    ],
}

_INTENT_KEYWORDS = {
    "forecast": ["forecast", "predict", "future", "next", "upcoming", "trend"],
    "anomaly_detection": ["anomal", "outlier", "unusual", "spike", "detect", "abnormal"],
    "decomposition": ["decompos", "seasonal", "trend", "component", "period"],
    "summary": ["summar", "describe", "max", "min", "mean", "average", "statistic", "about", "what"],
    "correlation": ["correlat", "relationship", "related", "autocorr", "lag"],
    "rolling_stats": ["rolling", "moving", "sliding", "window", "7-day", "weekly average"],
    "peak_detection": ["peak", "valley", "trough", "maxima", "minima", "local max", "local min"],
    "regression": ["regression", "linear", "trend line", "slope", "r-squared", "r2", "fit"],
    "clustering": ["cluster", "segment", "group", "k-means", "kmeans", "dbscan"],
    # Statistical tests
    "outlier_detection": ["outlier", "anomaly", "spike", "extreme", "deviation"],
    "stationarity_test": ["stationar", "trend stationary", "diffuse", "random walk"],
    "statistical_correlation": ["correlat", "relationship", " association", "dependency"],
    "one_sample_ttest": ["t-test", "one sample", "population mean", "compare to"],
    "two_sample_ttest": ["two sample", "independent", "compare two", "group a", "group b"],
    "runs_test": ["runs", "randomness", "random", "pattern", "series"],
    "entropy": ["entropy", "randomness", "uncertainty", "information"],
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
        valid_intents = ", ".join(_PLANS.keys())
        prompt = (
            f"Classify the following data analysis query into exactly ONE of these intents: "
            f"{valid_intents}.\n\n"
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
