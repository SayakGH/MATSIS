import asyncio
import json
import logging
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from llm.ollama_client import ollama_client
from config import settings
from models.schemas import ToolResult
from agents.memory import memory_agent

logger = logging.getLogger(__name__)


class AnalysisOutput(BaseModel):
    """Strict schema for analyst LLM output."""
    key_findings: List[str] = []
    probable_causes: List[str] = []
    confidence: float = 0.5
    trend_direction: Optional[str] = None          # "upward" | "downward" | "flat"
    actionable_insight: Optional[str] = None       # single plain-English recommendation
    follow_up_suggestions: List[str] = []          # shown as chips in the frontend


_FALLBACK = AnalysisOutput(
    key_findings=["Analysis completed"],
    probable_causes=["N/A"],
    confidence=0.5,
)


class AnalystAgent:
    def __init__(self):
        self.model = settings.ANALYST_MODEL

    async def analyze(
        self, query: str, tool_result: ToolResult, session_id: str | None = None
    ) -> dict:
        # Fetch conversation context (non-blocking; empty string if no history)
        context = ""
        if session_id:
            try:
                context = await memory_agent.get_context_prompt(session_id)
            except Exception:
                pass  # context is a bonus — never block on it

        context_block = f"\n{context}\n" if context else ""

        metrics_str = json.dumps(tool_result.metrics, indent=2)
        sample_str = str(tool_result.raw_output)[:600]

        prompt = f"""\
You are a senior data analyst.{context_block}
Task performed: "{tool_result.task}"
User question: "{query}"

Results:
Metrics:
{metrics_str}

Sample output (truncated):
{sample_str}

Identify key findings, probable causes, trend direction, and one actionable insight.
Suggest 2-3 short follow-up questions the user could ask next.

OUTPUT ONLY VALID JSON matching this schema exactly:
{{
    "key_findings": ["finding 1", "finding 2"],
    "probable_causes": ["cause 1"],
    "confidence": 0.85,
    "trend_direction": "upward",
    "actionable_insight": "Consider investigating...",
    "follow_up_suggestions": ["Show anomalies only", "Forecast next 14 days"]
}}

JSON:"""

        for attempt in range(2):  # retry once on bad JSON
            try:
                response_text = await ollama_client.generate(self.model, prompt)
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    parsed = json.loads(response_text[start:end])
                    validated = AnalysisOutput(**parsed)
                    return validated.model_dump()
                logger.warning(f"Analyst: no JSON found in response (attempt {attempt + 1})")
            except (json.JSONDecodeError, ValidationError) as e:
                logger.warning(f"Analyst: bad JSON on attempt {attempt + 1}: {e}")
            except Exception as e:
                logger.error(f"Analyst error: {e}")
                break

        return _FALLBACK.model_dump()


analyst_agent = AnalystAgent()
