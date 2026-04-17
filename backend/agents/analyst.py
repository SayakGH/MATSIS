import json
from llm.ollama_client import ollama_client
from config import settings
from models.schemas import ToolResult
import logging

logger = logging.getLogger(__name__)

class AnalystAgent:
    def __init__(self):
        self.model = settings.ANALYST_MODEL

    async def analyze(self, query: str, tool_result: ToolResult) -> dict:
        prompt = f"""
You are a data analyst. Given the results of a time series analysis task "{tool_result.task}", 
interpret the findings for the user query: "{query}".

Tool Results:
- Metrics: {tool_result.metrics}
- Sample raw output: {str(tool_result.raw_output)[:500]}

Identify key findings, probable causes, and your confidence.
OUTPUT ONLY VALID JSON:
{{
    "key_findings": ["finding 1", "finding 2"],
    "probable_causes": ["cause 1"],
    "confidence": 0.85
}}

JSON:
"""
        try:
            response_text = await ollama_client.generate(self.model, prompt)
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])
            return {"key_findings": ["Analysis completed"], "probable_causes": ["N/A"], "confidence": 0.5}
        except Exception as e:
            logger.error(f"Analyst error: {e}")
            return {"error": str(e), "key_findings": [], "probable_causes": [], "confidence": 0.0}

analyst_agent = AnalystAgent()
