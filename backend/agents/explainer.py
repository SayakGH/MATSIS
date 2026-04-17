from llm.ollama_client import ollama_client
from config import settings
import logging

logger = logging.getLogger(__name__)

class ExplainerAgent:
    def __init__(self):
        self.model = settings.EXPLAINER_MODEL

    async def explain_stream(self, query: str, analysis: dict):
        prompt = f"""
You are a friendly data expert. Convert this structured analysis into a clear,
concise explanation for a non-technical user. Use plain language. 2-3 paragraphs max.
Focus on answering the user's original question: "{query}"

Analysis Results:
{analysis}

Explanation:
"""
        try:
            async for token in ollama_client.generate_stream(self.model, prompt):
                yield token
        except Exception as e:
            logger.error(f"Explainer error: {e}")
            yield f"I processed your request but encountered an error: {e}"

explainer_agent = ExplainerAgent()
