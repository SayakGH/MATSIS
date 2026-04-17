from llm.ollama_client import ollama_client
from config import settings
from agents.memory import memory_agent
import logging

logger = logging.getLogger(__name__)


class ExplainerAgent:
    def __init__(self):
        self.model = settings.EXPLAINER_MODEL

    async def explain_stream(
        self, query: str, analysis: dict, session_id: str | None = None
    ):
        # Inject conversation context so the explainer can say "as we saw earlier…"
        context = ""
        if session_id:
            try:
                context = await memory_agent.get_context_prompt(session_id)
            except Exception:
                pass

        context_block = f"\n{context}" if context else ""

        # Surface any warnings added by the postprocessor
        warnings_block = ""
        if analysis.get("warnings"):
            warnings_block = "\nNote the following warnings:\n" + "\n".join(
                f"- {w}" for w in analysis["warnings"]
            )

        prompt = f"""\
You are a friendly data expert.{context_block}
Convert this structured analysis into a clear, concise explanation for a non-technical user.
Use plain language. 2–3 paragraphs max.
Focus on answering the user's original question: "{query}"{warnings_block}

Analysis:
{analysis}

Explanation:"""

        try:
            async for token in ollama_client.generate_stream(self.model, prompt):
                yield token
        except Exception as e:
            logger.error(f"Explainer error: {e}")
            yield f"I processed your request but encountered an error: {e}"


explainer_agent = ExplainerAgent()
