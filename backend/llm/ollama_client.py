import json
import httpx
from config import settings


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_URL

    async def generate(self, model: str, prompt: str) -> str:
        """Non-streaming: returns the full response string."""
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                data = response.json()
                if "error" in data:
                    print(f"⚠️ Ollama error for model '{model}': {data['error']}")
                    return ""
                return data.get("response", "")
            except Exception as e:
                print(f"❌ Ollama generate error (model={model}): {e}")
                return ""

    async def generate_stream(self, model: str, prompt: str):
        """Streaming: yields text tokens one by one."""
        async with httpx.AsyncClient(timeout=120) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": True},
                ) as response:
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if "error" in chunk:
                            print(f"⚠️ Ollama stream error (model={model}): {chunk['error']}")
                            yield f"[Model error: {chunk['error']}]"
                            return
                        if not chunk.get("done"):
                            token = chunk.get("response", "")
                            if token:
                                yield token
            except Exception as e:
                print(f"❌ Ollama stream error (model={model}): {e}")
                yield f"[Connection error: {e}]"

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{self.base_url}/api/tags")
                return r.status_code == 200
        except Exception:
            return False


ollama_client = OllamaClient()
