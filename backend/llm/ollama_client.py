import asyncio
import hashlib
import json
import httpx
from config import settings

# ── LRU response cache (planner + analyst non-streaming calls only) ────────────
_MAX_CACHE = 128
_response_cache: dict[str, str] = {}   # key → response text
_cache_order: list[str] = []           # insertion order for LRU eviction


def _cache_get(key: str) -> str | None:
    return _response_cache.get(key)


def _cache_set(key: str, value: str) -> None:
    if key in _response_cache:
        _cache_order.remove(key)
    elif len(_cache_order) >= _MAX_CACHE:
        oldest = _cache_order.pop(0)
        _response_cache.pop(oldest, None)
    _response_cache[key] = value
    _cache_order.append(key)


def _cache_key(model: str, prompt: str) -> str:
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()


def clear_cache() -> None:
    """Clears the entire response cache (useful for testing)."""
    _response_cache.clear()
    _cache_order.clear()


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_URL

    async def generate(self, model: str, prompt: str, use_cache: bool = True) -> str:
        """Non-streaming: returns the full response string.

        Results are cached by (model, prompt) hash. Pass use_cache=False to
        force a fresh call (e.g., from the explainer which always streams).
        """
        if use_cache:
            key = _cache_key(model, prompt)
            cached = _cache_get(key)
            if cached is not None:
                return cached

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
                result = data.get("response", "")
                if use_cache and result:
                    _cache_set(_cache_key(model, prompt), result)
                return result
            except Exception as e:
                print(f"❌ Ollama generate error (model={model}): {e}")
                return ""

    async def generate_stream(self, model: str, prompt: str):
        """Streaming: yields text tokens one by one. Never cached."""
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
