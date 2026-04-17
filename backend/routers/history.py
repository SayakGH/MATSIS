from fastapi import APIRouter
from agents.memory import memory_agent

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("")
async def get_history(session_id: str, limit: int = 20):
    return await memory_agent.get_history(session_id, limit)
