from db.connections import db_manager
from models.schemas import QueryRecord
from datetime import datetime, timedelta

class MemoryAgent:
    COLLECTION = "sessions"

    async def store_query(self, session_id: str, record: QueryRecord):
        await db_manager.db[self.COLLECTION].update_one(
            {"session_id": session_id},
            {
                "$push": {"entries": record.model_dump(mode="json")},
                "$setOnInsert": {
                    "created_at": datetime.utcnow(),
                    "expires_at": datetime.utcnow() + timedelta(days=7),
                },
            },
            upsert=True,
        )

    async def get_history(self, session_id: str, limit: int = 10):
        session = await db_manager.db[self.COLLECTION].find_one({"session_id": session_id})
        if session:
            return session.get("entries", [])[-limit:]
        return []

    async def get_context_prompt(self, session_id: str) -> str:
        history = await self.get_history(session_id, limit=3)
        if not history:
            return ""
        ctx = "Previous conversation:\n"
        for h in history:
            ctx += f"User: {h['raw_query']}\nAssistant: {(h.get('explanation') or '')[:200]}...\n"
        return ctx

memory_agent = MemoryAgent()
