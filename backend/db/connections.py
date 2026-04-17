from motor.motor_asyncio import AsyncIOMotorClient
from config import settings

class DBManager:
    def __init__(self):
        self.mongo_client = None
        self.db = None

    async def connect(self):
        self.mongo_client = AsyncIOMotorClient(settings.MONGO_URL)
        self.db = self.mongo_client[settings.DB_NAME]
        # Ping to verify connection is live
        await self.db.command("ping")
        print(f"✅ Connected to MongoDB at {settings.MONGO_URL}")

    async def disconnect(self):
        if self.mongo_client:
            self.mongo_client.close()

db_manager = DBManager()
