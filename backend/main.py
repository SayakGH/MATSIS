from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import upload, query, history, datasets
from db.connections import db_manager
from config import settings
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    await db_manager.connect()
    yield
    await db_manager.disconnect()


app = FastAPI(title=settings.PROJECT_NAME, version=settings.APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(query.router)
app.include_router(history.router)
app.include_router(datasets.router)
app.include_router(query.ws_router)


@app.get("/")
async def root():
    return {"message": "MATSIS API is running", "version": settings.APP_VERSION}
