from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "MATSIS"
    APP_VERSION: str = "1.0.0"

    # MongoDB — replace with your Atlas connection string if needed
    MONGO_URL: str = "mongodb://localhost:27017"
    DB_NAME: str = "matsis"

    # Ollama (local)
    OLLAMA_URL: str = "http://localhost:11434"
    PLANNER_MODEL: str = "phi3"
    ANALYST_MODEL: str = "mistral"
    EXPLAINER_MODEL: str = "llama3"

    # Storage
    DATA_DIR: str = "./data"

    class Config:
        env_file = ".env"

settings = Settings()
