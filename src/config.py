from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    MISTRAL_API_KEY: str
    GOOGLE_API_KEY: str
    GOOGLE_CSE_ID: str  # Custom Search Engine ID
    MODEL_NAME: str = "mistral-medium"
    TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2000

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings() 