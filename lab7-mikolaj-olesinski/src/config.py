import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    google_api_key: str
    redis_url: str
    gemini_model: str = "gemini-2.5-flash-lite"


def get_redis_url() -> str:
    return os.getenv("REDIS_URL", "redis://localhost:6379/0").strip()


def load_config() -> Config:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Brak GOOGLE_API_KEY. Skopiuj .env.example do .env i wklej klucz z https://aistudio.google.com/apikey"
        )
    return Config(google_api_key=api_key, redis_url=get_redis_url())
