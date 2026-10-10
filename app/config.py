"""
app/config.py
─────────────
Application configuration loaded from environment variables via pydantic-settings.
All pipeline tuning knobs live here — never hardcoded deeper in the stack.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Gemini ────────────────────────────────────────────────────────
    google_api_key: str

    # ── PostgreSQL ───────────────────────────────────────────────────────────
    database_url: str | None = None
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "erc_analyzer"
    postgres_user: str = "postgres"
    postgres_password: str = ""

    # ── Application ──────────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── RAG Pipeline ─────────────────────────────────────────────────────────
    chunk_size: int = 1200
    chunk_overlap: int = 200
    dense_top_k: int = 15
    sparse_top_k: int = 15
    rerank_top_n: int = 6
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cosine_floor: float = 0.50
    cross_encoder_floor: float = 0.0
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimension: int = 768
    llm_primary: str = "gemini-3.8-flash"
    llm_fallback_1: str = "gemini-3.5-flash"
    llm_fallback_2: str = "gemini-flash-latest"
    pgvector_collection: str = "erc_compliance_chunks"

    @property
    def postgres_dsn(self) -> str:
        """psycopg v3 DSN used by langchain-postgres PGVector."""
        if self.database_url:
            url = self.database_url
            if url.startswith("postgres://"):
                url = url.replace("postgres://", "postgresql+psycopg://", 1)
            elif url.startswith("postgresql://") and not url.startswith("postgresql+psycopg://"):
                url = url.replace("postgresql://", "postgresql+psycopg://", 1)
            return url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_dsn_async(self) -> str:
        """Async psycopg v3 DSN."""
        return self.postgres_dsn


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
