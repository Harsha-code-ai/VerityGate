
"""Application settings, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PLACEHOLDER_SECRETS = {
    "",
    "change-me",
    "change-me-to-a-long-random-string",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # --- general ---
    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("./data")
    secret_key: str = ""
    token_ttl_minutes: int = 720
    allow_registration: bool = True
    cors_origins: str = "http://localhost:8080,http://localhost:5173"
    rate_limit_per_minute: int = 30
    audit_store_questions: bool = False

    # --- upload limits ---
    max_upload_mb: int = 25
    max_pages: int = 300
    min_extracted_chars: int = 50
    max_documents_per_user: int = 100
    ocr_enabled: bool = False
    ocr_engine: str = "tesseract"
    ocr_language: str = "eng"
    ocr_dpi: int = 200
    ingest_concurrency: int = 1
    preload_models: bool = False

    # --- chunking ---
    chunk_words: int = 180
    chunk_overlap_words: int = 30

    # --- vector store ---
    qdrant_url: str = "http://qdrant:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "verigate_chunks"

    # --- embeddings / retrieval ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_query_prefix: str = ""
    embedding_document_prefix: str = ""
    retrieval_top_k: int = 6
    retrieval_candidates: int = 20
    rrf_k: int = 60
    min_dense_score: float = 0.25
    min_bm25_score: float = 1.0
    min_query_coverage: float = 0.34
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: str = ""
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""

    # --- Groq backup provider ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # --- Local LLM ---
    ollama_base_url: str = "http://ollama:11434"

    # Increased from 700 to 3000 to give reasoning models
    # more room to produce their final answer.
    llm_max_tokens: int = 3000
    llm_timeout_seconds: float = 14.0

    # --- verification ---
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_label_order: str = ""
    entailment_threshold: float = 0.80
    contradiction_threshold: float = 0.85
    conflict_threshold: float = 0.90
    verify_max_units: int = 6
    relevance_min_coverage: float = 0.50
    relevance_min_support_coverage: float = 0.30
    relevance_min_similarity: float = 0.50
    max_retries: int = Field(default=2, ge=0)

    # --- optional tracking ---
    mlflow_tracking_uri: str = ""

    @property
    def cors_origin_list(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "verigate.sqlite3"

    @property
    def effective_max_retries(self) -> int:
        return min(self.max_retries, 2)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def resolve_secret_key(self) -> str:
        """Return a usable signing key. In development a key is generated and stored once."""

        if self.secret_key not in PLACEHOLDER_SECRETS:
            if len(self.secret_key) < 32:
                raise RuntimeError(
                    "SECRET_KEY must be at least 32 characters long."
                )
            return self.secret_key

        if self.app_env == "production":
            raise RuntimeError(
                "SECRET_KEY is required when APP_ENV=production."
            )

        self.ensure_dirs()

        key_file = self.data_dir / ".dev_secret_key"

        if not key_file.exists():
            key_file.write_text(secrets.token_urlsafe(48))
            key_file.chmod(0o600)

        return key_file.read_text().strip()


@lru_cache
def get_settings() -> Settings:
    return Settings()
