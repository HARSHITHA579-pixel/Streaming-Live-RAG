"""
Configuration Settings for Streaming RAG

================================================================================
RESPONSIBILITY:
- Centralize all environment variables, model identifiers, API keys, indexing,
  retrieval hyperparameters, and runtime thresholds.
- Provide strongly typed configuration objects with default fallbacks and validation.

INPUTS:
- Operating system environment variables and `.env` file values.

OUTPUTS:
- Instantiated `Settings` object used across the application.

CONNECTED COMPONENTS:
- `corpus/build_index.py`: Uses chunk sizes, overlap, embedding models, and data paths.
- `app.controller`: Uses T0 stability thresholds and T1 LLM model parameters.
- `app.retriever`: Uses top-k per query, minimum evidence per intent, RRF-k, and reranker settings.
- `app.verifier`: Uses grounding thresholds and verification model parameters.
- `app.telemetry`: Uses log level and log directory configurations.
- `app.main`: Uses server host, port, and session management settings.

WHY THIS ARCHITECTURE:
- Prevents hardcoding of sensitive credentials and allows rapid experimentation with
  different retrieval parameters (e.g. RRF constant k, top-K evidence thresholds)
  without modifying business logic.
================================================================================
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Settings:
    """
    Centralized application configuration with environment variable loading and sensible defaults.
    """

    # --- LLM API Credentials & Models ---
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY", "")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gemini-1.5-pro")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )

    # --- Embeddings & Reranker Models ---
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    )
    embedding_dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768"))
    )
    reranker_model: str = field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "rerank-english-v3.0")
    )

    # --- Controller Hyperparameters (T0 & T1) ---
    t0_similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("T0_SIMILARITY_THRESHOLD", "0.85"))
    )
    t0_min_chunk_tokens: int = field(
        default_factory=lambda: int(os.getenv("T0_MIN_CHUNK_TOKENS", "5"))
    )

    # --- Retrieval & Fusion Hyperparameters ---
    retrieval_top_k_per_query: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K_PER_QUERY", "4"))
    )
    min_evidence_per_intent: int = field(
        default_factory=lambda: int(os.getenv("MIN_EVIDENCE_PER_INTENT", "2"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "60"))
    )

    # --- Corpus & Indexing ---
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64"))
    )
    corpus_dir: Path = field(
        default_factory=lambda: Path(os.getenv("CORPUS_DIR", "corpus/raw_docs"))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data"))
    )

    # --- Server & Telemetry ---
    app_host: str = field(
        default_factory=lambda: os.getenv("APP_HOST", "0.0.0.0")
    )
    app_port: int = field(
        default_factory=lambda: int(os.getenv("APP_PORT", "8000"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    telemetry_log_dir: Path = field(
        default_factory=lambda: Path(os.getenv("TELEMETRY_LOG_DIR", "logs"))
    )


# Singleton instance placeholder for global configuration access
# TODO: Initialize properly upon application startup or import
settings = Settings()
