"""
app/models.py
─────────────
Pydantic request / response schemas for the FastAPI layer.
These are *not* ORM models — they are the wire format only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ── Ingestion ─────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    """Returned by POST /documents/ingest."""

    status: Literal["INGESTED", "ALREADY_INGESTED"]
    document_id: str = Field(description="UUID of the document record")
    file_name: str
    page_count: int | None = None
    chunk_count: int | None = Field(
        default=None,
        description="Number of chunks written to pgvector (None if already ingested)",
    )
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body for POST /query."""

    query: str = Field(
        min_length=3,
        max_length=2000,
        description="The compliance question to answer",
    )
    document_filter: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of file names to restrict retrieval to. "
            "If omitted, all documents are searched."
        ),
    )


class CitationItem(BaseModel):
    """One grounded citation attached to the answer."""

    chunk_id: str
    document_name: str
    page_number: int
    similarity_score: float | None = Field(
        default=None,
        description="Cosine similarity from dense retrieval (0–1)",
    )
    excerpt: str = Field(description="First 300 chars of the chunk text")


class QueryResponse(BaseModel):
    """Returned by POST /query."""

    answer: str
    citations: list[CitationItem]
    citations_verified: bool = Field(
        description=(
            "True when every [Document: X, Page: Y] citation in the answer "
            "was present in the retrieved context."
        )
    )
    query: str


# ── Document Registry ─────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    """Returned by GET /documents."""

    document_id: str
    file_name: str
    page_count: int | None
    status: str
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["connected", "error"]
    version: str = "1.0.0"
