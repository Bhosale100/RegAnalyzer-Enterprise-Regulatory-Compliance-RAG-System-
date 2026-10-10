"""
app/routers/query.py
────────────────────
Endpoint:
  POST /query   — Submit a compliance question; receive a grounded answer
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from fastapi import APIRouter, HTTPException, status
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.db import AsyncSessionLocal, documents_table
from app.exceptions import CitationValidationError, InsufficientContextError
from app.models import CitationItem, QueryRequest, QueryResponse
from app.pipeline.rag_chain import _CITATION_RE, run_query

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/query", tags=["Query"])


async def _load_all_chunks() -> list[Document]:
    """
    Load all document chunks from langchain_pg_embedding directly via SQL.
    This fetches real chunks for BM25 indexing in milliseconds without calling external APIs.
    """
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                sa.text(
                    """
                    SELECT document, cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id = (
                        SELECT uuid FROM langchain_pg_collection WHERE name = :col_name LIMIT 1
                    )
                    """
                ),
                {"col_name": settings.pgvector_collection},
            )
            rows = result.fetchall()
            docs: list[Document] = []
            for doc_text, cmetadata in rows:
                meta = cmetadata if isinstance(cmetadata, dict) else {}
                docs.append(Document(page_content=doc_text or "", metadata=meta))
            return docs
    except Exception as exc:
        logger.warning("SQL load of chunks failed, falling back to empty list: %s", exc)
        return []


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a compliance question",
    description=(
        "Submit a compliance question. The engine retrieves grounded excerpts "
        "from ingested documents and returns an answer with page-level citations. "
        "Returns 404 if no relevant context is found. "
        "Returns 422 if the generated answer cannot be verified against retrieved chunks."
    ),
)
async def query_endpoint(body: QueryRequest) -> QueryResponse:
    # ── Verify there are ingested documents ───────────────────────────────────
    async with AsyncSessionLocal() as session:
        count_result = await session.execute(
            sa.select(sa.func.count()).select_from(documents_table).where(
                documents_table.c.status == "COMPLETED"
            )
        )
        doc_count = count_result.scalar_one()

    if doc_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents have been ingested yet. Upload a PDF first.",
        )

    # ── Load all chunks for BM25 ──────────────────────────────────────────────
    try:
        corpus_docs = await _load_all_chunks()
        logger.info("BM25 corpus size: %d chunks", len(corpus_docs))
    except Exception as exc:
        logger.exception("Failed to load corpus for BM25: %s", exc)
        corpus_docs = []

    # ── Run the RAG chain ──────────────────────────────────────────────────────
    try:
        result = await run_query(
            query=body.query,
            document_filter=body.document_filter,
            all_corpus_docs=corpus_docs,
        )
    except InsufficientContextError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=exc.reason,
        ) from exc
    except CitationValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "reason": exc.reason,
                "mismatches": exc.mismatches,
            },
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected RAG chain error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal error during query: {exc}",
        ) from exc

    # ── Build structured citation list ────────────────────────────────────────
    answer: str = result["answer"]
    source_docs = result["source_docs"]

    # Index retrieved chunks by (source_lower, page) for score lookup
    chunk_index: dict[tuple[str, int], dict] = {}
    for doc in source_docs:
        src = doc.metadata.get("source", "").lower()
        pg = int(doc.metadata.get("page", 0))
        chunk_index[(src, pg)] = {
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "score": doc.metadata.get("score") or doc.metadata.get("relevance_score"),
            "text": doc.page_content,
        }

    citations: list[CitationItem] = []
    seen_citations: set[tuple[str, int]] = set()

    for doc_name, page_str in _CITATION_RE.findall(answer):
        doc_name = doc_name.strip()
        page_num = int(page_str)
        key = (doc_name.lower(), page_num)
        if key in seen_citations:
            continue
        seen_citations.add(key)

        meta = chunk_index.get(key, {})
        citations.append(
            CitationItem(
                chunk_id=meta.get("chunk_id", ""),
                document_name=doc_name,
                page_number=page_num,
                similarity_score=meta.get("score"),
                excerpt=(meta.get("text", "")[:300] if meta.get("text") else ""),
            )
        )

    return QueryResponse(
        answer=answer,
        citations=citations,
        citations_verified=result.get("citations_verified", False),
        query=body.query,
    )
