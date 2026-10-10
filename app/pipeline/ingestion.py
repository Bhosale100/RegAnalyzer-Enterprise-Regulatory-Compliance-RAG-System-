"""
app/pipeline/ingestion.py
─────────────────────────
Stage 1 — Ingestion Pipeline

Flow:
  1. SHA-256 deduplication  → short-circuit if already stored
  2. PDF parse              → PyPDFLoader (one Document per page)
  3. Metadata normalisation → source, page (1-indexed), document_id
  4. Token-aware chunking   → RecursiveCharacterTextSplitter (tiktoken, 1200/200)
  5. Chunk ID assignment    → UUID per chunk in metadata["chunk_id"]
  6. Embed + store          → PGVector.aadd_documents()
  7. Mark COMPLETED         → UPDATE documents SET status = 'COMPLETED'
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from pathlib import Path

import sqlalchemy as sa
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.db import AsyncSessionLocal, documents_table, engine, init_db
from app.exceptions import AlreadyIngestedError, IngestionError

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        output_dimensionality=settings.embedding_dimension,
        google_api_key=settings.google_api_key,
    )


def _make_vector_store(embeddings: GoogleGenerativeAIEmbeddings) -> PGVector:
    return PGVector(
        collection_name=settings.pgvector_collection,
        connection=engine,
        embeddings=embeddings,
        use_jsonb=True,
        async_mode=True,
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

async def ingest_document(
    file_bytes: bytes,
    file_name: str,
    tmp_path: Path,
) -> dict:
    """
    Ingest a PDF into pgvector.

    Args:
        file_bytes:  Raw bytes of the uploaded PDF.
        file_name:   Original filename (used as `source` in chunk metadata).
        tmp_path:    Absolute path to a writable temp file that already contains
                     `file_bytes` (required by PyPDFLoader which reads from disk).

    Returns:
        dict with keys: status, document_id, file_name, page_count, chunk_count
    """
    # ── Step 1: SHA-256 deduplication ────────────────────────────────────────
    content_hash = _sha256(file_bytes)
    logger.info("Ingesting '%s' | hash=%s", file_name, content_hash[:16])

    try:
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                sa.select(documents_table).where(
                    documents_table.c.content_hash == content_hash
                )
            )
            row = existing.mappings().first()
    except Exception:
        # Auto-heal: Ensure table exists
        await init_db()
        async with AsyncSessionLocal() as session:
            existing = await session.execute(
                sa.select(documents_table).where(
                    documents_table.c.content_hash == content_hash
                )
            )
            row = existing.mappings().first()
        if row and row["status"] == "COMPLETED":
            raise AlreadyIngestedError(
                document_id=row["id"],
                file_name=row["file_name"],
            )

        if row and row["status"] != "COMPLETED":
            # Re-use existing record ID if previous ingestion was aborted or failed
            doc_id = row["id"]
            await session.execute(
                documents_table.update()
                .where(documents_table.c.id == doc_id)
                .values(status="PROCESSING", file_name=file_name)
            )
            await session.commit()
        else:
            # ── Step 2: Register document as PROCESSING ───────────────────────
            doc_id = str(uuid.uuid4())
            await session.execute(
                documents_table.insert().values(
                    id=doc_id,
                    file_name=file_name,
                    content_hash=content_hash,
                    status="PROCESSING",
                )
            )
            await session.commit()

    try:
        # ── Step 3: Parse PDF ────────────────────────────────────────────────
        loader = PyPDFLoader(str(tmp_path))
        pages = loader.load()  # one Document per page
        if not pages:
            raise IngestionError(file_name, "PDF produced zero pages")

        page_count = len(pages)
        logger.info("Parsed %d pages from '%s'", page_count, file_name)

        # ── Step 4: Normalize metadata ───────────────────────────────────────
        for i, doc in enumerate(pages):
            doc.metadata["source"] = file_name
            doc.metadata["page"] = i + 1          # 1-indexed
            doc.metadata["document_id"] = doc_id

        # ── Step 5: Token-aware chunking ─────────────────────────────────────
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(pages)
        logger.info("Split into %d chunks", len(chunks))

        # ── Step 6: Assign chunk IDs ─────────────────────────────────────────
        chunk_ids: list[str] = []
        for idx, chunk in enumerate(chunks):
            cid = str(uuid.uuid4())
            chunk.metadata["chunk_id"] = cid
            chunk.metadata["chunk_index"] = idx
            chunk_ids.append(cid)

        # ── Step 7: Embed + store ─────────────────────────────────────────────
        embeddings = _make_embeddings()
        vector_store = _make_vector_store(embeddings)

        # PGVector.aadd_documents writes chunks + vectors + metadata
        await vector_store.aadd_documents(chunks, ids=chunk_ids)
        logger.info("Stored %d chunks in pgvector", len(chunks))

        # ── Step 8: Mark COMPLETED ────────────────────────────────────────────
        async with AsyncSessionLocal() as session:
            await session.execute(
                documents_table.update()
                .where(documents_table.c.id == doc_id)
                .values(status="COMPLETED", page_count=page_count)
            )
            await session.commit()

        logger.info("Ingestion COMPLETED for '%s' (doc_id=%s)", file_name, doc_id)
        return {
            "status": "INGESTED",
            "document_id": doc_id,
            "file_name": file_name,
            "page_count": page_count,
            "chunk_count": len(chunks),
        }

    except IngestionError:
        raise
    except Exception as exc:
        # Mark document as FAILED
        async with AsyncSessionLocal() as session:
            await session.execute(
                documents_table.update()
                .where(documents_table.c.id == doc_id)
                .values(status="FAILED")
            )
            await session.commit()
        logger.exception("Ingestion FAILED for '%s'", file_name)
        raise IngestionError(file_name, str(exc)) from exc
