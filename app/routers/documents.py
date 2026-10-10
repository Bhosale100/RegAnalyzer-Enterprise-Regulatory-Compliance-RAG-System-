"""
app/routers/documents.py
────────────────────────
Endpoints:
  POST /documents/ingest   — Upload a PDF and trigger the ingestion pipeline
  GET  /documents          — List all ingested documents
  DELETE /documents/{id}   — Remove a document and its chunks
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.db import AsyncSessionLocal, documents_table
from app.exceptions import AlreadyIngestedError, IngestionError
from app.models import DocumentInfo, DocumentListResponse, IngestResponse
from app.pipeline.ingestion import ingest_document

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/documents", tags=["Documents"])

_ALLOWED_CONTENT_TYPES = {"application/pdf", "application/octet-stream"}
_MAX_FILE_SIZE_MB = 50


# ── POST /documents/ingest ────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF",
    description=(
        "Upload a regulatory PDF. The file is deduplicated by SHA-256, "
        "parsed, chunked, embedded, and stored in PostgreSQL/pgvector. "
        "Returns ALREADY_INGESTED if an identical file was previously uploaded."
    ),
)
async def ingest_endpoint(
    file: UploadFile = File(..., description="PDF file to ingest"),
) -> IngestResponse:
    # ── Basic validation ──────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > _MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {_MAX_FILE_SIZE_MB} MB limit ({size_mb:.1f} MB).",
        )

    # ── Write to temp file (PyPDFLoader needs a path) ─────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        result = await ingest_document(
            file_bytes=file_bytes,
            file_name=file.filename,
            tmp_path=tmp_path,
        )
    except AlreadyIngestedError as exc:
        logger.info("Duplicate upload: %s", exc)
        return IngestResponse(
            status="ALREADY_INGESTED",
            document_id=exc.document_id,
            file_name=exc.file_name,
            message=f"'{exc.file_name}' was already ingested (id={exc.document_id}).",
        )
    except IngestionError as exc:
        logger.exception("Ingestion error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {exc.detail}",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(
        status="INGESTED",
        document_id=result["document_id"],
        file_name=result["file_name"],
        page_count=result["page_count"],
        chunk_count=result["chunk_count"],
        message=(
            f"Successfully ingested '{result['file_name']}' — "
            f"{result['page_count']} pages, {result['chunk_count']} chunks."
        ),
    )


# ── GET /documents ────────────────────────────────────────────────────────────

@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List all ingested documents",
)
async def list_documents() -> DocumentListResponse:
    try:
        async with AsyncSessionLocal() as session:
            rows = await session.execute(
                sa.select(documents_table).order_by(
                    documents_table.c.created_at.desc()
                )
            )
            docs = rows.mappings().all()
    except Exception as exc:
        logger.warning("Documents table query failed, attempting auto-heal init_db: %s", exc)
        from app.db import init_db
        await init_db()
        async with AsyncSessionLocal() as session:
            rows = await session.execute(
                sa.select(documents_table).order_by(
                    documents_table.c.created_at.desc()
                )
            )
            docs = rows.mappings().all()

    items = [
        DocumentInfo(
            document_id=row["id"],
            file_name=row["file_name"],
            page_count=row["page_count"],
            status=row["status"],
            created_at=row["created_at"],
        )
        for row in docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


# ── DELETE /documents/{document_id} ──────────────────────────────────────────

@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
)
async def delete_document(document_id: str) -> None:
    async with AsyncSessionLocal() as session:
        # Fetch the document to get file_name for chunk deletion
        result = await session.execute(
            sa.select(documents_table).where(
                documents_table.c.id == document_id
            )
        )
        row = result.mappings().first()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        file_name = row["file_name"]

        # Delete chunks from pgvector via LangChain
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=settings.google_api_key,
            )
            vs = PGVector(
                collection_name=settings.pgvector_collection,
                connection=settings.postgres_dsn,
                embeddings=embeddings,
                use_jsonb=True,
            )
            # PGVector doesn't have a bulk-delete-by-metadata method, so we
            # delete directly via SQL for efficiency.
            await session.execute(
                sa.text(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'source' = :source
                    """
                ),
                {"source": file_name},
            )
        except Exception as exc:
            logger.warning("Could not delete chunks for '%s': %s", file_name, exc)

        # Delete the document record
        await session.execute(
            documents_table.delete().where(documents_table.c.id == document_id)
        )
        await session.commit()

    logger.info("Deleted document '%s' (id=%s)", file_name, document_id)
