"""
app/main.py
───────────
FastAPI application entry point.

Lifespan:
  startup  → init_db() (pgvector ext + our tables + indexes)
             + PGVector.acreate_collection() (LangChain collection)
  shutdown → dispose engine

Endpoints registered:
  GET  /health            — liveness check
  POST /documents/ingest  — upload + ingest a PDF
  GET  /documents         — list ingested documents
  DELETE /documents/{id}  — remove a document
  POST /query             — compliance question → grounded answer
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

# ── Windows Event Loop Policy & Uvicorn Loop Setup ────────────────────────────
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        import uvicorn.loops.asyncio
        uvicorn.loops.asyncio.asyncio_loop_factory = lambda use_subprocess=False: asyncio.SelectorEventLoop
        import uvicorn.loops.auto
        uvicorn.loops.auto.auto_loop_factory = lambda use_subprocess=False: asyncio.SelectorEventLoop
    except Exception:
        pass

import sqlalchemy as sa
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_postgres.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings
from app.db import engine, init_db
from app.models import HealthResponse
from app.routers.documents import router as documents_router
from app.routers.query import router as query_router

# ── Logging ───────────────────────────────────────────────────────────────────
settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting ERC Analyzer...")

    # 1. Bootstrap our schema (pgvector ext + documents table)
    await init_db()

    # 2. Ensure LangChain's pgvector collection + tables exist
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            output_dimensionality=settings.embedding_dimension,
            google_api_key=settings.google_api_key,
        )
        vs = PGVector(
            collection_name=settings.pgvector_collection,
            connection=engine,
            embeddings=embeddings,
            use_jsonb=True,
            async_mode=True,
        )
        # langchain_postgres: acreate_tables_if_not_exists creates tables if needed
        try:
            await vs.acreate_tables_if_not_exists()
        except Exception as table_err:
            logger.debug("Table check: %s", table_err)

        logger.info("PGVector collection '%s' ready", settings.pgvector_collection)

        # Re-run index creation now that the table definitely exists
        await init_db()

    except Exception as exc:
        logger.warning("PGVector collection init warning (may be non-fatal): %s", exc)

    logger.info("ERC Analyzer ready — http://localhost:8000")
    yield

    logger.info("Shutting down ERC Analyzer...")
    await engine.dispose()


# ── Application ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Regulatory Compliance RAG Engine",
    description=(
        "Upload regulatory PDFs and ask compliance questions. "
        "Answers are strictly grounded in uploaded documents with "
        "page-level citations and deterministic anti-hallucination guards."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Frontend ───────────────────────────────────────────────────────────
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(documents_router)
app.include_router(query_router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Liveness check",
)
async def health_check() -> HealthResponse:
    db_status = "connected"
    try:
        async with engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
    )


# ── Frontend & Static Serving ──────────────────────────────────────────────────
_INDEX_FILE = _FRONTEND_DIR / "index.html"
_ASSETS_DIR = _FRONTEND_DIR / "assets"

if _ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")

@app.get("/", include_in_schema=False)
async def serve_index():
    """Explicit root route to ensure the frontend is always served reliably."""
    if _INDEX_FILE.exists():
        return FileResponse(_INDEX_FILE)
    return HTMLResponse("<h1>ERC Analyzer Frontend Loading...</h1>")

# Catch-all for SPA routes / other static files
app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")

