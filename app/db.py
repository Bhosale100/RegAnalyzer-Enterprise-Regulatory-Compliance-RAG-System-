"""
app/db.py
─────────
PostgreSQL wiring:
  • SQLAlchemy engine / session for our own `documents` table
  • Helper to bootstrap the pgvector extension and our schema
"""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Async Engine ─────────────────────────────────────────────────────────────────
engine = create_async_engine(
    settings.postgres_dsn_async,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=(settings.app_env == "development"),
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# ── ORM Metadata ─────────────────────────────────────────────────────────────────
metadata = sa.MetaData()

documents_table = sa.Table(
    "documents",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),          # UUID
    sa.Column("file_name", sa.Text, nullable=False),
    sa.Column("content_hash", sa.String(64), nullable=False, unique=True),  # SHA-256
    sa.Column("page_count", sa.Integer, nullable=True),
    sa.Column(
        "status",
        sa.Text,
        nullable=False,
        default="PROCESSING",
    ),  # PROCESSING | COMPLETED | FAILED
    sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    ),
    sa.Column(
        "updated_at",
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        onupdate=sa.func.now(),
        nullable=False,
    ),
)


async def init_db() -> None:
    """
    Bootstrap sequence (idempotent):
      1. Enable pgvector extension
      2. Create our `documents` table
      3. Add HNSW + GIN indexes on LangChain's embedding table (if it exists)
    """
    async with engine.begin() as conn:
        # pgvector extension — must exist before LangChain creates its table
        await conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))
        logger.info("pgvector extension ready")

        # Our documents registry (explicit DDL + metadata create_all for idempotency)
        await conn.execute(
            sa.text(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id VARCHAR(36) PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    content_hash VARCHAR(64) NOT NULL UNIQUE,
                    page_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'PROCESSING',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
                );
                """
            )
        )
        await conn.run_sync(metadata.create_all)
        logger.info("documents table ready")

        # HNSW index on LangChain's embedding table — created after LangChain
        # initialises the table (PGVector.acreate_collection is called first in
        # the lifespan).  We wrap in try/except so startup doesn't fail if the
        # table doesn't exist yet on a fresh DB.
        try:
            await conn.execute(
                sa.text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_lc_embedding_hnsw
                    ON langchain_pg_embedding
                    USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
                    """
                )
            )
            await conn.execute(
                sa.text(
                    """
                    CREATE INDEX IF NOT EXISTS idx_lc_embedding_metadata_gin
                    ON langchain_pg_embedding
                    USING gin (cmetadata)
                    """
                )
            )
            logger.info("HNSW + GIN indexes ready on langchain_pg_embedding")
        except Exception as exc:  # table may not exist on first boot
            logger.debug("Index creation skipped (table may not exist yet): %s", exc)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
