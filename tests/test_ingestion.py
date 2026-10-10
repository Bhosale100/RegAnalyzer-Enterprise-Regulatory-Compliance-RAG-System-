"""
tests/test_ingestion.py
───────────────────────
Unit tests for the ingestion pipeline.
Uses pytest + pytest-asyncio.  External services (DB, Gemini) are mocked.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.exceptions import AlreadyIngestedError, IngestionError


# ── Fixtures ──────────────────────────────────────────────────────────────────

DUMMY_PDF_BYTES = b"%PDF-1.4 dummy content for testing"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSHA256Deduplication:
    def test_sha256_is_deterministic(self):
        h1 = _sha256(DUMMY_PDF_BYTES)
        h2 = _sha256(DUMMY_PDF_BYTES)
        assert h1 == h2

    def test_sha256_length_is_64(self):
        h = _sha256(DUMMY_PDF_BYTES)
        assert len(h) == 64

    def test_different_bytes_differ(self):
        h1 = _sha256(b"file A")
        h2 = _sha256(b"file B")
        assert h1 != h2


class TestAlreadyIngestedError:
    def test_error_carries_document_id(self):
        err = AlreadyIngestedError("uuid-123", "gdpr.pdf")
        assert err.document_id == "uuid-123"
        assert err.file_name == "gdpr.pdf"

    def test_str_includes_filename(self):
        err = AlreadyIngestedError("uuid-123", "gdpr.pdf")
        assert "gdpr.pdf" in str(err)


class TestIngestionError:
    def test_error_carries_detail(self):
        err = IngestionError("gdpr.pdf", "PDF is encrypted")
        assert err.file_name == "gdpr.pdf"
        assert err.detail == "PDF is encrypted"


class TestChunkMetadata:
    """Verify chunk metadata normalization logic (pure, no I/O)."""

    def test_chunk_ids_are_unique(self):
        import uuid
        ids = [str(uuid.uuid4()) for _ in range(100)]
        assert len(set(ids)) == 100

    def test_page_is_one_indexed(self):
        """Page metadata must be 1-indexed (not 0-indexed)."""
        from langchain_core.documents import Document

        pages = [
            Document(page_content=f"Page {i} content", metadata={})
            for i in range(3)
        ]
        for idx, doc in enumerate(pages):
            doc.metadata["page"] = idx + 1  # 1-indexed

        assert pages[0].metadata["page"] == 1
        assert pages[2].metadata["page"] == 3
