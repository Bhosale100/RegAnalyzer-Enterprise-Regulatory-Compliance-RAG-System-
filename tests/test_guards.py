"""
tests/test_guards.py
────────────────────
Unit tests for the three deterministic anti-hallucination guards.
All tests run without LLM or DB access.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from app.exceptions import CitationValidationError, InsufficientContextError
from app.pipeline.rag_chain import (
    _CITATION_RE,
    _format_context,
    _guard1_cosine_floor,
    _guard2_cross_encoder_floor,
    _union_results,
    _validate_citations,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _doc(text: str, source: str = "gdpr.pdf", page: int = 1, score: float = 0.0, chunk_id: str = "cid-1") -> Document:
    return Document(
        page_content=text,
        metadata={"source": source, "page": page, "score": score, "chunk_id": chunk_id, "relevance_score": score},
    )


# ── Guard 1 Tests ─────────────────────────────────────────────────────────────

class TestGuard1CosineFoor:
    def test_passes_when_cosine_above_floor(self):
        dense = [_doc("text", score=0.75, chunk_id="c1")]
        sparse = []
        # Should not raise
        _guard1_cosine_floor({"dense_docs": dense, "sparse_docs": sparse})

    def test_passes_when_bm25_has_results_even_low_cosine(self):
        dense = [_doc("text", score=0.10, chunk_id="c1")]
        sparse = [_doc("keyword match", chunk_id="c2")]
        # BM25 present → passes regardless of cosine
        _guard1_cosine_floor({"dense_docs": dense, "sparse_docs": sparse})

    def test_raises_when_cosine_low_and_no_bm25(self):
        dense = [_doc("text", score=0.10, chunk_id="c1")]
        sparse = []
        with pytest.raises(InsufficientContextError):
            _guard1_cosine_floor({"dense_docs": dense, "sparse_docs": sparse})

    def test_raises_when_no_results_at_all(self):
        with pytest.raises(InsufficientContextError):
            _guard1_cosine_floor({"dense_docs": [], "sparse_docs": []})


# ── Guard 2 Tests ─────────────────────────────────────────────────────────────

class TestGuard2CrossEncoderFoor:
    def test_passes_when_score_at_zero(self):
        docs = [_doc("text", score=0.0)]
        _guard2_cross_encoder_floor({"reranked_docs": docs})

    def test_passes_when_score_positive(self):
        docs = [_doc("text", score=0.85)]
        _guard2_cross_encoder_floor({"reranked_docs": docs})

    def test_raises_when_empty(self):
        with pytest.raises(InsufficientContextError):
            _guard2_cross_encoder_floor({"reranked_docs": []})

    def test_raises_when_score_below_floor(self):
        # Default floor is 0.0, so negative score triggers failure
        docs = [_doc("text", score=-0.5)]
        # relevance_score is used by guard 2
        docs[0].metadata["relevance_score"] = -0.5
        with pytest.raises(InsufficientContextError):
            _guard2_cross_encoder_floor({"reranked_docs": docs})


# ── Guard 3 Tests ─────────────────────────────────────────────────────────────

class TestGuard3CitationValidator:
    def _source_docs(self) -> list[Document]:
        return [
            _doc("Article 5 text", source="gdpr.pdf", page=12, chunk_id="c1"),
            _doc("Article 6 text", source="gdpr.pdf", page=15, chunk_id="c2"),
        ]

    def test_passes_with_valid_citation(self):
        answer = "Controllers must comply [Document: gdpr.pdf, Page: 12]."
        result = _validate_citations({"answer": answer, "source_docs": self._source_docs()})
        assert result["citations_verified"] is True

    def test_raises_when_zero_citations(self):
        answer = "There are some requirements but I won't cite them."
        with pytest.raises(CitationValidationError) as exc_info:
            _validate_citations({"answer": answer, "source_docs": self._source_docs()})
        assert "zero citations" in str(exc_info.value).lower()

    def test_raises_when_citation_not_in_retrieved(self):
        # Page 99 was never retrieved
        answer = "Some obligation [Document: gdpr.pdf, Page: 99]."
        with pytest.raises(CitationValidationError) as exc_info:
            _validate_citations({"answer": answer, "source_docs": self._source_docs()})
        assert exc_info.value.mismatches

    def test_multiple_valid_citations(self):
        answer = (
            "Lawfulness [Document: gdpr.pdf, Page: 12]. "
            "Consent [Document: gdpr.pdf, Page: 15]."
        )
        result = _validate_citations({"answer": answer, "source_docs": self._source_docs()})
        assert result["citations_verified"] is True


# ── Citation Regex Tests ──────────────────────────────────────────────────────

class TestCitationRegex:
    def test_standard_format(self):
        text = "See [Document: gdpr.pdf, Page: 5]."
        matches = _CITATION_RE.findall(text)
        assert len(matches) == 1
        assert matches[0] == ("gdpr.pdf", "5")

    def test_case_insensitive(self):
        text = "[document: GDPR.PDF, page: 10]"
        matches = _CITATION_RE.findall(text)
        assert len(matches) == 1

    def test_multiple_citations(self):
        text = "[Document: a.pdf, Page: 1] and [Document: b.pdf, Page: 2]"
        matches = _CITATION_RE.findall(text)
        assert len(matches) == 2

    def test_no_match_on_plain_text(self):
        assert _CITATION_RE.findall("No citations here.") == []


# ── Union Tests ───────────────────────────────────────────────────────────────

class TestUnionResults:
    def test_deduplicates_by_chunk_id(self):
        d1 = _doc("text A", chunk_id="same-id")
        d2 = _doc("text A duplicate", chunk_id="same-id")
        d3 = _doc("text B", chunk_id="other-id")
        result = _union_results([d1], [d2, d3])
        chunk_ids = [d.metadata["chunk_id"] for d in result]
        assert chunk_ids.count("same-id") == 1
        assert len(result) == 2

    def test_merges_both_lists(self):
        dense = [_doc("d1", chunk_id="c1"), _doc("d2", chunk_id="c2")]
        sparse = [_doc("s1", chunk_id="c3")]
        result = _union_results(dense, sparse)
        assert len(result) == 3


# ── Context Formatting Tests ──────────────────────────────────────────────────

class TestFormatContext:
    def test_includes_document_and_page_labels(self):
        docs = [_doc("Article 5 requires...", source="gdpr.pdf", page=7)]
        ctx = _format_context(docs)
        assert "Document: gdpr.pdf" in ctx
        assert "Page: 7" in ctx
        assert "Article 5 requires..." in ctx

    def test_multiple_excerpts_numbered(self):
        docs = [
            _doc("First", source="a.pdf", page=1, chunk_id="c1"),
            _doc("Second", source="b.pdf", page=2, chunk_id="c2"),
        ]
        ctx = _format_context(docs)
        assert "EXCERPT 1" in ctx
        assert "EXCERPT 2" in ctx
