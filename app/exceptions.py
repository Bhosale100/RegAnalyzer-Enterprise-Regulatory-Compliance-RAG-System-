"""
app/exceptions.py
─────────────────
Domain exceptions raised by the pipeline and mapped to HTTP status codes
in the FastAPI exception handlers.
"""


class ERCBaseError(Exception):
    """Root exception for all ERC Analyzer errors."""


class AlreadyIngestedError(ERCBaseError):
    """Raised when the SHA-256 of an uploaded file already exists in `documents`."""

    def __init__(self, document_id: str, file_name: str) -> None:
        self.document_id = document_id
        self.file_name = file_name
        super().__init__(
            f"Document '{file_name}' already ingested (id={document_id})"
        )


class InsufficientContextError(ERCBaseError):
    """
    Raised by Guard 1 or Guard 2 when no retrieved chunks are relevant
    enough to ground an answer.  Maps to HTTP 404.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Insufficient context: {reason}")


class CitationValidationError(ERCBaseError):
    """
    Raised by Guard 3 when the LLM answer contains zero citations or
    references chunks that were never retrieved.  Maps to HTTP 422.
    """

    def __init__(self, reason: str, mismatches: list[str] | None = None) -> None:
        self.reason = reason
        self.mismatches = mismatches or []
        super().__init__(f"Citation validation failed: {reason}")


class IngestionError(ERCBaseError):
    """Raised when PDF parsing or embedding storage fails."""

    def __init__(self, file_name: str, detail: str) -> None:
        self.file_name = file_name
        self.detail = detail
        super().__init__(f"Ingestion failed for '{file_name}': {detail}")
