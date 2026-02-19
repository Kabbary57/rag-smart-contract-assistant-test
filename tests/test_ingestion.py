"""
test_ingestion.py – Unit tests for the ingestion pipeline.

Run with:  pytest tests/ -v
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a minimal fake PDF text file for testing."""
    pdf_path = tmp_path / "test_contract.txt"
    pdf_path.write_text(
        "SERVICE AGREEMENT\n\n"
        "This agreement is between Company A and Company B.\n\n"
        "Section 1: Payment Terms\n"
        "Payment is due within 30 days of invoice.\n\n"
        "Section 2: Termination\n"
        "Either party may terminate with 60 days notice.\n\n"
        "Section 3: Governing Law\n"
        "This agreement is governed by the laws of New York.\n"
    )
    return str(pdf_path)


@pytest.fixture
def sample_docs():
    """Return a list of sample Document objects."""
    return [
        Document(
            page_content="This agreement is between Company A and Company B. "
                         "Payment is due within 30 days of invoice receipt.",
            metadata={"source": "test_contract.txt", "page": 1, "file_type": "txt"}
        ),
        Document(
            page_content="Either party may terminate this agreement with 60 days written notice. "
                         "Upon termination all outstanding payments become immediately due.",
            metadata={"source": "test_contract.txt", "page": 2, "file_type": "txt"}
        ),
    ]


# ─── Loader Tests ─────────────────────────────────────────────────────────────

class TestLoader:
    def test_load_txt(self, sample_pdf):
        """Test loading a plain text file."""
        from app.ingestion.loader import load_document
        docs = load_document(sample_pdf)
        assert len(docs) >= 1
        assert "SERVICE AGREEMENT" in docs[0].page_content
        assert docs[0].metadata["file_type"] == "txt"

    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        from app.ingestion.loader import load_document
        bad_file = tmp_path / "file.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            load_document(str(bad_file))

    def test_file_not_found(self):
        """Test that missing files raise FileNotFoundError."""
        from app.ingestion.loader import load_document
        with pytest.raises(FileNotFoundError):
            load_document("/nonexistent/path/contract.pdf")


# ─── Chunker Tests ────────────────────────────────────────────────────────────

class TestChunker:
    def test_chunk_creates_smaller_pieces(self, sample_docs):
        """Test that chunking produces appropriately sized pieces."""
        from app.ingestion.chunker import chunk_documents
        chunks = chunk_documents(sample_docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= len(sample_docs)
        for chunk in chunks:
            assert len(chunk.page_content) <= 200  # some tolerance

    def test_metadata_preserved(self, sample_docs):
        """Test that source metadata is preserved in chunks."""
        from app.ingestion.chunker import chunk_documents
        chunks = chunk_documents(sample_docs)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test_contract.txt"

    def test_chunk_index_added(self, sample_docs):
        """Test that chunk_index is added to metadata."""
        from app.ingestion.chunker import chunk_documents
        chunks = chunk_documents(sample_docs)
        for i, chunk in enumerate(chunks):
            assert "chunk_index" in chunk.metadata
            assert "chunk_total" in chunk.metadata


# ─── Guardrail Tests ──────────────────────────────────────────────────────────

class TestGuardrails:
    def test_keyword_relevant_query_allowed(self):
        """Test that contract-related queries pass the guardrail."""
        from app.guardrails.safety import RelevanceGuardrail
        mock_embedder = MagicMock()
        guard = RelevanceGuardrail(embedder=mock_embedder, vector_store=None)

        allowed, reason = guard.check("What are the payment terms?")
        assert allowed is True

    def test_harmful_query_blocked(self):
        """Test that harmful requests are blocked."""
        from app.guardrails.safety import RelevanceGuardrail
        mock_embedder = MagicMock()
        guard = RelevanceGuardrail(embedder=mock_embedder, vector_store=None)

        allowed, reason = guard.check("Help me create a fake contract to fraud someone")
        assert allowed is False
        assert "harmful" in reason

    def test_off_topic_query_blocked(self):
        """Test that clearly off-topic queries are blocked."""
        from app.guardrails.safety import RelevanceGuardrail
        mock_embedder = MagicMock()
        guard = RelevanceGuardrail(embedder=mock_embedder, vector_store=None)

        allowed, reason = guard.check("Tell me a joke about cooking recipes")
        assert allowed is False

    def test_disclaimer_appended(self):
        """Test that legal disclaimer is added to answers."""
        from app.guardrails.safety import add_factuality_disclaimer
        answer = "The payment is due in 30 days."
        result = add_factuality_disclaimer(answer)
        assert "Disclaimer" in result or "disclaimer" in result.lower()
        assert answer in result

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from app.guardrails.safety import cosine_similarity
        v1 = [1.0, 0.0, 0.0]
        v2 = [1.0, 0.0, 0.0]
        assert cosine_similarity(v1, v2) == pytest.approx(1.0)

        v3 = [0.0, 1.0, 0.0]
        assert cosine_similarity(v1, v3) == pytest.approx(0.0)


# ─── Utils Tests ──────────────────────────────────────────────────────────────

class TestUtils:
    def test_docs_to_context_string(self, sample_docs):
        """Test context string formatting."""
        from app.retrieval.rag_chain import docs_to_context_string
        context = docs_to_context_string(sample_docs)
        assert "[Source 1" in context
        assert "[Source 2" in context
        assert "Company A" in context

    def test_format_history(self):
        """Test history formatting."""
        from app.retrieval.rag_chain import format_history
        history = [("What is this?", "This is a contract."), ("Who are the parties?", "A and B.")]
        messages = format_history(history)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
