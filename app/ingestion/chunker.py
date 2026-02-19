"""
chunker.py – Split documents into overlapping chunks for embedding.

Uses LangChain's RecursiveCharacterTextSplitter with contract-aware
separators (section headers, clause markers, newlines, spaces).
"""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings

logger = logging.getLogger(__name__)

# Contract-specific separators – ordered from coarsest to finest
CONTRACT_SEPARATORS = [
    "\n\n\n",        # major section breaks
    "\n\n",          # paragraph breaks
    "\nSection ",    # section headers
    "\nClause ",     # clause markers
    "\nArticle ",    # article markers
    "\n",            # line breaks
    ". ",            # sentence boundaries
    " ",             # word boundaries
    "",              # character level (last resort)
]


def create_splitter(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build a text splitter tuned for legal/contract documents.

    Args:
        chunk_size:    Max chars per chunk (defaults to settings.chunk_size).
        chunk_overlap: Overlap between consecutive chunks (defaults to settings.chunk_overlap).

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        length_function=len,
        separators=CONTRACT_SEPARATORS,
        keep_separator=True,
    )


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunks.

    Adds a ``chunk_index`` field to each document's metadata so chunks
    can be traced back to their originating page / section.

    Args:
        documents:    Raw documents from the loader.
        chunk_size:   Override default chunk size.
        chunk_overlap: Override default overlap.

    Returns:
        List of chunked Document objects with enriched metadata.
    """
    splitter = create_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_total"] = len(chunks)
        # Truncate content for a human-readable preview
        chunk.metadata["preview"] = chunk.page_content[:120].replace("\n", " ")

    logger.info(
        f"Chunking: {len(documents)} docs → {len(chunks)} chunks "
        f"(size={chunk_size or settings.chunk_size}, "
        f"overlap={chunk_overlap or settings.chunk_overlap})"
    )
    return chunks
