"""
embedder.py – Embedding creation and vector store management.
Defaults to sentence_transformers (fully free, no API key needed).
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


def get_embedder() -> Embeddings:
    """Return embedding model. Defaults to sentence_transformers (free, local)."""
    provider = settings.embed_provider.lower()

    if provider == "openai":
        logger.info("Using OpenAI embeddings")
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=settings.openai_api_key or None,
        )

    elif provider == "nvidia":
        logger.info(f"Using NVIDIA embeddings: {settings.nvidia_embed_model}")
        from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
        return NVIDIAEmbeddings(
            model=settings.nvidia_embed_model,
            truncate="END",
            nvidia_api_key=settings.nvidia_api_key or None,
        )

    else:  # sentence_transformers (default — free, local, no API key)
        logger.info(f"Using SentenceTransformers: {settings.st_model_name}")
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=settings.st_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )


def build_vector_store(chunks: List[Document], embedder: Optional[Embeddings] = None):
    """Build a FAISS or Chroma vector store from document chunks."""
    if embedder is None:
        embedder = get_embedder()

    store_type = settings.vector_store_type.lower()

    if store_type == "chroma":
        from langchain_community.vectorstores import Chroma
        store = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            collection_name=settings.chroma_collection,
            persist_directory=settings.vector_store_path,
        )
    else:  # faiss
        logger.info(f"Building FAISS store from {len(chunks)} chunks …")
        from langchain_community.vectorstores import FAISS
        store = FAISS.from_documents(documents=chunks, embedding=embedder)
        store.save_local(settings.vector_store_path)
        logger.info(f"FAISS index saved to {settings.vector_store_path}/")

    return store


def load_vector_store(embedder: Optional[Embeddings] = None):
    """Load an existing vector store from disk."""
    if embedder is None:
        embedder = get_embedder()

    path = settings.vector_store_path

    if settings.vector_store_type.lower() == "chroma":
        from langchain_community.vectorstores import Chroma
        if os.path.isdir(path):
            return Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=embedder,
                persist_directory=path,
            )
    else:
        from langchain_community.vectorstores import FAISS
        if (Path(path) / "index.faiss").exists():
            return FAISS.load_local(path, embedder, allow_dangerous_deserialization=True)

    return None


def ingest_document(file_path: str) -> dict:
    """Full ingestion pipeline: load → chunk → embed → store."""
    from app.ingestion.loader import load_document
    from app.ingestion.chunker import chunk_documents

    logger.info(f"Starting ingestion for: {file_path}")

    documents = load_document(file_path)
    logger.info(f"Loaded {len(documents)} pages")

    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks")

    embedder = get_embedder()
    store = build_vector_store(chunks, embedder)

    return {
        "store": store,
        "embedder": embedder,
        "num_chunks": len(chunks),
        "num_pages": len(documents),
        "source_file": Path(file_path).name,
    }
