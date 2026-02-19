"""
safety.py – Input and output guardrails for the Smart Contract Assistant.

Two main checks:
  1. Relevance Guard (input): Is the question related to the document?
     Uses cosine similarity between the query embedding and the document
     centroid embedding to catch completely off-topic queries.

  2. Factuality Disclaimer (output): Append a disclaimer if the answer
     contains speculative language (model may not be grounded).

  3. Harmful Content Filter (input): Reject requests to generate harmful,
     discriminatory, or manipulative legal language.
"""

import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


# ─── Cosine Similarity ───────────────────────────────────────────────────────

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ─── Relevance Guardrail ─────────────────────────────────────────────────────

# Keywords that are almost always relevant to contract QA
CONTRACT_KEYWORDS = {
    "contract", "agreement", "clause", "party", "parties", "obligation",
    "term", "condition", "payment", "liability", "indemnif", "warrant",
    "breach", "termination", "notice", "govern", "jurisdiction", "confidential",
    "intellectual property", "ip", "force majeure", "arbitration", "dispute",
    "penalty", "default", "amendment", "exhibit", "schedule", "appendix",
    "effective date", "scope", "service", "delivery", "warranty", "represent",
    "summary", "summarize", "summarise", "explain", "what", "who", "when",
    "how", "describe", "list", "find", "section", "article",
}

# Patterns that are clearly off-topic
OFF_TOPIC_PATTERNS = [
    r"\b(recipe|cook|bake|food|restaurant)\b",
    r"\b(sports?|football|basketball|cricket)\b",
    r"\b(weather|temperature|forecast)\b",
    r"\b(joke|funny|meme)\b",
    r"\b(stock|crypto|bitcoin|trading)\b",
    r"\b(celebrity|actor|movie|music)\b",
]

HARMFUL_PATTERNS = [
    r"\b(fraud|deceive|scam|cheat|exploit)\b",
    r"\b(discriminat|bias against|exclude)\b",
    r"generate.*illegal",
    r"create.*fake.*contract",
    r"forge|falsif",
]


class RelevanceGuardrail:
    """
    Multi-signal relevance check:
    1. Keyword-based fast check (no embedding call needed for clearly relevant queries)
    2. Embedding cosine similarity against document centroid (for uncertain cases)
    3. Explicit off-topic pattern matching
    4. Harmful content detection
    """

    def __init__(self, embedder: Embeddings, vector_store=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self._doc_centroid: Optional[List[float]] = None

        if vector_store is not None:
            self._compute_centroid()

    def _compute_centroid(self):
        """Compute the mean embedding of all document chunks."""
        try:
            docs = list(self.vector_store.docstore._dict.values())
            # Sample up to 50 chunks for efficiency
            sample = docs[::max(1, len(docs) // 50)][:50]
            texts = [d.page_content[:500] for d in sample]
            embeddings = self.embedder.embed_documents(texts)
            self._doc_centroid = np.mean(embeddings, axis=0).tolist()
            logger.info(f"Document centroid computed from {len(texts)} chunks.")
        except Exception as e:
            logger.warning(f"Could not compute document centroid: {e}")
            self._doc_centroid = None

    def update_store(self, vector_store):
        """Update the vector store and recompute centroid."""
        self.vector_store = vector_store
        self._compute_centroid()

    def _keyword_relevant(self, query: str) -> bool:
        q_lower = query.lower()
        return any(kw in q_lower for kw in CONTRACT_KEYWORDS)

    def _is_harmful(self, query: str) -> bool:
        q_lower = query.lower()
        return any(re.search(p, q_lower) for p in HARMFUL_PATTERNS)

    def _is_off_topic_pattern(self, query: str) -> bool:
        q_lower = query.lower()
        return any(re.search(p, q_lower) for p in OFF_TOPIC_PATTERNS)

    def check(self, query: str) -> Tuple[bool, str]:
        """
        Check if a query is safe and relevant.

        Returns:
            (is_allowed: bool, reason: str)
        """
        # 1. Harmful content – always block
        if self._is_harmful(query):
            return False, "harmful_content"

        # 2. Explicit off-topic pattern
        if self._is_off_topic_pattern(query):
            return False, "off_topic_pattern"

        # 3. Fast keyword check – if clearly relevant, allow immediately
        if self._keyword_relevant(query):
            return True, "keyword_match"

        # 4. Embedding similarity check (if centroid available)
        if self._doc_centroid is not None and self.embedder is not None:
            try:
                query_embedding = self.embedder.embed_query(query)
                sim = cosine_similarity(query_embedding, self._doc_centroid)
                logger.debug(f"Relevance similarity: {sim:.3f} (threshold={settings.guardrail_similarity_threshold})")
                if sim >= settings.guardrail_similarity_threshold:
                    return True, f"embedding_similarity={sim:.3f}"
                else:
                    return False, f"low_similarity={sim:.3f}"
            except Exception as e:
                logger.warning(f"Embedding check failed: {e}. Allowing query.")
                return True, "embedding_check_failed"

        # 5. Fallback – allow (better to answer than to over-block)
        return True, "fallback_allow"


# ─── Output Factuality Disclaimer ────────────────────────────────────────────

SPECULATIVE_PHRASES = [
    "i think", "i believe", "probably", "might be", "could be",
    "i'm not sure", "i am not sure", "possibly", "perhaps",
    "it seems", "it appears",
]

DISCLAIMER = (
    "\n\n---\n"
    "⚠️ *Disclaimer: This answer is based solely on the uploaded document. "
    "It is not legal advice. Always consult a qualified legal professional "
    "before acting on contract information.*"
)


def add_factuality_disclaimer(answer: str) -> str:
    """Append legal disclaimer to every answer."""
    return answer + DISCLAIMER


def check_output_quality(answer: str) -> dict:
    """
    Analyse output for quality signals.

    Returns:
        dict with keys: has_speculation, is_empty, word_count
    """
    lower = answer.lower()
    return {
        "has_speculation": any(p in lower for p in SPECULATIVE_PHRASES),
        "is_empty": len(answer.strip()) < 10,
        "word_count": len(answer.split()),
    }
