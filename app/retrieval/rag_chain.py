"""
rag_chain.py – RAG pipeline with conversation history and source citations.
Supports: OpenAI, Groq, NVIDIA, HuggingFace
"""

import logging
from operator import itemgetter
from typing import Generator, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_transformers import LongContextReorder

from app.config import settings

logger = logging.getLogger(__name__)


# ─── LLM Factory ────────────────────────────────────────────────────────────

def get_llm():
    """Return the configured LLM."""
    provider = settings.llm_provider.lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        logger.info(f"Using Groq LLM: {settings.groq_model}")
        return ChatGroq(
            model=settings.groq_model,
            groq_api_key=settings.groq_api_key or None,
            temperature=0.2,
        )

    elif provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        logger.info(f"Using NVIDIA LLM: {settings.nvidia_llm_model}")
        return ChatNVIDIA(
            model=settings.nvidia_llm_model,
            nvidia_api_key=settings.nvidia_api_key or None,
        )

    elif provider == "huggingface":
        from langchain_community.llms import HuggingFacePipeline
        from transformers import pipeline
        logger.info(f"Using HuggingFace LLM: {settings.hf_model_id}")
        pipe = pipeline("text2text-generation", model=settings.hf_model_id, max_new_tokens=512)
        return HuggingFacePipeline(pipeline=pipe)

    else:  # openai
        from langchain_openai import ChatOpenAI
        logger.info(f"Using OpenAI LLM: {settings.openai_model}")
        return ChatOpenAI(
            model=settings.openai_model,
            openai_api_key=settings.openai_api_key or None,
            temperature=0.2,
            streaming=True,
        )


# ─── Helpers ────────────────────────────────────────────────────────────────

def docs_to_context_string(docs: List[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i} – {source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(history: List[tuple]) -> List[dict]:
    messages = []
    for human, ai in history:
        if human:
            messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})
    return messages


# ─── Prompts ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Smart Contract Assistant – a precise and helpful AI
specialised in analysing legal contracts, agreements, and policy documents.

RULES you must follow:
1. Answer ONLY based on the retrieved document context provided below.
2. If the context does not contain the answer, say "I could not find this information in the document."
3. Always cite the source and page number when referencing specific clauses.
4. Never fabricate clauses, dates, parties, or obligations.
5. Be concise but complete. Use bullet points for lists of obligations or rights.
6. If asked to summarise, provide a structured summary with key sections.

Retrieved Context:
{context}
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

summarise_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Smart Contract Assistant.
Summarise the following contract document clearly and concisely.
Structure your response with these sections:
- **Parties Involved**
- **Purpose / Scope**
- **Key Obligations**
- **Payment Terms** (if any)
- **Duration & Termination**
- **Important Clauses** (e.g., liability, confidentiality, dispute resolution)
- **Risks / Red Flags** (if any)

Contract Content:
{context}
"""),
    ("human", "Please provide a structured summary of this contract."),
])


# ─── RAG Chain ───────────────────────────────────────────────────────────────

class RAGChain:
    def __init__(self, vector_store, embedder=None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.llm = get_llm()
        self._last_source_docs = []
        self._build_chain()

    def _build_chain(self):
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retrieval_k},
        )

        def get_context(input_dict):
            docs = retriever.invoke(input_dict["input"])
            if settings.reorder_long_context:
                docs = LongContextReorder().transform_documents(docs)
            return docs_to_context_string(docs)

        def get_docs(input_dict):
            docs = retriever.invoke(input_dict["input"])
            if settings.reorder_long_context:
                docs = LongContextReorder().transform_documents(docs)
            return docs

        self.retrieval_chain = RunnablePassthrough.assign(
            context=RunnableLambda(get_context),
            source_docs=RunnableLambda(get_docs),
        )

        self.generator_chain = chat_prompt | self.llm | StrOutputParser()

        self.full_chain = self.retrieval_chain | {
            "output": self.generator_chain,
            "source_docs": itemgetter("source_docs"),
        }

    def invoke(self, question: str, history: List[tuple] = None) -> dict:
        history = history or []
        result = self.full_chain.invoke({
            "input": question,
            "history": format_history(history),
        })
        return {
            "answer": result["output"],
            "source_docs": result.get("source_docs", []),
        }

    def stream(self, question: str, history: List[tuple] = None) -> Generator:
        history = history or []
        retrieved = self.retrieval_chain.invoke({
            "input": question,
            "history": format_history(history),
        })
        self._last_source_docs = retrieved.get("source_docs", [])
        for token in self.generator_chain.stream({
            "input": question,
            "context": retrieved["context"],
            "history": format_history(history),
        }):
            yield token

    def get_last_sources(self) -> List[Document]:
        return self._last_source_docs

    def summarise(self) -> str:
        all_docs = list(self.vector_store.docstore._dict.values())
        step = max(1, len(all_docs) // 20)
        sample = all_docs[::step][:20]
        context = docs_to_context_string(sample)
        chain = summarise_prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context})
