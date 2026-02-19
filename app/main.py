"""
main.py – FastAPI + LangServe server entry point.

Endpoints:
  GET  /               → health check
  GET  /health         → detailed health status
  POST /ingest         → ingest a document (multipart form)
  POST /chat           → single-turn Q&A
  POST /summarise      → document summarisation
  POST /evaluate       → run LLM-as-a-Judge evaluation

LangServe routes (if USE_LANGSERVE=true):
  POST /retriever/invoke
  POST /generator/invoke
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart Contract Assistant API",
    description="RAG-powered contract analysis and Q&A service.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ─────────────────────────────────────────────────────────────

_state = {
    "rag_chain": None,
    "guardrail": None,
    "ingestion_info": {},
}


# ─── Pydantic Models ──────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    history: List[List[str]] = []
    add_disclaimer: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []
    disclaimer_added: bool = False


class EvaluationRequest(BaseModel):
    num_questions: int = 5


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "Smart Contract Assistant", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "document_loaded": _state["rag_chain"] is not None,
        "ingestion_info": {
            k: v for k, v in _state["ingestion_info"].items()
            if k not in ("store", "embedder")
        },
        "config": {
            "llm_provider": settings.llm_provider,
            "embed_provider": settings.embed_provider,
            "vector_store_type": settings.vector_store_type,
            "chunk_size": settings.chunk_size,
            "retrieval_k": settings.retrieval_k,
        },
    }


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload and ingest a contract document."""
    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt", ".md"):
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        from app.ingestion.embedder import ingest_document
        from app.retrieval.rag_chain import RAGChain
        from app.guardrails.safety import RelevanceGuardrail

        result = ingest_document(tmp_path)

        _state["rag_chain"] = RAGChain(
            vector_store=result["store"],
            embedder=result["embedder"],
        )
        _state["guardrail"] = RelevanceGuardrail(
            embedder=result["embedder"],
            vector_store=result["store"],
        )
        _state["ingestion_info"] = {
            k: v for k, v in result.items() if k not in ("store", "embedder")
        }
        _state["ingestion_info"]["original_filename"] = file.filename

        return {
            "status": "success",
            "message": f"Document '{file.filename}' ingested successfully.",
            "details": _state["ingestion_info"],
        }
    except Exception as e:
        logger.exception(f"Ingestion error: {e}")
        raise HTTPException(500, f"Ingestion failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Answer a question about the ingested document."""
    rag = _state.get("rag_chain")
    guard = _state.get("guardrail")

    if rag is None:
        raise HTTPException(400, "No document has been ingested yet. Call /ingest first.")

    # Guardrail
    if guard is not None:
        allowed, reason = guard.check(req.question)
        if not allowed:
            if "harmful" in reason:
                msg = "🚫 I cannot assist with that request."
            else:
                msg = settings.off_topic_response
            return ChatResponse(answer=msg)

    # RAG
    result = rag.invoke(req.question, history=req.history)
    answer = result["answer"]

    sources = []
    for doc in result.get("source_docs", []):
        sources.append({
            "source": doc.metadata.get("source", ""),
            "page": doc.metadata.get("page", ""),
            "preview": doc.page_content[:200],
        })

    if req.add_disclaimer:
        from app.guardrails.safety import add_factuality_disclaimer
        answer = add_factuality_disclaimer(answer)

    return ChatResponse(
        answer=answer,
        sources=sources,
        disclaimer_added=req.add_disclaimer,
    )


@app.post("/summarise")
def summarise():
    """Generate a structured summary of the ingested document."""
    rag = _state.get("rag_chain")
    if rag is None:
        raise HTTPException(400, "No document has been ingested yet.")
    try:
        summary = rag.summarise()
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(500, f"Summarisation failed: {str(e)}")


@app.post("/evaluate")
def evaluate(req: EvaluationRequest):
    """Run LLM-as-a-Judge evaluation on the ingested document."""
    rag = _state.get("rag_chain")
    if rag is None:
        raise HTTPException(400, "No document has been ingested yet.")
    try:
        from app.evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator(rag_chain=rag, num_questions=req.num_questions)
        report = evaluator.run()
        return {
            "preference_score": report.preference_score,
            "num_passed": report.num_passed,
            "num_total": report.num_total,
            "summary": report.summary,
            "results": [
                {
                    "question": r.question,
                    "synth_answer": r.synth_answer,
                    "rag_answer": r.rag_answer,
                    "score": r.score,
                    "passed": r.passed,
                    "judge_output": r.judge_output,
                }
                for r in report.results
            ],
            "markdown_report": report.to_markdown(),
        }
    except Exception as e:
        logger.exception(f"Evaluation error: {e}")
        raise HTTPException(500, f"Evaluation failed: {str(e)}")


# ─── LangServe Integration (optional) ────────────────────────────────────────

if os.getenv("USE_LANGSERVE", "false").lower() == "true":
    try:
        from langserve import add_routes
        from langchain_core.runnables import RunnableLambda

        def retriever_fn(query: str):
            rag = _state.get("rag_chain")
            if rag is None:
                return {"error": "No document loaded"}
            docs = rag.vector_store.similarity_search(query, k=settings.retrieval_k)
            return [{"content": d.page_content, "metadata": d.metadata} for d in docs]

        def generator_fn(payload: dict):
            rag = _state.get("rag_chain")
            if rag is None:
                return {"error": "No document loaded"}
            result = rag.invoke(payload.get("input", ""), payload.get("history", []))
            return result["answer"]

        add_routes(app, RunnableLambda(retriever_fn), path="/retriever")
        add_routes(app, RunnableLambda(generator_fn), path="/generator")
        logger.info("LangServe routes registered: /retriever, /generator")
    except ImportError:
        logger.warning("langserve not installed – LangServe routes skipped.")


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
