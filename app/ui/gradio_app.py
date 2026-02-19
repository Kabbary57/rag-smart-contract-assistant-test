"""
gradio_app.py – Gradio UI for the Smart Contract Assistant.

Tabs:
  📄 Upload & Ingest   – Upload PDF/DOCX, trigger ingestion pipeline
  💬 Chat              – Ask questions about the uploaded document
  📝 Summary           – Auto-generate a structured contract summary
  📊 Evaluation        – Run LLM-as-a-Judge evaluation and view report
  ℹ️  About            – Project info and usage guide
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from app.config import settings

logger = logging.getLogger(__name__)

# ─── Shared State (module-level, reset on new upload) ───────────────────────
_rag_chain = None
_guardrail = None
_ingestion_info = {}


def _get_rag() :
    return _rag_chain


def _get_guard():
    return _guardrail


# ─── Ingestion Handler ───────────────────────────────────────────────────────

def handle_upload(file_obj) -> Tuple[str, str]:
    """
    Process uploaded file through ingestion pipeline.
    Returns (status_message, file_info_markdown).
    """
    global _rag_chain, _guardrail, _ingestion_info

    if file_obj is None:
        return "⚠️ No file uploaded.", ""

    file_path = file_obj.name
    file_name = Path(file_path).name
    file_size = Path(file_path).stat().st_size / (1024 * 1024)

    if file_size > settings.max_file_size_mb:
        return (
            f"❌ File too large ({file_size:.1f} MB). Max allowed: {settings.max_file_size_mb} MB.",
            "",
        )

    try:
        from app.ingestion.embedder import ingest_document
        from app.retrieval.rag_chain import RAGChain
        from app.guardrails.safety import RelevanceGuardrail

        # Run ingestion
        result = ingest_document(file_path)

        # Build RAG chain
        _rag_chain = RAGChain(
            vector_store=result["store"],
            embedder=result["embedder"],
        )

        # Build guardrail
        _guardrail = RelevanceGuardrail(
            embedder=result["embedder"],
            vector_store=result["store"],
        )

        _ingestion_info = result

        status = f"✅ Successfully ingested **{file_name}**!"
        info_md = (
            f"| Field | Value |\n|---|---|\n"
            f"| 📄 File | `{file_name}` |\n"
            f"| 📦 Size | `{file_size:.2f} MB` |\n"
            f"| 📃 Pages | `{result['num_pages']}` |\n"
            f"| 🔢 Chunks | `{result['num_chunks']}` |\n"
            f"| 🗄️ Store | `{settings.vector_store_type.upper()}` |\n"
            f"| 🧠 Embeddings | `{settings.embed_provider}` |"
        )
        return status, info_md

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return f"❌ Ingestion failed: {str(e)}", ""


# ─── Chat Handler ────────────────────────────────────────────────────────────

def handle_chat(
    user_message: str,
    history: List[dict],
    show_sources: bool,
    add_disclaimer: bool,
) -> Tuple[str, List[dict], str]:
    """
    Process a chat message.
    Returns (empty_input, updated_history, sources_markdown).
    history format for Gradio 6: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    rag = _get_rag()
    guard = _get_guard()

    if not user_message.strip():
        return "", history, ""

    if rag is None:
        bot_reply = "⚠️ Please upload a document first using the **Upload & Ingest** tab."
        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": bot_reply})
        return "", history, ""

    # Guardrail check
    if guard is not None:
        allowed, reason = guard.check(user_message)
        if not allowed:
            if "harmful" in reason:
                bot_reply = "🚫 I cannot assist with that request. Please ask appropriate questions about the contract."
            else:
                bot_reply = settings.off_topic_response
            history.append({"role": "user", "content": user_message})
            history.append({"role": "assistant", "content": bot_reply})
            return "", history, ""

    try:
        # Convert Gradio 6 history format to tuples for rag_chain
        history_tuples = []
        msgs = list(history)
        for i in range(0, len(msgs) - 1, 2):
            if msgs[i]["role"] == "user" and msgs[i+1]["role"] == "assistant":
                history_tuples.append((msgs[i]["content"], msgs[i+1]["content"]))

        # Stream the answer
        answer_parts = []
        for token in rag.stream(user_message, history=history_tuples):
            answer_parts.append(token)

        answer = "".join(answer_parts)

        # Add disclaimer
        if add_disclaimer:
            from app.guardrails.safety import add_factuality_disclaimer
            answer = add_factuality_disclaimer(answer)

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": answer})

        # Format source citations
        sources_md = ""
        if show_sources:
            source_docs = rag.get_last_sources()
            if source_docs:
                lines = ["### 📎 Source Citations\n"]
                for i, doc in enumerate(source_docs, 1):
                    src = doc.metadata.get("source", "Document")
                    page = doc.metadata.get("page", "?")
                    preview = doc.page_content[:200].replace("\n", " ")
                    lines.append(
                        f"**[{i}] {src} – Page {page}**\n"
                        f"> {preview}…\n"
                    )
                sources_md = "\n".join(lines)

        return "", history, sources_md

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        error_msg = f"❌ Error generating response: {str(e)}"
        history.append((user_message, error_msg))
        return "", history, ""


# ─── Summary Handler ─────────────────────────────────────────────────────────

def handle_summary() -> str:
    rag = _get_rag()
    if rag is None:
        return "⚠️ Please upload a document first."
    try:
        return rag.summarise()
    except Exception as e:
        logger.exception(f"Summary error: {e}")
        return f"❌ Summary failed: {str(e)}"


# ─── Evaluation Handler ──────────────────────────────────────────────────────

def handle_evaluation(num_questions: int, progress=gr.Progress()) -> str:
    rag = _get_rag()
    if rag is None:
        return "⚠️ Please upload a document first."

    try:
        from app.evaluation.evaluator import RAGEvaluator

        evaluator = RAGEvaluator(rag_chain=rag, num_questions=int(num_questions))

        def prog_cb(step, total, msg):
            progress(step / total, desc=msg)

        report = evaluator.run(progress_callback=prog_cb)
        return report.to_markdown()

    except Exception as e:
        logger.exception(f"Evaluation error: {e}")
        return f"❌ Evaluation failed: {str(e)}"


# ─── UI Layout ───────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    THEME = gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="📄 Smart Contract Assistant") as demo:

        # ── Header ───────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-text">
            <h1>📄 Smart Contract Assistant</h1>
            <p style="color:#666">Upload a contract · Ask questions · Get summaries · Evaluate quality</p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Upload ─────────────────────────────────────────────
            with gr.Tab("📁 Upload & Ingest"):
                gr.Markdown("### Upload your contract (PDF or DOCX)")
                with gr.Row():
                    with gr.Column(scale=2):
                        upload_btn = gr.File(
                            label="Drop file here or click to browse",
                            file_types=[".pdf", ".docx", ".txt", ".md"],
                            type="filepath",
                        )
                        ingest_btn = gr.Button("🚀 Ingest Document", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        upload_status = gr.Markdown("*No document loaded yet.*")
                        file_info = gr.Markdown()

                ingest_btn.click(
                    fn=handle_upload,
                    inputs=[upload_btn],
                    outputs=[upload_status, file_info],
                )

            # ── Tab 2: Chat ───────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                gr.Markdown("### Ask questions about the uploaded contract")
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=450,
                        )
                        with gr.Row():
                            chat_input = gr.Textbox(
                                placeholder="Ask a question about the contract…",
                                label="",
                                scale=5,
                                lines=1,
                            )
                            send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                        with gr.Row():
                            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
                            show_sources = gr.Checkbox(label="Show source citations", value=True)
                            add_disclaimer = gr.Checkbox(label="Add legal disclaimer", value=True)

                    with gr.Column(scale=2):
                        sources_display = gr.Markdown(
                            label="Source Citations",
                            elem_classes=["source-panel"],
                        )

                # Example questions
                gr.Examples(
                    examples=[
                        ["What are the main obligations of each party?"],
                        ["What is the payment schedule?"],
                        ["When does this contract expire?"],
                        ["What are the termination conditions?"],
                        ["Is there a confidentiality clause?"],
                        ["What happens in case of a breach?"],
                        ["What is the governing law?"],
                    ],
                    inputs=chat_input,
                    label="💡 Example Questions",
                )

                # Event handlers
                send_btn.click(
                    fn=handle_chat,
                    inputs=[chat_input, chatbot, show_sources, add_disclaimer],
                    outputs=[chat_input, chatbot, sources_display],
                )
                chat_input.submit(
                    fn=handle_chat,
                    inputs=[chat_input, chatbot, show_sources, add_disclaimer],
                    outputs=[chat_input, chatbot, sources_display],
                )
                clear_btn.click(lambda: ([], ""), outputs=[chatbot, sources_display])

            # ── Tab 3: Summary ────────────────────────────────────────────
            with gr.Tab("📝 Summary"):
                gr.Markdown("### Auto-generate a structured contract summary")
                summarise_btn = gr.Button("📋 Generate Summary", variant="primary", size="lg")
                summary_output = gr.Markdown(
                    label="Contract Summary",
                    value="*Click the button above to generate a summary.*",
                )
                summarise_btn.click(
                    fn=handle_summary,
                    outputs=[summary_output],
                )

            # ── Tab 4: Evaluation ─────────────────────────────────────────
            with gr.Tab("📊 Evaluation"):
                gr.Markdown(
                    "### LLM-as-a-Judge Evaluation\n"
                    "Generates synthetic Q&A pairs from the document, then evaluates "
                    "whether the RAG pipeline answers them correctly.\n\n"
                    "**Score [2]** = RAG answer is better or equivalent to the ground truth."
                )
                with gr.Row():
                    num_q_slider = gr.Slider(
                        minimum=2, maximum=10, value=5, step=1,
                        label="Number of evaluation questions",
                    )
                    eval_btn = gr.Button("▶️ Run Evaluation", variant="primary")

                eval_output = gr.Markdown(
                    value="*Run evaluation to see results.*",
                    label="Evaluation Report",
                )
                eval_btn.click(
                    fn=handle_evaluation,
                    inputs=[num_q_slider],
                    outputs=[eval_output],
                )

            # ── Tab 5: About ──────────────────────────────────────────────
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
## Smart Contract Assistant

An end-to-end **Retrieval-Augmented Generation (RAG)** application for analysing
legal contracts, insurance policies, and long-form agreements.

### 🏗️ Architecture

```
PDF/DOCX Upload
      │
      ▼
Ingestion Pipeline
  ├─ Document Loader (PyMuPDF / python-docx)
  ├─ Chunker (RecursiveCharacterTextSplitter)
  └─ Embedder → Vector Store (FAISS / Chroma)
      │
      ▼
Query Processing
  ├─ Relevance Guardrail (embedding cosine similarity)
  ├─ Retriever (top-k semantic search + LongContextReorder)
  └─ LLM Generator (with conversation history)
      │
      ▼
Evaluation
  └─ LLM-as-a-Judge (synthetic QA → pairwise scoring)
```

### ⚙️ Configuration

Set the following environment variables (or edit `.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` / `nvidia` / `huggingface` |
| `EMBED_PROVIDER` | `openai` | `openai` / `nvidia` / `sentence_transformers` |
| `OPENAI_API_KEY` | – | Required for OpenAI |
| `NVIDIA_API_KEY` | – | Required for NVIDIA NIM |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` / `chroma` |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `RETRIEVAL_K` | `5` | Top-k chunks retrieved |

### ⚠️ Legal Disclaimer

This tool is for informational purposes only and does not constitute legal advice.
Always consult a qualified legal professional before acting on any contract information.
                """)

    return demo


def launch(share: bool = False, server_port: int = None):
    """Launch the Gradio interface."""
    demo = build_ui()
    demo.launch(
        server_name=settings.host,
        server_port=server_port or settings.gradio_port,
        share=share,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
