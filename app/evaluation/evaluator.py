"""
evaluator.py – LLM-as-a-Judge evaluation pipeline.

Implements the pairwise evaluation pattern from Notebook 08:
  1. Sample random chunks from the vector store.
  2. Generate synthetic question-answer pairs from those chunks.
  3. Get the RAG chain's answers to those questions.
  4. Use a judge LLM to score each (synth_answer vs rag_answer).
  5. Aggregate into a preference score (% where RAG >= synth).

Score key:
  [1] RAG answer is worse / introduces inconsistencies
  [2] RAG answer is better or equivalent to the ground truth
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class SyntheticQA:
    question: str
    synth_answer: str
    source_chunks: List[Document]


@dataclass
class EvalResult:
    question: str
    synth_answer: str
    rag_answer: str
    judge_output: str
    score: int          # 1 or 2
    passed: bool        # True if score == 2


@dataclass
class EvaluationReport:
    results: List[EvalResult] = field(default_factory=list)
    preference_score: float = 0.0
    num_passed: int = 0
    num_total: int = 0
    summary: str = ""

    def to_markdown(self) -> str:
        lines = [
            "# 📊 RAG Evaluation Report",
            f"\n**Preference Score:** {self.preference_score:.1%}  ({self.num_passed}/{self.num_total} passed)\n",
            "---",
        ]
        for i, r in enumerate(self.results, 1):
            icon = "✅" if r.passed else "❌"
            lines += [
                f"\n## {icon} Question {i}",
                f"**Q:** {r.question}\n",
                f"**Ground-Truth (Synthetic):**\n> {r.synth_answer}\n",
                f"**RAG Answer:**\n> {r.rag_answer}\n",
                f"**Judge:**\n```\n{r.judge_output}\n```",
                "---",
            ]
        lines.append(f"\n**Conclusion:** {self.summary}")
        return "\n".join(lines)


# ─── Prompts ─────────────────────────────────────────────────────────────────

QA_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are creating evaluation data for a contract Q&A system. "
     "Given the two document excerpts, generate ONE specific, answerable question "
     "and its correct answer. Focus on specific facts: dates, parties, obligations, "
     "payment terms, clauses, or conditions mentioned in the text.\n\n"
     "Format EXACTLY:\n"
     "Question: <the question>\n\n"
     "Answer: <the answer derived directly from the text>"),
    ("user",
     "Excerpt 1:\n{chunk1}\n\n"
     "Excerpt 2:\n{chunk2}"),
])

JUDGE_PROMPT = ChatPromptTemplate.from_template("""You are an impartial judge evaluating the quality of a contract Q&A system.

QUESTION: {question}

ANSWER 1 (Ground Truth – generated directly from document text, assume this is correct):
{synth_answer}

ANSWER 2 (RAG System Answer – to be evaluated):
{rag_answer}

Evaluation criteria:
[1] Answer 2 is WORSE: it is vague, incorrect, contradicts Answer 1, or fails to address the question.
[2] Answer 2 is BETTER or EQUIVALENT: it is accurate, complete, cites the right information, and is consistent with Answer 1.

Respond with ONLY:
[Score] Brief justification (1-2 sentences)

EVALUATION:""")


# ─── Evaluator Class ─────────────────────────────────────────────────────────

class RAGEvaluator:
    """
    Runs the full LLM-as-a-Judge evaluation loop.

    Args:
        rag_chain: An initialised RAGChain instance.
        num_questions: Number of synthetic QA pairs to generate.
    """

    def __init__(self, rag_chain, num_questions: int = 5):
        self.rag_chain = rag_chain
        self.num_questions = num_questions
        self.llm = rag_chain.llm
        self.str_llm = self.llm | StrOutputParser()

    def _format_chunk(self, doc: Document) -> str:
        source = doc.metadata.get("source", "Document")
        page = doc.metadata.get("page", "?")
        return f"[{source}, Page {page}]\n{doc.page_content[:800]}"

    def _generate_synthetic_qa(self, doc1: Document, doc2: Document) -> Optional[SyntheticQA]:
        """Generate a synthetic QA pair from two document chunks."""
        try:
            chain = QA_GENERATION_PROMPT | self.str_llm
            output = chain.invoke({
                "chunk1": self._format_chunk(doc1),
                "chunk2": self._format_chunk(doc2),
            })

            # Parse "Question: ...\n\nAnswer: ..."
            parts = output.split("\n\nAnswer:", 1)
            if len(parts) != 2:
                parts = output.split("\nAnswer:", 1)
            if len(parts) != 2:
                logger.warning(f"Could not parse QA output: {output[:100]}")
                return None

            question = parts[0].replace("Question:", "").strip()
            answer = parts[1].strip()

            return SyntheticQA(
                question=question,
                synth_answer=answer,
                source_chunks=[doc1, doc2],
            )
        except Exception as e:
            logger.error(f"QA generation failed: {e}")
            return None

    def _get_rag_answer(self, question: str) -> str:
        """Get the RAG chain's answer to a question."""
        try:
            result = self.rag_chain.invoke(question, history=[])
            return result["answer"]
        except Exception as e:
            logger.error(f"RAG answer failed: {e}")
            return ""

    def _judge(self, qa: SyntheticQA, rag_answer: str) -> Tuple[str, int]:
        """Use judge LLM to score the RAG answer vs the synthetic answer."""
        try:
            chain = JUDGE_PROMPT | self.str_llm
            output = chain.invoke({
                "question": qa.question,
                "synth_answer": qa.synth_answer,
                "rag_answer": rag_answer if rag_answer.strip() else "(no answer provided)",
            })
            score = 2 if "[2]" in output else 1
            return output.strip(), score
        except Exception as e:
            logger.error(f"Judge LLM failed: {e}")
            return f"Error: {e}", 1

    def run(self, progress_callback=None) -> EvaluationReport:
        """
        Run the full evaluation pipeline.

        Args:
            progress_callback: Optional callable(step: int, total: int, message: str)

        Returns:
            EvaluationReport with all results and aggregated score.
        """
        # Get all chunks from the vector store
        all_docs = list(self.rag_chain.vector_store.docstore._dict.values())
        if len(all_docs) < 2:
            logger.warning("Not enough documents in store for evaluation.")
            return EvaluationReport(summary="Insufficient documents for evaluation.")

        report = EvaluationReport()
        num_q = min(self.num_questions, len(all_docs) // 2)
        total_steps = num_q * 3  # generate + answer + judge

        step = 0
        for i in range(num_q):
            doc1, doc2 = random.sample(all_docs, 2)

            # Step 1: Generate synthetic QA
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"Generating question {i+1}/{num_q}…")

            qa = self._generate_synthetic_qa(doc1, doc2)
            if qa is None:
                continue

            logger.info(f"Q{i+1}: {qa.question[:80]}…")

            # Step 2: Get RAG answer
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"Getting RAG answer {i+1}/{num_q}…")

            rag_answer = self._get_rag_answer(qa.question)

            # Step 3: Judge
            step += 1
            if progress_callback:
                progress_callback(step, total_steps, f"Judging answer {i+1}/{num_q}…")

            judge_output, score = self._judge(qa, rag_answer)
            passed = score == 2

            report.results.append(EvalResult(
                question=qa.question,
                synth_answer=qa.synth_answer,
                rag_answer=rag_answer,
                judge_output=judge_output,
                score=score,
                passed=passed,
            ))

            logger.info(f"Q{i+1} → Score: {score} ({'PASS' if passed else 'FAIL'})")

        # Aggregate
        report.num_total = len(report.results)
        report.num_passed = sum(r.passed for r in report.results)
        report.preference_score = (
            report.num_passed / report.num_total if report.num_total > 0 else 0.0
        )

        if report.preference_score >= 0.8:
            report.summary = "🟢 Excellent – RAG pipeline performs strongly on this document."
        elif report.preference_score >= 0.6:
            report.summary = "🟡 Good – RAG pipeline performs adequately. Some answers could be improved."
        elif report.preference_score >= 0.4:
            report.summary = "🟠 Fair – RAG pipeline struggles on some questions. Consider adjusting chunk size or retrieval_k."
        else:
            report.summary = "🔴 Poor – RAG pipeline needs improvement. Review chunking strategy and prompt templates."

        logger.info(
            f"Evaluation complete: {report.num_passed}/{report.num_total} passed "
            f"({report.preference_score:.1%})"
        )
        return report
