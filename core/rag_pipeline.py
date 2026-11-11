"""
core/rag_pipeline.py — Complete RAG pipeline with hybrid reranking,
relevance assessment, source attribution, and hallucination control.

Connects:
1. Embedding generation (Sentence Transformers all-MiniLM-L6-v2)
2. Vector Store (ChromaDB session-scoped filtering)
3. Dynamic Retrieval Decisions (core.rag_context)
4. Hybrid Relevance Assessment & Reranking (rejecting irrelevant chunks)
5. Grounded LLM Generation via Gemini & LCEL (StrOutputParser)

Public API
----------
RAGPipeline.ask(question, doc_id, top_k=None, llm=None) -> RAGResponse
    Executes the full end-to-end RAG pipeline for a user question.

RAGPipeline.decide_and_retrieve(question, doc_id, top_k=None)
    -> tuple[list[Document], RetrievalAssessment, RetrievalDecision]
    Retrieval and assessment step without invoking the LLM.
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config.settings import settings
from core.llm import get_llm, DailyQuotaExceededError
from core.rag_context import (
    decide_retrieval,
    RetrievalDecision,
    InsufficientContextError,
    DocumentNotIngestedError,
    normalize_agent_name,
)
from core.vector_store import (
    similarity_search_with_score,
    is_document_ingested,
)
from schemas.rag_schema import RAGResponse, RetrievalAssessment
from schemas.summary_schema import RetrievedSource

logger = logging.getLogger(__name__)

# Standard stop words ignored during keyword extraction and lexical overlap
_STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "on", "of", "for", "to",
    "and", "or", "did", "does", "do", "how", "why", "who", "which", "where",
    "when", "was", "were", "been", "being", "have", "has", "had", "can",
    "could", "should", "would", "will", "shall", "explain", "describe",
    "detail", "paper", "research", "authors", "report", "proposed", "method",
    "give", "tell", "show", "please", "about", "used", "use", "using"
}

# ── Explicit System Prompt with Strict Hallucination Control ──────────────────
_RAG_SYSTEM_PROMPT = """\
You are an expert academic AI research assistant analyzing an uploaded scientific research paper.
Your task is to answer the user's question using ONLY the provided RETRIEVED PAPER CONTEXT.

STRICT GROUNDING & HALLUCINATION-CONTROL RULES:
1. STRICT GROUNDING: Base your answer strictly and exclusively on the facts, methodology, datasets, and experimental results contained in the RETRIEVED PAPER CONTEXT below.
2. SOURCE ATTRIBUTION: Whenever citing specific findings, architectures, or metrics, explicitly include the source page number in square brackets, for example: [Page 2] or [Page 3], matching the [Page X] tags in the context.
3. ABSOLUTE HONESTY ON MISSING INFORMATION:
   - If the retrieved context DOES NOT contain the answer, or if the user asks about an unsupported topic (e.g. an unmentioned dataset, audio, or a different domain), you MUST explicitly state:
     "The provided research paper does not contain sufficient information to answer this question."
   - Explain what related topics or benchmarks the paper actually covers instead.
   - NEVER invent or hallucinate metrics, baselines, formulas, authors, or findings.
4. HONEST UNCERTAINTY: Clearly distinguish direct empirical findings from reasonable technical interpretations. Never present speculation as an established fact.
5. CONCISE & ACADEMIC: Maintain a professional, college-level academic tone.
"""

_RAG_HUMAN_PROMPT = """\
RETRIEVED PAPER CONTEXT:
{context}

USER QUESTION:
{question}

Provide your grounded answer below:
"""


def _extract_content_keywords(text: str) -> list[str]:
    """Extract lowercase content words (length >= 3) excluding standard stopwords."""
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


class RAGPipeline:
    """
    End-to-end Agentic RAG pipeline for research paper QA.

    Designed for academic rigor, transparent retrieval decisions,
    explainable relevance assessment, and strict hallucination prevention.
    """

    @classmethod
    def build_chain(cls, llm=None):
        """Construct the LCEL prompt | llm | StrOutputParser chain."""
        active_llm = llm if llm is not None else get_llm()
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _RAG_SYSTEM_PROMPT),
                ("human", _RAG_HUMAN_PROMPT),
            ]
        )
        return prompt | active_llm | StrOutputParser()

    @classmethod
    def decide_and_retrieve(
        cls,
        question: str,
        doc_id: str,
        top_k: int | None = None,
    ) -> tuple[list[Document], RetrievalAssessment, RetrievalDecision]:
        """
        Execute dynamic retrieval decision, hybrid scoring, and relevance evaluation.

        Parameters
        ----------
        question : str
            User query.
        doc_id : str
            Session-scoped document identifier.
        top_k : int | None
            Number of chunks to return. Defaults to settings.RAG_TOP_K.

        Returns
        -------
        tuple[list[Document], RetrievalAssessment, RetrievalDecision]
        """
        if not doc_id or not doc_id.strip():
            raise DocumentNotIngestedError(
                "No active document loaded in session. Please upload a PDF before asking questions."
            )
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace.")

        k = top_k if top_k is not None else settings.RAG_TOP_K
        keywords = _extract_content_keywords(question)

        # ── Step 1: Retrieval Decision ───────────────────────────────────────
        decision = decide_retrieval("extract_concepts", question, doc_id)

        # ── Step 2: Fetch Candidates with Distance Scores ────────────────────
        # For QA, searching the focused user question avoids diluting the embedding
        # with generic domain terms, while hybrid reranking scores topical relevance.
        candidate_k = max(k * 3, 8)
        candidates = similarity_search_with_score(
            query=question,
            doc_id=doc_id,
            top_k=candidate_k,
        )

        # ── Step 3: Hybrid Scoring & Reranking ───────────────────────────────
        scored_chunks = []
        for doc, dist in candidates:
            # Semantic similarity derived from squared L2 distance on unit vectors
            sem_sim = max(0.0, 1.0 - (dist / 2.0))

            # Lexical overlap score
            text_lower = doc.page_content.lower()
            if keywords:
                matches = sum(1 for kw in keywords if kw in text_lower)
                lex_score = matches / len(keywords)
            else:
                lex_score = 0.5

            # Composite hybrid score (50% semantic + 50% keyword presence)
            composite = (0.5 * sem_sim) + (0.5 * lex_score)
            scored_chunks.append({
                "doc": doc,
                "dist": dist,
                "sem_sim": sem_sim,
                "lex_score": lex_score,
                "composite": composite,
            })

        # Sort descending by composite relevance
        scored_chunks.sort(key=lambda x: x["composite"], reverse=True)

        # ── Step 4: Reject Clearly Irrelevant Chunks ─────────────────────────
        # Reject chunks where composite score is low (< 0.25)
        accepted = [
            item for item in scored_chunks
            if item["composite"] >= 0.25
        ][:k]

        refinement_triggered = False

        # ── Step 5: Query Refinement (if initial retrieval is weak) ──────────
        if not accepted or (accepted and accepted[0]["composite"] < settings.RAG_RELEVANCE_THRESHOLD):
            logger.info("RAGPipeline: Initial retrieval weak. Triggering query refinement.")
            refinement_triggered = True

            # Refinement: focus query strictly on extracted keywords
            refined_query = " ".join(keywords) if keywords else question
            refined_candidates = similarity_search_with_score(
                query=refined_query,
                doc_id=doc_id,
                top_k=candidate_k,
            )

            refined_scored = []
            for doc, dist in refined_candidates:
                sem_sim = max(0.0, 1.0 - (dist / 2.0))
                text_lower = doc.page_content.lower()
                matches = sum(1 for kw in keywords if kw in text_lower) if keywords else 0
                lex_score = (matches / len(keywords)) if keywords else 0.5
                composite = (0.5 * sem_sim) + (0.5 * lex_score)
                refined_scored.append({
                    "doc": doc,
                    "dist": dist,
                    "sem_sim": sem_sim,
                    "lex_score": lex_score,
                    "composite": composite,
                })
            refined_scored.sort(key=lambda x: x["composite"], reverse=True)

            if refined_scored and refined_scored[0]["composite"] > (accepted[0]["composite"] if accepted else 0.0):
                accepted = [
                    item for item in refined_scored
                    if item["composite"] >= 0.25
                ][:k]
                decision.strategy = "user_refined"
                decision.reason += " (query refined via keyword focus)"

        # ── Step 6: Construct Assessment Metrics ─────────────────────────────
        final_docs = [item["doc"] for item in accepted]
        if accepted:
            avg_dist = sum(item["dist"] for item in accepted) / len(accepted)
            max_comp = accepted[0]["composite"]
            avg_comp = sum(item["composite"] for item in accepted) / len(accepted)
            relevance_score = round(min(1.0, (0.7 * max_comp) + (0.3 * avg_comp)), 2)

            # Determine sufficiency:
            # Context is sufficient if:
            # 1. Relevance score exceeds threshold
            # 2. At least 50% of the query's distinct content keywords match the top chunk
            top_lex_score = accepted[0]["lex_score"]
            is_sufficient = (
                relevance_score >= settings.RAG_RELEVANCE_THRESHOLD
                and (not keywords or top_lex_score >= 0.50)
            )
            explanation = (
                f"Accepted {len(accepted)} chunks with top composite score {max_comp:.2f} "
                f"(Semantic: {accepted[0]['sem_sim']:.2f}, Lexical: {accepted[0]['lex_score']:.2f})."
            )
        else:
            avg_dist = 2.0
            relevance_score = 0.0
            is_sufficient = False
            explanation = "No chunks met the minimum relevance threshold."

        assessment = RetrievalAssessment(
            chunk_count_retrieved=len(candidates),
            chunk_count_accepted=len(accepted),
            average_distance=round(avg_dist, 3),
            relevance_score=relevance_score,
            is_sufficient=is_sufficient,
            refinement_triggered=refinement_triggered,
            explanation=explanation,
        )

        return final_docs, assessment, decision

    @classmethod
    def ask(
        cls,
        question: str,
        doc_id: str,
        top_k: int | None = None,
        llm=None,
    ) -> RAGResponse:
        """
        Execute end-to-end RAG question answering.

        Parameters
        ----------
        question : str
            The user's natural language question.
        doc_id : str
            Active session-scoped document ID.
        top_k : int | None
            Number of chunks to retrieve.
        llm : Any | None
            Optional LLM override (e.g. for testing with mocks).

        Returns
        -------
        RAGResponse
        """
        # Validate inputs
        if not question or not question.strip():
            raise ValueError("Question cannot be empty or whitespace.")
        if not doc_id or not doc_id.strip():
            raise DocumentNotIngestedError("No active document loaded in session.")

        # Check for conversational / greeting inputs
        clean_q = re.sub(r"[^\w\s]", "", question.strip().lower())
        conversational_triggers = {"hi", "hello", "hey", "who are you", "what can you do", "help", "greetings"}
        is_greeting = (
            clean_q in conversational_triggers
            or any(clean_q.startswith(t) for t in ["hello", "hi ", "hey "])
            or clean_q.startswith("who are you")
        )
        if is_greeting:
            decision = RetrievalDecision(
                needs_retrieval=False,
                agent_name="conversational",
                constructed_query="",
                strategy="broad_fallback",
                doc_id=doc_id,
                reason="Conversational greeting does not require document retrieval.",
            )
            assessment = RetrievalAssessment(
                chunk_count_retrieved=0,
                chunk_count_accepted=0,
                average_distance=0.0,
                relevance_score=1.0,
                is_sufficient=True,
                refinement_triggered=False,
                explanation="Conversational query handled without retrieval.",
            )
            return RAGResponse(
                question=question,
                answer=(
                    "Hello! I am your AI Research Assistant. You have an active research paper "
                    "loaded in this session. Ask me about the paper's problem statement, proposed "
                    "architecture, methodology, datasets, results, limitations, or future work!"
                ),
                sources=[],
                assessment=assessment,
                strategy_used="direct_no_retrieval",
                is_grounded=False,
                has_sufficient_context=True,
                error=None,
            )

        # ── Step 1 & 2: Retrieval and Relevance Assessment ───────────────────
        try:
            docs, assessment, decision = cls.decide_and_retrieve(question, doc_id, top_k=top_k)
        except Exception as exc:
            logger.error("RAGPipeline retrieval error: %s", exc)
            return RAGResponse(
                question=question,
                answer="Failed to retrieve context from the document.",
                sources=[],
                assessment=RetrievalAssessment(
                    chunk_count_retrieved=0,
                    chunk_count_accepted=0,
                    average_distance=2.0,
                    relevance_score=0.0,
                    is_sufficient=False,
                    refinement_triggered=False,
                    explanation=f"Retrieval error: {exc}",
                ),
                strategy_used="failed",
                is_grounded=False,
                has_sufficient_context=False,
                error=str(exc),
            )

        # ── Step 3: Format Context and Extract Sources ───────────────────────
        sources: list[RetrievedSource] = []
        context_parts: list[str] = []

        seen_sources = set()
        for i, doc in enumerate(docs, start=1):
            p = doc.metadata.get("page_number", 1)
            idx = doc.metadata.get("chunk_index", 0)
            src_file = doc.metadata.get("source", "document.pdf")

            context_parts.append(
                f"[Chunk {i} | Page {p} | Source: {src_file}]\n{doc.page_content.strip()}"
            )
            if (p, idx) not in seen_sources:
                seen_sources.add((p, idx))
                sources.append(RetrievedSource(page_number=p, chunk_index=idx, source=src_file))

        sources.sort(key=lambda s: (s.page_number, s.chunk_index))

        if not context_parts or not assessment.is_sufficient:
            context_str = (
                "[INSUFFICIENT CONTEXT WARNING: The retrieved context does not contain sufficient "
                "evidence to answer this specific query. Explicitly inform the user that the paper "
                "does not provide this information. Do not hallucinate.]\n\n"
            )
            if context_parts:
                context_str += "\n\n---\n\n".join(context_parts)
            else:
                context_str += "[No matching chunks found in the uploaded paper.]"
        else:
            context_str = "\n\n---\n\n".join(context_parts)

        # ── Step 4: Execute LCEL Generation Chain ────────────────────────────
        chain = cls.build_chain(llm=llm)
        try:
            raw_answer = chain.invoke({"context": context_str, "question": question})
            answer = raw_answer.strip()
        except DailyQuotaExceededError as q_err:
            logger.error("RAGPipeline daily quota exceeded: %s", q_err)
            delay_text = f" (Retry available in {q_err.retry_delay})" if q_err.retry_delay else ""
            return RAGResponse(
                question=question,
                answer=(
                    f"⚠️ **Gemini Free-Tier Daily Quota Exceeded (Limit: 20 requests/day)**\n\n"
                    f"The daily request quota for your Google Gemini API key has been exhausted.{delay_text}\n\n"
                    f"Please wait for your daily quota to reset or upgrade to Pay-As-You-Go in Google AI Studio."
                ),
                sources=[],
                assessment=assessment,
                strategy_used=decision.strategy,
                is_grounded=False,
                has_sufficient_context=False,
                error=str(q_err),
            )
        except Exception as exc:
            logger.error("RAGPipeline LLM generation failed: %s", exc)
            return RAGResponse(
                question=question,
                answer="Generation failed due to an LLM API error. Please check your API credentials.",
                sources=sources if assessment.is_sufficient else [],
                assessment=assessment,
                strategy_used=decision.strategy,
                is_grounded=False,
                has_sufficient_context=assessment.is_sufficient,
                error=f"LLM API error: {exc}",
            )

        # ── Step 5: Post-Processing & Hallucination Auditing ──────────────────
        # If the LLM indicates that the paper lacks the information, do not claim grounding
        unsupported_markers = [
            "does not contain sufficient information",
            "does not mention",
            "not mentioned in the paper",
            "does not contain information",
            "no information is provided",
        ]
        is_unsupported = any(m in answer.lower() for m in unsupported_markers)

        final_is_grounded = assessment.is_sufficient and not is_unsupported
        final_sources = sources if final_is_grounded else []

        return RAGResponse(
            question=question,
            answer=answer,
            sources=final_sources,
            assessment=assessment,
            strategy_used=decision.strategy,
            is_grounded=final_is_grounded,
            has_sufficient_context=assessment.is_sufficient and not is_unsupported,
            error=None,
        )
