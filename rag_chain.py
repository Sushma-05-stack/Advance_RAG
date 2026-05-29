"""Document-only RAG: optimize → hybrid retrieve → rerank → grounded answer."""
import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import config
from llm import get_chat_llm
from retrieval import HybridRetriever, QueryOptimizer, RetrievalDebugInfo
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a document-only assistant. Answer ONLY using the context below.

RULES:
1. Use ONLY information from the context excerpts.
2. If the answer is not in the context, say exactly:
   "I could not find an answer to your question in the provided documents."
3. Do NOT use outside knowledge.
4. Cite source file names at the end of your answer.
5. Be concise and accurate.

Context:
{context}
"""


class DocumentRAGChain:
    def __init__(self, vsm: VectorStoreManager):
        config.validate()
        self.vsm = vsm
        self.llm = get_chat_llm()
        self.optimizer = QueryOptimizer(self.llm)
        self.retriever = HybridRetriever(vsm, vsm.bm25_index)
        self.answer_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "{question}"),
            ])
            | self.llm
            | StrOutputParser()
        )

    def _format_context(self, docs: List[Tuple[Document, float]]):
        parts, sources = [], []
        for i, (doc, score) in enumerate(docs, 1):
            src = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "")
            page_info = f", page {int(page) + 1}" if page != "" and page is not None else ""
            cid = doc.metadata.get("chunk_id", doc.metadata.get("chunk_index", i))
            parts.append(f"[Excerpt {i} | {src}{page_info} | chunk: {cid}]\n{doc.page_content}")
            sources.append({
                "excerpt_id": i,
                "source_file": src,
                "page": int(page) + 1 if page != "" and page is not None else None,
                "chunk_id": cid,
                "relevance_score": round(float(score), 4),
                "preview": doc.page_content[:200],
            })
        return "\n\n---\n\n".join(parts), sources

    def query(
        self,
        question: str,
        k: Optional[int] = None,
        use_optimization: bool = True,
        use_hybrid: bool = True,
        use_rerank: bool = True,
        debug: Optional[RetrievalDebugInfo] = None,
    ) -> dict:
        if not question.strip():
            return {"error": "Question cannot be empty."}

        k = k or config.TOP_K_RESULTS
        orig = (config.ENABLE_QUERY_OPTIMIZATION, config.ENABLE_HYBRID_SEARCH, config.ENABLE_RERANKING)
        try:
            config.ENABLE_QUERY_OPTIMIZATION = use_optimization
            config.ENABLE_HYBRID_SEARCH = use_hybrid
            config.ENABLE_RERANKING = use_rerank

            queries = self.optimizer.optimize(question) if use_optimization else [question]
            if debug:
                debug.search_queries = queries

            docs = self.retriever.retrieve(
                queries=queries,
                k=k,
                primary_query=queries[0],
                debug=debug,
            )

            if not docs:
                return {
                    "question": question,
                    "search_queries": queries,
                    "answer": "I could not find any relevant information in the uploaded documents.",
                    "sources": [],
                    "num_chunks_retrieved": 0,
                    "retrieval_mode": debug.mode if debug else "",
                }

            context, sources = self._format_context(docs)
            answer = self.answer_chain.invoke({"context": context, "question": question})

            return {
                "question": question,
                "search_queries": queries,
                "answer": answer.strip(),
                "sources": sources,
                "num_chunks_retrieved": len(docs),
                "retrieval_mode": debug.mode if debug else "",
            }
        finally:
            config.ENABLE_QUERY_OPTIMIZATION, config.ENABLE_HYBRID_SEARCH, config.ENABLE_RERANKING = orig
