"""Query optimization, BM25, hybrid search, and re-ranking."""
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from config import config

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """Rewrite this question into a keyword-rich search query for document retrieval.
Return ONLY the rewritten query.

Question: {question}
Rewritten:"""

MULTI_QUERY_PROMPT = """Generate {num_queries} different search queries for this question.
One per line, no numbering.

Question: {question}
Queries:"""


@dataclass
class RetrievalDebugInfo:
    search_queries: List[str] = field(default_factory=list)
    chunks: List[dict] = field(default_factory=list)
    retrieval_time_ms: float = 0.0
    top_k: int = 5
    mode: str = ""


class QueryOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self._rewrite = ChatPromptTemplate.from_template(REWRITE_PROMPT) | llm | StrOutputParser()
        self._multi = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT) | llm | StrOutputParser()

    def optimize(self, question: str) -> List[str]:
        if not config.ENABLE_QUERY_OPTIMIZATION:
            return [question]

        queries: List[str] = []
        if config.ENABLE_QUERY_REWRITE:
            try:
                q = self._rewrite.invoke({"question": question}).strip()
                if q:
                    queries.append(q)
            except Exception as e:
                logger.warning("Rewrite failed: %s", e)

        if config.ENABLE_MULTI_QUERY:
            try:
                raw = self._multi.invoke({"question": question, "num_queries": config.MULTI_QUERY_COUNT})
                for line in raw.strip().splitlines():
                    line = line.strip().lstrip("0123456789.-) ")
                    if len(line) > 3:
                        queries.append(line)
            except Exception as e:
                logger.warning("Multi-query failed: %s", e)

        seen: Set[str] = set()
        unique = []
        for q in [question] + queries:
            k = q.lower().strip()
            if k not in seen:
                seen.add(k)
                unique.append(q)
        return unique[: config.MULTI_QUERY_COUNT + 1]


class BM25Index:
    def __init__(self):
        self._docs: List[Document] = []
        self._bm25: Optional[BM25Okapi] = None

    def rebuild(self, documents: List[Document]):
        self._docs = list(documents)
        tok = [d.page_content.lower().split() for d in self._docs]
        self._bm25 = BM25Okapi(tok) if tok else None

    def search(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        return [(self._docs[i], float(s)) for i, s in ranked if s > 0]


def _doc_key(doc: Document) -> str:
    return f"{doc.metadata.get('source_file','')}::{doc.metadata.get('chunk_index','')}::{hash(doc.page_content[:200])}"


def reciprocal_rank_fusion(lists: List[List[Document]], k: int = 60) -> List[Tuple[Document, float]]:
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    for results in lists:
        for rank, doc in enumerate(results, 1):
            key = _doc_key(doc)
            scores[key] += 1.0 / (k + rank)
            doc_map[key] = doc
    return [(doc_map[key], s) for key, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class Reranker:
    def __init__(self):
        self._model: Optional[CrossEncoder] = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(config.RERANKER_MODEL)
        return self._model

    def rerank(self, query: str, candidates: List[Tuple[Document, float]], top_n: int):
        if not candidates:
            return []
        pairs = [(query, d.page_content) for d, _ in candidates]
        scores = self._load().predict(pairs)
        ranked = sorted(zip([d for d, _ in candidates], scores), key=lambda x: float(x[1]), reverse=True)
        return [(d, float(s)) for d, s in ranked[:top_n]]


class HybridRetriever:
    def __init__(self, vsm, bm25: BM25Index):
        self.vsm = vsm
        self.bm25 = bm25
        self.reranker = Reranker() if config.ENABLE_RERANKING else None

    def retrieve(
        self,
        queries: List[str],
        k: int,
        filter_metadata: Optional[dict] = None,
        primary_query: Optional[str] = None,
        debug: Optional[RetrievalDebugInfo] = None,
    ) -> List[Tuple[Document, float]]:
        t0 = time.perf_counter()
        fetch_k = max(k, config.RETRIEVAL_CANDIDATE_K)
        primary = primary_query or queries[0]
        lists: List[List[Document]] = []

        for query in queries:
            if config.ENABLE_HYBRID_SEARCH:
                lists.append(self.vsm.similarity_search(query, fetch_k, filter_metadata))
                sparse = self.bm25.search(query, fetch_k)
                if sparse:
                    lists.append([d for d, _ in sparse])
            else:
                dense = self.vsm.similarity_search_with_scores(query, fetch_k, filter_metadata)
                lists.append([d for d, _ in dense])

        if not lists:
            if debug:
                debug.retrieval_time_ms = (time.perf_counter() - t0) * 1000
            return []

        merged = reciprocal_rank_fusion(lists)[:fetch_k]
        if self.reranker and config.ENABLE_RERANKING:
            result = self.reranker.rerank(primary, merged, k)
        else:
            result = merged[:k]

        if debug:
            debug.retrieval_time_ms = (time.perf_counter() - t0) * 1000
            debug.top_k = k
            debug.search_queries = queries
            debug.mode = self._mode_label()
            for doc, score in result:
                debug.chunks.append({
                    "content": doc.page_content,
                    "source_file": doc.metadata.get("source_file", "Unknown"),
                    "page": doc.metadata.get("page"),
                    "chunk_id": doc.metadata.get("chunk_id", doc.metadata.get("chunk_index")),
                    "similarity_score": round(float(score), 4),
                    "metadata": dict(doc.metadata),
                })
        return result

    @staticmethod
    def _mode_label() -> str:
        parts = []
        if config.ENABLE_QUERY_OPTIMIZATION:
            parts.append("query-optimized")
        parts.append("hybrid" if config.ENABLE_HYBRID_SEARCH else "vector")
        if config.ENABLE_RERANKING:
            parts.append("reranked")
        return "+".join(parts)
