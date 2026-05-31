"""ChromaDB vector store — uses Groq embeddings API (no local model, no compilation).

Connects to ChromaDB Cloud when CHROMA_API_KEY/TENANT/DATABASE are set,
otherwise falls back to local PersistentClient.
"""
import logging
from typing import List, Optional, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config import config
from retrieval import BM25Index

logger = logging.getLogger(__name__)


def _get_embeddings() -> OpenAIEmbeddings:
    """
    Use Groq's embeddings endpoint (nomic-embed-text-v1.5).
    Groq is OpenAI-compatible so OpenAIEmbeddings works by pointing at Groq's base URL.
    """
    return OpenAIEmbeddings(
        api_key=config.GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
        model="nomic-embed-text-v1.5",
    )


class VectorStoreManager:
    def __init__(self):
        self.embeddings = _get_embeddings()
        self.bm25_index = BM25Index()
        self.vector_store: Optional[Chroma] = None
        self._init_store()
        self._sync_bm25()

    def _client(self):
        if config.use_chroma_cloud:
            logger.info("Connecting to ChromaDB Cloud (tenant=%s, db=%s)",
                        config.CHROMA_TENANT, config.CHROMA_DATABASE)
            return chromadb.CloudClient(
                tenant=config.CHROMA_TENANT,
                database=config.CHROMA_DATABASE,
                api_key=config.CHROMA_API_KEY,
            )
        logger.info("Using local ChromaDB at %s", config.CHROMA_PERSIST_DIR)
        return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)

    def _init_store(self):
        self.vector_store = Chroma(
            client=self._client(),
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    def _sync_bm25(self):
        if not config.ENABLE_HYBRID_SEARCH:
            return
        try:
            result = self.vector_store._collection.get(include=["documents", "metadatas"])
            docs = result.get("documents") or []
            metas = result.get("metadatas") or []
            if docs:
                self.bm25_index.rebuild([
                    Document(page_content=t, metadata=m or {})
                    for t, m in zip(docs, metas)
                ])
        except Exception as e:
            logger.warning("BM25 sync failed: %s", e)

    def add_documents(self, chunks: List[Document]) -> List[str]:
        if not chunks:
            return []
        ids = self.vector_store.add_documents(chunks)
        self._sync_bm25()
        return ids

    def similarity_search(self, query: str, k: int, filter_metadata: Optional[dict] = None):
        return self.vector_store.similarity_search(query=query, k=k, filter=filter_metadata)

    def similarity_search_with_scores(self, query: str, k: int, filter_metadata: Optional[dict] = None):
        return self.vector_store.similarity_search_with_score(query=query, k=k, filter=filter_metadata)

    def get_document_count(self) -> int:
        try:
            return self.vector_store._collection.count()
        except Exception:
            return 0

    def list_sources(self) -> List[str]:
        try:
            result = self.vector_store._collection.get(include=["metadatas"])
            return sorted({
                m["source_file"] for m in result.get("metadatas", []) if m and "source_file" in m
            })
        except Exception:
            return []

    def get_all_metadata(self) -> List[dict]:
        try:
            result = self.vector_store._collection.get(include=["metadatas", "documents"])
            rows = []
            for meta, text in zip(result.get("metadatas", []), result.get("documents", [])):
                rows.append({**(meta or {}), "preview": (text or "")[:120]})
            return rows
        except Exception:
            return []

    def delete_by_source(self, source_file: str):
        try:
            result = self.vector_store._collection.get(
                where={"source_file": source_file},
                include=["metadatas"],
            )
            ids = result.get("ids", [])
            if ids:
                self.vector_store._collection.delete(ids=ids)
                logger.info("Deleted %d chunks from '%s'", len(ids), source_file)
                self._sync_bm25()
            else:
                logger.warning("No chunks found for source: %s", source_file)
        except Exception as e:
            logger.error("Failed to delete source '%s': %s", source_file, e)

    def delete_collection(self):
        self.clear()

    def clear(self):
        self.vector_store.delete_collection()
        self.bm25_index = BM25Index()
        self._init_store()
