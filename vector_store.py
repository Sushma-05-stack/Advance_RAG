"""ChromaDB vector store with BM25 sync for hybrid search."""
import logging
from typing import List, Optional, Tuple

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from config import config
from retrieval import BM25Index

logger = logging.getLogger(__name__)


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.bm25_index = BM25Index()
        self.vector_store: Optional[Chroma] = None
        self._init_store()
        self._sync_bm25()

    def _client(self):
        if config.use_chroma_cloud:
            return chromadb.CloudClient(
                tenant=config.CHROMA_TENANT,
                database=config.CHROMA_DATABASE,
                api_key=config.CHROMA_API_KEY,
            )
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

    def clear(self):
        self.vector_store.delete_collection()
        self.bm25_index = BM25Index()
        self._init_store()
