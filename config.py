"""Configuration for Document-Only Advanced RAG.

All values are read lazily from os.environ so Streamlit secrets
injected before load_components() are always picked up.
"""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class Config:
    """Reads every value fresh from os.environ each time it is accessed."""

    # ── LLM ──────────────────────────────────────────────────────────────────
    @property
    def LLM_PROVIDER(self) -> str:
        return os.environ.get("LLM_PROVIDER", "groq").lower()

    @property
    def GROQ_API_KEY(self) -> str:
        return os.environ.get("GROQ_API_KEY", "")

    @property
    def GROQ_BASE_URL(self) -> str:
        return os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    @property
    def OPENAI_API_KEY(self) -> str:
        return os.environ.get("OPENAI_API_KEY", "")

    @property
    def LLM_MODEL(self) -> str:
        return os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

    @property
    def LLM_TEMPERATURE(self) -> float:
        return _float("LLM_TEMPERATURE", 0.0)

    @property
    def LLM_BASE_URL(self) -> str:
        return os.environ.get("LLM_BASE_URL", "")

    # ── LangSmith ─────────────────────────────────────────────────────────────
    @property
    def LANGCHAIN_API_KEY(self) -> str:
        return os.environ.get("LANGCHAIN_API_KEY", "") or os.environ.get("LANGSMITH_API_KEY", "")

    @property
    def LANGCHAIN_PROJECT(self) -> str:
        return os.environ.get("LANGCHAIN_PROJECT", "") or os.environ.get("LANGSMITH_PROJECT", "document-rag")

    @property
    def LANGCHAIN_ENDPOINT(self) -> str:
        return os.environ.get("LANGCHAIN_ENDPOINT", "") or os.environ.get(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        )

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    @property
    def CHROMA_PERSIST_DIR(self) -> str:
        return os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

    @property
    def CHROMA_COLLECTION_NAME(self) -> str:
        return os.environ.get("CHROMA_COLLECTION_NAME", "rag_documents")

    @property
    def CHROMA_API_KEY(self) -> str:
        return os.environ.get("CHROMA_API_KEY", "")

    @property
    def CHROMA_TENANT(self) -> str:
        return os.environ.get("CHROMA_TENANT", "")

    @property
    def CHROMA_DATABASE(self) -> str:
        return os.environ.get("CHROMA_DATABASE", "")

    # ── Chunking & retrieval ──────────────────────────────────────────────────
    @property
    def EMBEDDING_MODEL(self) -> str:
        return os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    @property
    def CHUNK_SIZE(self) -> int:
        return _int("CHUNK_SIZE", 1000)

    @property
    def CHUNK_OVERLAP(self) -> int:
        return _int("CHUNK_OVERLAP", 200)

    @property
    def TOP_K_RESULTS(self) -> int:
        return _int("TOP_K_RESULTS", 5)

    @property
    def RETRIEVAL_CANDIDATE_K(self) -> int:
        return _int("RETRIEVAL_CANDIDATE_K", 20)

    @property
    def MULTI_QUERY_COUNT(self) -> int:
        return _int("MULTI_QUERY_COUNT", 3)

    # ── Feature flags ─────────────────────────────────────────────────────────
    @property
    def ENABLE_QUERY_OPTIMIZATION(self) -> bool:
        return _bool("ENABLE_QUERY_OPTIMIZATION", True)

    @ENABLE_QUERY_OPTIMIZATION.setter
    def ENABLE_QUERY_OPTIMIZATION(self, value: bool):
        os.environ["ENABLE_QUERY_OPTIMIZATION"] = str(value)

    @property
    def ENABLE_QUERY_REWRITE(self) -> bool:
        return _bool("ENABLE_QUERY_REWRITE", True)

    @property
    def ENABLE_MULTI_QUERY(self) -> bool:
        return _bool("ENABLE_MULTI_QUERY", True)

    @property
    def ENABLE_HYBRID_SEARCH(self) -> bool:
        return _bool("ENABLE_HYBRID_SEARCH", True)

    @ENABLE_HYBRID_SEARCH.setter
    def ENABLE_HYBRID_SEARCH(self, value: bool):
        os.environ["ENABLE_HYBRID_SEARCH"] = str(value)

    @property
    def ENABLE_RERANKING(self) -> bool:
        return _bool("ENABLE_RERANKING", True)

    @ENABLE_RERANKING.setter
    def ENABLE_RERANKING(self, value: bool):
        os.environ["ENABLE_RERANKING"] = str(value)

    SUPPORTED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".md"]

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def use_chroma_cloud(self) -> bool:
        return bool(self.CHROMA_API_KEY and self.CHROMA_TENANT and self.CHROMA_DATABASE)

    @property
    def llm_api_key(self) -> str:
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        if self.GROQ_API_KEY:
            return self.GROQ_API_KEY
        return ""

    @property
    def llm_base_url(self):
        if self.LLM_BASE_URL:
            return self.LLM_BASE_URL
        if self.LLM_PROVIDER == "groq":
            return self.GROQ_BASE_URL
        return None

    def setup_langsmith(self):
        api_key = self.LANGCHAIN_API_KEY
        tracing = (
            _bool("LANGCHAIN_TRACING_V2", False)
            or _bool("LANGSMITH_TRACING", False)
        )
        if tracing and api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT.strip('"')
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGSMITH_PROJECT"] = self.LANGCHAIN_PROJECT.strip('"')
            os.environ["LANGSMITH_TRACING"] = "true"

    def validate(self):
        if not self.llm_api_key:
            raise ValueError(
                "LLM API key missing. Set GROQ_API_KEY in secrets or .env"
            )


config = Config()
