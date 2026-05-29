"""Configuration for Document-Only Advanced RAG."""
import os
from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


class Config:
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_BASE_URL: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "") or os.getenv("LANGSMITH_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "") or os.getenv("LANGSMITH_PROJECT", "document-rag")
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "") or os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )

    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "rag_documents")
    CHROMA_API_KEY: str = os.getenv("CHROMA_API_KEY", "")
    CHROMA_TENANT: str = os.getenv("CHROMA_TENANT", "")
    CHROMA_DATABASE: str = os.getenv("CHROMA_DATABASE", "")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    RETRIEVAL_CANDIDATE_K: int = int(os.getenv("RETRIEVAL_CANDIDATE_K", "20"))

    ENABLE_QUERY_OPTIMIZATION: bool = _bool("ENABLE_QUERY_OPTIMIZATION", True)
    ENABLE_QUERY_REWRITE: bool = _bool("ENABLE_QUERY_REWRITE", True)
    ENABLE_MULTI_QUERY: bool = _bool("ENABLE_MULTI_QUERY", True)
    MULTI_QUERY_COUNT: int = int(os.getenv("MULTI_QUERY_COUNT", "3"))
    ENABLE_HYBRID_SEARCH: bool = _bool("ENABLE_HYBRID_SEARCH", True)
    ENABLE_RERANKING: bool = _bool("ENABLE_RERANKING", True)
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    SUPPORTED_EXTENSIONS: list = [".pdf", ".txt", ".docx", ".md"]

    @property
    def use_chroma_cloud(self) -> bool:
        return bool(self.CHROMA_API_KEY and self.CHROMA_TENANT and self.CHROMA_DATABASE)

    @property
    def llm_api_key(self) -> str:
        if self.LLM_PROVIDER == "openai":
            return self.OPENAI_API_KEY
        if self.LLM_PROVIDER == "mistral":
            return self.MISTRAL_API_KEY
        if self.GROQ_API_KEY:
            return self.GROQ_API_KEY
        if self.OPENAI_API_KEY.startswith("gsk_"):
            return self.OPENAI_API_KEY
        return self.GROQ_API_KEY

    @property
    def llm_base_url(self):
        if self.LLM_BASE_URL:
            return self.LLM_BASE_URL
        if self.LLM_PROVIDER == "groq":
            return self.GROQ_BASE_URL
        if self.LLM_PROVIDER == "mistral":
            return self.MISTRAL_BASE_URL
        return None

    @property
    def llm_display_name(self) -> str:
        names = {"groq": "Groq LLM", "mistral": "Mistral AI", "openai": "OpenAI"}
        return names.get(self.LLM_PROVIDER, "LLM")

    def setup_langsmith(self):
        tracing = (
            _bool("LANGCHAIN_TRACING_V2", False)
            or _bool("LANGSMITH_TRACING", False)
        )
        api_key = self.LANGCHAIN_API_KEY
        if tracing and api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT.strip('"')
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT
            # LangSmith SDK also reads these aliases
            os.environ["LANGSMITH_API_KEY"] = api_key
            os.environ["LANGSMITH_PROJECT"] = self.LANGCHAIN_PROJECT.strip('"')
            os.environ["LANGSMITH_TRACING"] = "true"

    def validate(self):
        if not self.llm_api_key:
            raise ValueError(
                "LLM API key missing. Set GROQ_API_KEY, MISTRAL_API_KEY, or OPENAI_API_KEY in .env"
            )


config = Config()
config.setup_langsmith()
