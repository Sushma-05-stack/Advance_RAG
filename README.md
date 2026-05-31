# Advanced RAG — Document Q&A

Document-grounded Q&A using **ChromaDB** (vector similarity search) + **Groq LLM** (fast, free inference). Answers come **only** from your uploaded documents — no hallucination from outside knowledge.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://advance-rag.streamlit.app)

---

## Architecture

```
User Question
     │
     ▼
Query Rewriter + Multi-Query (Groq LLM)   ← improves retrieval quality
     │
     ▼
ChromaDB Cloud Similarity Search          ← dense vector search
     +
BM25 Keyword Search                       ← sparse keyword search
     │
     ▼
Reciprocal Rank Fusion + Reranking        ← merges & reranks results
     │
     ▼
Context Assembly + Grounding              ← strict "only from docs" prompt
     │
     ▼
Groq LLM (llama-3.3-70b-versatile)       ← generates cited answer
     │
     ▼
Answer + Source Citations
```

## Features

- **Strict grounding** — LLM answers only from retrieved context
- **Query rewriting + multi-query** — generates multiple search queries for better recall
- **Hybrid search** — combines dense vector search (ChromaDB) + BM25 keyword search
- **Reciprocal Rank Fusion** — merges results from multiple queries
- **Source citations** — every answer cites the source document and page
- **Relevance scores** — shows how closely each chunk matched the query
- **Multi-format** — PDF, TXT, DOCX, Markdown
- **ChromaDB Cloud** — persistent vector store, no local disk needed
- **Streamlit UI** — clean web interface deployed on Streamlit Cloud

---

## 🚀 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Select your repo, branch `main`, file path `streamlit_app.py`

4. Click **Advanced settings → Secrets** and paste:
   ```toml
   LLM_PROVIDER = "groq"
   GROQ_API_KEY = "gsk_your_groq_key_here"
   GROQ_BASE_URL = "https://api.groq.com/openai/v1"
   LLM_MODEL = "llama-3.3-70b-versatile"

   CHROMA_API_KEY = "ck_your_chroma_cloud_key"
   CHROMA_TENANT = "your_tenant_id"
   CHROMA_DATABASE = "your_database_name"
   CHROMA_COLLECTION_NAME = "rag_documents"
   ```

5. Click **Deploy**

> Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## Local Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
copy .env.example .env   # Windows
cp .env.example .env     # Mac/Linux
```

Edit `.env` and set your Groq API key:
```
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
```

### 3. Run Streamlit locally
```bash
streamlit run streamlit_app.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Free at console.groq.com |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Groq model |
| `CHROMA_API_KEY` | — | ChromaDB Cloud key |
| `CHROMA_TENANT` | — | ChromaDB Cloud tenant ID |
| `CHROMA_DATABASE` | — | ChromaDB Cloud database name |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |

## Project Structure

```
Advance_RAG/
├── streamlit_app.py       # Streamlit web UI
├── rag_chain.py           # RAG pipeline
├── retrieval.py           # Hybrid search + reranking
├── vector_store.py        # ChromaDB operations
├── document_processor.py  # Load & chunk documents
├── llm.py                 # LLM factory (Groq)
├── config.py              # Configuration
├── requirements.txt
├── .env.example
├── .gitignore
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```
