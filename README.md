# Advanced RAG — Document Q&A

Document-grounded Q&A using **ChromaDB** (vector similarity search) + **OpenAI GPT** (answer generation). Answers come **only** from your uploaded documents — no hallucination from outside knowledge.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://advance-rag.streamlit.app)

---

## Architecture

```
User Question
     │
     ▼
Query Rewriter (LLM)          ← improves retrieval quality
     │
     ▼
ChromaDB Similarity Search    ← finds top-K relevant chunks
     │
     ▼
Context Assembly + Grounding  ← strict "only from docs" prompt
     │
     ▼
OpenAI GPT (gpt-4o-mini)      ← generates cited answer
     │
     ▼
Answer + Source Citations
```

## Features

- **Strict grounding** — LLM answers only from retrieved context
- **Query rewriting** — rewrites questions into better search queries
- **Source citations** — every answer cites the source document and page
- **Relevance scores** — shows how closely each chunk matched the query
- **Multi-format** — PDF, TXT, DOCX, Markdown
- **Persistent store** — ChromaDB persists to disk between sessions
- **Streamlit UI** — clean web interface, deployable to Streamlit Cloud
- **REST API** — FastAPI backend for programmatic access
- **CLI** — command-line interface for scripting

---

## 🚀 Deploy to Streamlit Cloud (Free)

1. **Fork / push this repo to GitHub**

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Select your repo, branch `main`, and set **Main file path** to `streamlit_app.py`

4. Click **Advanced settings → Secrets** and add:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```

5. Click **Deploy** — done!

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
Edit `.env` and set your OpenAI API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run Streamlit locally
```bash
streamlit run streamlit_app.py
```

---

## CLI Usage

```bash
# Ingest a file
python cli.py ingest ./my_document.pdf

# Ingest a whole directory
python cli.py ingest ./documents/

# Ask a question
python cli.py query "What is the refund policy?"

# Retrieve more context
python cli.py query "Summarize the key findings" --k 8

# Filter to one document
python cli.py query "What are the revenue figures?" --source annual_report.pdf

# List ingested documents
python cli.py sources

# Clear the store
python cli.py reset
```

## API Usage

```bash
uvicorn api:app --reload --port 8000
```

Docs at: http://localhost:8000/docs

```bash
# Upload
curl -X POST http://localhost:8000/upload -F "file=@./doc.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the return policy?"}'
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embeddings (no API key) |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Chunks retrieved per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |

## Project Structure

```
Advance_RAG/
├── streamlit_app.py       # Streamlit web UI
├── api.py                 # FastAPI REST API
├── cli.py                 # Command-line interface
├── rag_chain.py           # RAG pipeline
├── vector_store.py        # ChromaDB operations
├── document_processor.py  # Load & chunk documents
├── config.py              # Configuration
├── demo.py                # Quick demo script
├── requirements.txt
├── .env.example
├── .gitignore
├── .streamlit/
│   ├── config.toml        # Streamlit theme
│   └── secrets.toml.example
└── sample_docs/
    └── company_policy.txt
```
