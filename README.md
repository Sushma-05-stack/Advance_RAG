# Document-Only Advanced RAG

Streamlit app for document-grounded Q&A with **ChromaDB**, **hybrid search**, **query optimization**, **re-ranking**, and **LangSmith** tracing.

## Features

- Upload PDF, DOCX, TXT, MD (multiple files)
- ChromaDB similarity search + BM25 hybrid search
- Query rewrite & multi-query expansion
- Cross-encoder re-ranking
- Answers **only** from uploaded documents
- LangSmith observability

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
# Add GROQ_API_KEY and optional LANGCHAIN_API_KEY (LangSmith)
```

PDF uploads require **pypdf** (included in `requirements.txt`). If you see a missing package error:

```bash
pip install pypdf
```

Use `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, and `LANGCHAIN_PROJECT` in `.env` (do not use `export` lines — `python-dotenv` ignores them).

## Run

```bash
streamlit run streamlit_app.py
```

1. Upload documents in the sidebar  
2. Click **Build Vector DB**  
3. Ask questions on the **Chat** tab  

## Environment

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Groq LLM (free tier) |
| `LLM_PROVIDER` | `groq`, `mistral`, or `openai` |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith |
| `CHROMA_*` | Optional Chroma Cloud |
"# Advance_RAG" 
