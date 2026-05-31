"""
Streamlit UI for the Advanced RAG system.
Run locally:  streamlit run streamlit_app.py
Deploy:       Push to GitHub → connect to share.streamlit.io
"""
import os
from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Advanced RAG — Document Q&A",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject secrets from Streamlit Cloud into env vars ───────────────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "LANGCHAIN_API_KEY" in st.secrets:
    os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
if "LANGCHAIN_TRACING_V2" in st.secrets:
    os.environ["LANGCHAIN_TRACING_V2"] = st.secrets["LANGCHAIN_TRACING_V2"]
if "LANGCHAIN_PROJECT" in st.secrets:
    os.environ["LANGCHAIN_PROJECT"] = st.secrets["LANGCHAIN_PROJECT"]
# ChromaDB Cloud
if "CHROMA_API_KEY" in st.secrets:
    os.environ["CHROMA_API_KEY"] = st.secrets["CHROMA_API_KEY"]
if "CHROMA_TENANT" in st.secrets:
    os.environ["CHROMA_TENANT"] = st.secrets["CHROMA_TENANT"]
if "CHROMA_DATABASE" in st.secrets:
    os.environ["CHROMA_DATABASE"] = st.secrets["CHROMA_DATABASE"]
if "CHROMA_COLLECTION_NAME" in st.secrets:
    os.environ["CHROMA_COLLECTION_NAME"] = st.secrets["CHROMA_COLLECTION_NAME"]


# ── Lazy imports (avoid crashing before secrets are set) ────────────────────
@st.cache_resource(show_spinner="Loading models & vector store…")
def load_components():
    from config import config
    from document_processor import DocumentProcessor
    from vector_store import VectorStoreManager
    from rag_chain import DocumentRAGChain

    config.validate()
    processor = DocumentProcessor()
    vsm = VectorStoreManager()
    rag = DocumentRAGChain(vsm)
    return processor, vsm, rag


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
    .source-chip {
        display: inline-block;
        background: #e0e7ff;
        color: #3730a3;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.8rem;
        margin: 2px 4px 2px 0;
    }
    .score-good { color: #16a34a; font-weight: 600; }
    .score-ok   { color: #d97706; font-weight: 600; }
    .score-low  { color: #dc2626; font-weight: 600; }
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🔍 Advanced RAG — Document Q&A")
st.caption("Answers come **only** from your uploaded documents. Powered by ChromaDB + Groq LLM.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # API key input (for local use; on Streamlit Cloud use secrets.toml)
    if not os.environ.get("GROQ_API_KEY"):
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            placeholder="gsk_...",
            help="Free key at https://console.groq.com — not stored anywhere.",
        )
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key
            # clear cached components so they reinitialise with the new key
            st.cache_resource.clear()
    else:
        st.success("✅ Groq API key loaded")

    st.divider()

    top_k = st.slider("Chunks to retrieve (k)", min_value=1, max_value=15, value=5)

    st.divider()
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, DOCX, or MD",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("📥 Ingest Documents", use_container_width=True):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please enter your Groq API key first.")
        else:
            try:
                processor, vsm, rag = load_components()
                progress = st.progress(0, text="Ingesting…")
                total_chunks = 0

                for i, uf in enumerate(uploaded_files):
                    try:
                        file_bytes = uf.read()
                        chunks = processor.process_upload(uf.name, file_bytes)
                        ids = vsm.add_documents(chunks)
                        total_chunks += len(ids)
                        st.toast(f"✅ {uf.name} → {len(ids)} chunks")
                    except Exception as upload_err:
                        st.error(f"Failed to process {uf.name}: {upload_err}")

                    progress.progress(
                        (i + 1) / len(uploaded_files),
                        text=f"Processed {i + 1}/{len(uploaded_files)} files",
                    )

                progress.empty()
                st.success(f"Ingested {len(uploaded_files)} file(s) → {total_chunks} chunks total")
                st.rerun()

            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.divider()

    # Document management (only shown once key is present)
    if os.environ.get("GROQ_API_KEY"):
        try:
            _, vsm, _ = load_components()
            sources = vsm.list_sources()
            total_chunks = vsm.get_document_count()

            if sources:
                st.header("📚 Ingested Documents")
                st.caption(f"{total_chunks} total chunks")
                for src in sources:
                    col1, col2 = st.columns([4, 1])
                    col1.markdown(f"📄 `{src}`")
                    if col2.button("🗑️", key=f"del_{src}", help=f"Delete {src}"):
                        vsm.delete_by_source(src)
                        st.toast(f"Deleted {src}")
                        st.rerun()

                st.divider()
                if st.button("🔴 Reset All", use_container_width=True,
                             help="Delete everything from the vector store"):
                    vsm.delete_collection()
                    st.toast("Vector store cleared")
                    st.rerun()
        except Exception:
            pass


# ── Main area ────────────────────────────────────────────────────────────────

if not os.environ.get("GROQ_API_KEY"):
    st.info("👈 Enter your Groq API key in the sidebar to get started.\n\n"
            "Get a free key at https://console.groq.com")
    st.stop()

try:
    processor, vsm, rag = load_components()
except Exception as e:
    st.error(f"Initialization error: {e}")
    st.stop()

doc_count = vsm.get_document_count()
sources = vsm.list_sources()

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="stat-card">📦 <b>{doc_count}</b><br><small>Chunks stored</small></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="stat-card">📄 <b>{len(sources)}</b><br><small>Documents</small></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="stat-card">🔍 <b>{top_k}</b><br><small>Chunks per query</small></div>',
        unsafe_allow_html=True,
    )

st.divider()

if doc_count == 0:
    st.warning("⚠️ No documents ingested yet. Upload files using the sidebar.")
    st.stop()

# Source filter
filter_source = None
if sources:
    filter_options = ["All documents"] + sources
    selected = st.selectbox("🔎 Search within", filter_options, index=0)
    if selected != "All documents":
        filter_source = selected

# Question input
question = st.text_input(
    "💬 Ask a question about your documents",
    placeholder="e.g. What are the main findings?",
)

ask_col, clear_col = st.columns([5, 1])
ask_btn = ask_col.button("🚀 Ask", use_container_width=True, type="primary")
clear_btn = clear_col.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.session_state.pop("history", None)
    st.rerun()

# ── Query & Answer ────────────────────────────────────────────────────────────
if ask_btn and question.strip():
    with st.spinner("Searching documents and generating answer…"):
        result = rag.query(
            question=question,
            k=top_k,
        )

    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.insert(0, result)

elif ask_btn and not question.strip():
    st.warning("Please enter a question.")

# ── Display history ───────────────────────────────────────────────────────────
if "history" in st.session_state and st.session_state.history:
    for idx, result in enumerate(st.session_state.history):
        with st.container():
            st.markdown(f"**❓ {result['question']}**")

            # Show rewritten / expanded queries
            queries = result.get("search_queries", [])
            if queries and queries[0] != result["question"]:
                st.caption(f"🔄 Search queries used: *{' | '.join(queries)}*")

            # Answer
            st.markdown(
                f'<div class="answer-box">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )

            # Sources expander
            if result.get("sources"):
                with st.expander(
                    f"📚 {result['num_chunks_retrieved']} source chunk(s) retrieved",
                    expanded=False,
                ):
                    for src in result["sources"]:
                        score = src["relevance_score"]
                        if score < 0.5:
                            score_class, score_label = "score-good", "High relevance"
                        elif score < 1.0:
                            score_class, score_label = "score-ok", "Medium relevance"
                        else:
                            score_class, score_label = "score-low", "Low relevance"

                        page_info = f" · page {src['page']}" if src.get("page") else ""
                        st.markdown(
                            f"**[{src['excerpt_id']}]** "
                            f'<span class="source-chip">{src["source_file"]}{page_info}</span> '
                            f'<span class="{score_class}">score: {score} ({score_label})</span>',
                            unsafe_allow_html=True,
                        )
                        st.caption(src["preview"])
                        st.divider()

            if idx < len(st.session_state.history) - 1:
                st.divider()
