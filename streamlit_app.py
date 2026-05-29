"""
Document-Only Advanced RAG — Streamlit UI
Run:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import List, Optional

import streamlit as st

from config import config
from document_processor import DocumentProcessor
from rag_chain import DocumentRAGChain
from retrieval import RetrievalDebugInfo
from vector_store import VectorStoreManager
from evaluation import RAGEvaluator

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document-Only RAG System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    body { background: #000000; color: #ffffff; }
    [data-testid="stAppViewContainer"] { background: #000000 !important; }
    .main-title  { font-size:2.1rem; font-weight:800; color:#dbeafe; margin:0; }
    .sub-title   { color:#93c5fd; font-size:1rem; margin-bottom:1.2rem; }
    .status-ok   { color:#34d399; font-weight:700; }
    .status-warn { color:#fbbf24; font-weight:700; }
    .badge {
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:.82rem; font-weight:600;
    }
    .badge-green { background:#052e16; color:#6ee7b7; border:1px solid #0f5132; }
    .badge-amber { background:#3b2f00; color:#fbbf24; border:1px solid #b45309; }
    .badge-blue  { background:#1e3a8a; color:#bfdbfe; border:1px solid #1d4ed8; }
    .hist-q { font-weight:700; color:#dbeafe; margin-bottom:.2rem; }
    .hist-a { background:#050505; border-left:4px solid #2563eb;
              padding:.6rem .9rem; border-radius:4px; margin-bottom:.4rem; }
    .arch-step {
        display:flex; align-items:center; gap:.7rem;
        background:#0b0b0b; border:1px solid #2a2a2a;
        border-radius:8px; padding:.55rem .9rem; margin:.35rem 0;
        font-size:.95rem;
        color:#ffffff;
    }
    .arch-arrow { color:#6b7a8d; font-size:1.1rem; text-align:center; }
    .metric-card {
        background:linear-gradient(135deg,#070707,#0f0f0f);
        border:1px solid #2a2a2a; border-radius:10px;
        padding:.85rem 1rem; margin-bottom:.5rem;
        font-size:.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── session defaults ─────────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "uploaded_files": [],
    "db_ready": False,
    "vsm": None,
    "rag": None,
    "last_updated": None,
    "last_debug": None,
    "indexed_doc_names": [],
    "chat_history": [],          # list of {question, answer, sources, queries, mode}
    "pending_question": "",      # carries question across rerun
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_vector_store() -> VectorStoreManager:
    return VectorStoreManager()


def refresh_stats() -> int:
    vsm = st.session_state.vsm or get_vector_store()
    st.session_state.vsm = vsm
    count = vsm.get_document_count()
    st.session_state.db_ready = count > 0
    return count


def build_database(chunk_size: int, chunk_overlap: int, bar, txt):
    vsm = get_vector_store()
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    files = st.session_state.uploaded_files
    if not files:
        raise ValueError("No files selected.")

    vsm.clear()
    all_chunks, doc_names = [], []
    total = len(files)

    for i, uf in enumerate(files):
        txt.text(f"Reading {uf.name}  ({i + 1}/{total})")
        bar.progress(int(i / total * 80))
        chunks = processor.process_upload(uf.name, uf.getvalue())
        all_chunks.extend(chunks)
        doc_names.append(uf.name)

    txt.text(f"Embedding {len(all_chunks)} chunks into ChromaDB …")
    bar.progress(85)
    batch = 50
    for start in range(0, len(all_chunks), batch):
        vsm.add_documents(all_chunks[start : start + batch])
        pct = 85 + int(min(1.0, (start + batch) / len(all_chunks)) * 15)
        bar.progress(pct)

    bar.progress(100)
    txt.text("Done!")
    st.session_state.vsm = vsm
    st.session_state.rag = DocumentRAGChain(vsm)
    st.session_state.db_ready = True
    st.session_state.indexed_doc_names = doc_names
    st.session_state.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return len(all_chunks), len(doc_names)


def clear_database():
    vsm = st.session_state.vsm or get_vector_store()
    vsm.clear()
    get_vector_store.clear()
    st.session_state.vsm = get_vector_store()
    st.session_state.rag = None
    st.session_state.db_ready = False
    st.session_state.indexed_doc_names = []
    st.session_state.chat_history = []
    st.session_state.last_debug = None
    st.session_state.pending_question = ""


def run_query(question: str, top_k: int) -> Optional[dict]:
    try:
        config.validate()
        vsm = st.session_state.vsm or get_vector_store()
        if st.session_state.rag is None:
            st.session_state.rag = DocumentRAGChain(vsm)

        debug = RetrievalDebugInfo()
        with st.spinner("Searching and generating answer …"):
            result = st.session_state.rag.query(
                question=question,
                k=top_k,
                use_optimization=st.session_state.get("opt_query", True),
                use_hybrid=st.session_state.get("opt_hybrid", True),
                use_rerank=st.session_state.get("opt_rerank", True),
                debug=debug,
            )
        st.session_state.last_debug = debug
        return result
    except Exception as exc:
        msg = str(exc)
        st.error(f"Error: {exc}")

        # Common case: LangSmith auth / 401 invalid API key
        if ("401" in msg) or ("invalid_api_key" in msg) or ("Invalid API Key" in msg):
            with st.expander("Fix authentication (one click)"):
                st.warning("This looks like an authentication issue (often LangSmith tracing).")
                if st.button("Disable LangSmith tracing and retry", type="primary"):
                    import os

                    os.environ["LANGSMITH_TRACING"] = "false"
                    os.environ["LANGCHAIN_TRACING_V2"] = "false"
                    st.session_state.last_debug = None
                    st.rerun()
        return None


# ─── sidebar ──────────────────────────────────────────────────────────────────
def render_sidebar() -> int:
    with st.sidebar:
        st.markdown("## 📄 Document RAG")
        st.caption("Upload documents, build the index, then ask questions.")
        st.divider()

        # Upload
        st.markdown("### 📁 Upload Documents")
        uploaded = st.file_uploader(
            "PDF · DOCX · TXT · MD",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            st.session_state.uploaded_files = uploaded
            st.success(f"{len(uploaded)} file(s) ready")

        st.divider()

        # DB controls
        st.markdown("### 🗄️ Database Controls")
        chunk_size    = st.number_input("Chunk Size",    200, 4000, config.CHUNK_SIZE,    100)
        chunk_overlap = st.number_input("Chunk Overlap",   0, 1000, config.CHUNK_OVERLAP,  50)

        c1, c2 = st.columns(2)
        build_btn = c1.button("🔨 Build DB", use_container_width=True, type="primary")
        clear_btn = c2.button("🗑️ Clear",    use_container_width=True)

        if build_btn:
            if not st.session_state.uploaded_files:
                st.error("Select at least one file first.")
            else:
                bar = st.progress(0)
                txt = st.empty()
                try:
                    config.validate()
                    n_chunks, n_docs = build_database(chunk_size, chunk_overlap, bar, txt)
                    st.success(f"Indexed {n_docs} doc(s) → {n_chunks} chunks")
                    time.sleep(0.4)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        if clear_btn:
            clear_database()
            st.warning("Database cleared.")
            st.rerun()

        chunk_count = refresh_stats()
        doc_count = len(st.session_state.indexed_doc_names) or len(
            (st.session_state.vsm or get_vector_store()).list_sources()
        )
        st.markdown(
            f'<div class="metric-card">'
            f"<b>Documents indexed:</b> {doc_count}<br>"
            f"<b>Total chunks:</b> {chunk_count}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.divider()

        # Retrieval settings
        st.markdown("### ⚙️ Retrieval Settings")
        top_k = st.slider("Top-K Results", 1, 20, config.TOP_K_RESULTS, key="top_k_slider")
        st.checkbox("Query optimization",      value=config.ENABLE_QUERY_OPTIMIZATION, key="opt_query")
        st.checkbox("Hybrid search (BM25 + Vector)", value=config.ENABLE_HYBRID_SEARCH, key="opt_hybrid")
        st.checkbox("Re-ranking (Cross-Encoder)",    value=config.ENABLE_RERANKING,     key="opt_rerank")

        st.divider()

        # Status
        st.markdown("### 📊 System Status")
        up_ok = bool(st.session_state.uploaded_files)
        db_ok = st.session_state.db_ready
        st.markdown(
            f"Documents uploaded: {'✅' if up_ok else '❌'}  \n"
            f"Database ready: {'✅' if db_ok else '❌'}  \n"
            f"LLM: `{config.LLM_PROVIDER}` / `{config.LLM_MODEL}`  \n"
            f"ChromaDB: {'☁️ Cloud' if config.use_chroma_cloud else '💾 Local'}  \n"
            f"Last updated: {st.session_state.last_updated or '—'}"
        )

    return top_k


# ─── TAB 1 — Chat ─────────────────────────────────────────────────────────────
def tab_chat(top_k: int):
    st.markdown('<p class="main-title">📄 Document-Only RAG System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-title">Answers are generated strictly from your uploaded documents '
        "via ChromaDB similarity search + BM25 hybrid retrieval + re-ranking.</p>",
        unsafe_allow_html=True,
    )

    # Status banner
    if st.session_state.db_ready:
        st.success("✅ Documents indexed — ready to answer questions.")
    else:
        st.warning("⚠️ Upload documents and click **Build DB** in the sidebar before asking questions.")

    # ── input form (avoids button-disable/Enter-key quirk) ──
    with st.form("question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask about your documents …",
            placeholder="e.g. What is the refund policy?",
            key="question_field",
        )
        submitted = st.form_submit_button(
            "🔍 Search & Answer",
            type="primary",
            use_container_width=True,
        )

    # guard: must have DB
    if submitted and question.strip():
        if not st.session_state.db_ready:
            st.error("Please build the vector database first (sidebar → Build DB).")
        else:
            result = run_query(question.strip(), top_k)
            if result and "error" not in result:
                st.session_state.chat_history.insert(0, {
                    "question": question.strip(),
                    "answer":   result.get("answer", ""),
                    "sources":  result.get("sources", []),
                    "queries":  result.get("search_queries", []),
                    "mode":     result.get("retrieval_mode", ""),
                })

    # ── chat history ──
    history: list = st.session_state.chat_history
    if not history:
        if st.session_state.db_ready:
            st.info("Type a question above and press **Search & Answer**.")
        return

    st.markdown("---")
    st.markdown(f"#### 💬 Conversation History  ({len(history)} turn{'s' if len(history)!=1 else ''})")

    for idx, turn in enumerate(history):
        with st.container(border=True):
            # question row
            col_q, col_badge = st.columns([8, 2])
            col_q.markdown(f"**Q{len(history)-idx}:** {turn['question']}")
            col_badge.markdown(
                f'<span class="badge badge-blue">{turn["mode"] or "search"}</span>',
                unsafe_allow_html=True,
            )

            # answer
            st.markdown(turn["answer"])

            # sources
            if turn.get("sources"):
                with st.expander(f"📚 Sources ({len(turn['sources'])})"):
                    for s in turn["sources"]:
                        pg = f" · page {s['page']}" if s.get("page") else ""
                        st.markdown(
                            f"- **{s['source_file']}**{pg} — "
                            f"chunk `{s.get('chunk_id','—')}` "
                            f"(score `{s.get('relevance_score','—')}`)"
                        )
                        st.caption(s.get("preview", "")[:180] + "…")

            # expanded queries
            if turn.get("queries") and len(turn["queries"]) > 1:
                with st.expander("🔍 Optimised search queries"):
                    for q in turn["queries"]:
                        st.code(q, language="")

    if st.button("🗑️ Clear chat history", key="clear_hist"):
        st.session_state.chat_history = []
        st.rerun()


# ─── TAB 2 — Sources ──────────────────────────────────────────────────────────
def tab_sources():
    st.markdown("### 📂 Indexed Document Sources")
    vsm   = st.session_state.vsm or get_vector_store()
    srcs  = vsm.list_sources()
    metas = vsm.get_all_metadata()

    if not srcs:
        st.warning("No documents indexed yet. Upload files and click **Build DB**.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", len(srcs))
    c2.metric("Total chunks", vsm.get_document_count())
    pages = {(m.get("source_file"), m.get("page")) for m in metas if m.get("page") is not None}
    c3.metric("Pages indexed", len(pages))

    st.markdown("#### Files")
    for s in srcs:
        st.markdown(f"- `{s}`")

    with st.expander("Chunk metadata table", expanded=False):
        st.dataframe(
            [
                {
                    "source":   m.get("source_file"),
                    "page":     m.get("page"),
                    "chunk_id": m.get("chunk_id", m.get("chunk_index")),
                    "preview":  (m.get("preview") or "")[:80],
                }
                for m in metas
            ],
            use_container_width=True,
        )


# ─── TAB 3 — Retrieval Debug ──────────────────────────────────────────────────
def tab_debug():
    st.markdown("### 🔬 Retrieval Debug")
    debug: Optional[RetrievalDebugInfo] = st.session_state.last_debug

    if not debug or not debug.chunks:
        st.info("Run a query in the **Chat** tab to see retrieval details here.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval time (ms)", f"{debug.retrieval_time_ms:.1f}")
    c2.metric("Top-K returned",      debug.top_k)
    c3.metric("Queries generated",   len(debug.search_queries))

    st.caption(f"Mode: **{debug.mode}**")

    if debug.search_queries:
        st.markdown("**Search queries used:**")
        for i, q in enumerate(debug.search_queries, 1):
            st.code(f"{i}. {q}", language="")

    st.markdown(f"#### Retrieved Chunks  ({len(debug.chunks)})")
    for i, c in enumerate(debug.chunks, 1):
        label = (
            f"#{i}  {c['source_file']}"
            + (f" · page {c['page']}" if c.get("page") is not None else "")
            + f"  |  score {c['similarity_score']}  |  {c.get('chunk_id','')}"
        )
        with st.expander(label):
            st.write(c["content"])
            st.json(c.get("metadata", {}))


# ─── TAB 4 — Architecture ─────────────────────────────────────────────────────
def tab_architecture():
    st.markdown("### 🏗️ System Architecture (Flowchart)")

    llm_label = f"LLM ({config.llm_display_name})"
    search_label = (
        "Hybrid Search (ChromaDB + BM25)"
        if config.ENABLE_HYBRID_SEARCH
        else "Similarity Search (ChromaDB)"
    )
    rerank_label = (
        "Re-ranking (Cross-Encoder)" if config.ENABLE_RERANKING else "Re-ranking (disabled)"
    )
    opt_label = (
        "Query Optimization (Rewrite + Multi-Query)"
        if config.ENABLE_QUERY_OPTIMIZATION
        else "Query Optimization (disabled)"
    )

    st.markdown(
        f"""
        <div style="background:#000000; padding:10px; border-radius:10px;">
          <div style="font-weight:800; color:#bfdbfe; margin-bottom:10px;">
            Upload Documents → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Answer
          </div>
          <div style="display:flex; flex-direction:column; gap:8px; max-width:900px;">
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              📄 Upload Documents<br/><span style="color:#9ca3af; font-weight:500;">(PDF/DOCX/TXT/MD)</span>
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              ✂️ Chunking
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              🔢 Embeddings<br/><span style="color:#9ca3af; font-weight:500;">Sentence Transformers</span>
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              🗄️ ChromaDB
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              🔍 {opt_label}
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              ⚡ {search_label}
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              🏆 {rerank_label}
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              📋 Context Retrieval<br/><span style="color:#9ca3af; font-weight:500;">Document-only grounding</span>
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              🤖 {llm_label}
            </div>
            <div style="text-align:center; color:#6b7280; font-size:22px;">↓</div>
            <div style="background:#0b0b0b; border:1px solid #2a2a2a; border-radius:10px; padding:12px; color:#fff; font-weight:700;">
              💬 Answer with Sources
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Feature flags (live from `.env`)")
    st.json(
        {
            "LLM_PROVIDER": config.LLM_PROVIDER,
            "LLM_MODEL": config.LLM_MODEL,
            "ENABLE_QUERY_OPTIMIZATION": config.ENABLE_QUERY_OPTIMIZATION,
            "ENABLE_HYBRID_SEARCH": config.ENABLE_HYBRID_SEARCH,
            "ENABLE_RERANKING": config.ENABLE_RERANKING,
            "CHROMA": "Cloud" if config.use_chroma_cloud else "Local",
            "LANGCHAIN_TRACING_V2": config.LANGCHAIN_TRACING_V2,
        }
    )


def tab_evaluation():
    st.markdown("### 📈 LLM-as-a-Judge Evaluation")
    st.markdown("Run an automated evaluation suite to measure **Retrieval Accuracy (Context Relevance)** and **Answer Quality (Faithfulness & Answer Relevance)**.")
    
    num_samples = st.slider("Number of samples to evaluate", 1, 50, 5)
    
    if st.button("🚀 Run Evaluation Suite", type="primary"):
        if not st.session_state.db_ready:
            st.error("Please build the vector database first in the sidebar!")
            return
            
        if st.session_state.rag is None:
            st.session_state.rag = DocumentRAGChain(st.session_state.vsm)
            
        progress_text = "Evaluating... Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        def update_progress(current, total):
            progress = current / total
            my_bar.progress(progress, text=f"Evaluating sample {current} of {total}...")
            
        evaluator = RAGEvaluator()
        
        with st.spinner("Running evaluation..."):
            results = evaluator.run_evaluation_suite("eval_dataset.json", st.session_state.rag, num_samples, progress_callback=update_progress)
            
        my_bar.empty()
        
        if results:
            avg_ctx = sum(r['context_relevance'] for r in results) / len(results)
            avg_faith = sum(r['faithfulness'] for r in results) / len(results)
            avg_ans = sum(r['answer_relevance'] for r in results) / len(results)
            
            st.markdown("#### Aggregate Metrics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Context Relevance", f"{avg_ctx:.2f}", delta=None, delta_color="normal")
            c2.metric("Faithfulness", f"{avg_faith:.2f}", delta=None, delta_color="normal")
            c3.metric("Answer Relevance", f"{avg_ans:.2f}", delta=None, delta_color="normal")
            
            st.markdown("#### Detailed Results")
            st.dataframe(results, use_container_width=True)
            st.success("Evaluation completed!")

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    if st.session_state.vsm is None:
        st.session_state.vsm = get_vector_store()
        refresh_stats()

    top_k = render_sidebar()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Chat",
        "📂 Sources",
        "🔬 Retrieval Debug",
        "🏗️ Architecture",
        "📈 Evaluation",
    ])
    with tab1:
        tab_chat(top_k)
    with tab2:
        tab_sources()
    with tab3:
        tab_debug()
    with tab4:
        tab_architecture()
    with tab5:
        tab_evaluation()


if __name__ == "__main__":
    main()
