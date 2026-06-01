"""
Ingest all documents from sample_docs/ into ChromaDB.
Run once: python ingest_docs.py
"""
import os
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager

DOCS_DIR = Path("./sample_docs")

def main():
    print("Initializing...")
    proc = DocumentProcessor()
    vsm = VectorStoreManager()

    existing = vsm.list_sources()
    print(f"Already ingested: {len(existing)} sources, {vsm.get_document_count()} chunks")

    files = list(DOCS_DIR.rglob("*.txt")) + \
            list(DOCS_DIR.rglob("*.pdf")) + \
            list(DOCS_DIR.rglob("*.docx")) + \
            list(DOCS_DIR.rglob("*.md"))

    print(f"Found {len(files)} documents to ingest\n")

    total_chunks = 0
    for i, fpath in enumerate(files, 1):
        if fpath.name in existing:
            print(f"  [{i}/{len(files)}] Skipping (already ingested): {fpath.name}")
            continue
        try:
            data = fpath.read_bytes()
            chunks = proc.process_upload(fpath.name, data)
            ids = vsm.add_documents(chunks)
            total_chunks += len(ids)
            print(f"  [{i}/{len(files)}] ✅ {fpath.name} → {len(ids)} chunks")
        except Exception as e:
            print(f"  [{i}/{len(files)}] ❌ {fpath.name}: {e}")

    print(f"\nDone! Total chunks in store: {vsm.get_document_count()}")

if __name__ == "__main__":
    main()
