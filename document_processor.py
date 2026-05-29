"""Load and chunk documents for ingestion."""
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader

from config import config

logger = logging.getLogger(__name__)


def _require_pypdf():
    try:
        from pypdf import PdfReader  # noqa: F401
        return PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf package not found. Install it with: pip install pypdf"
        ) from e


class DocumentProcessor:
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        size = chunk_size or config.CHUNK_SIZE
        overlap = chunk_overlap or config.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def load_bytes(self, filename: str, data: bytes) -> List[Document]:
        ext = Path(filename).suffix.lower()
        if ext not in config.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported type: {ext}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        try:
            docs = self._load_path(tmp_path, ext)
            for doc in docs:
                doc.metadata["source_file"] = filename
                doc.metadata["file_type"] = ext.lstrip(".")
            return docs
        finally:
            os.unlink(tmp_path)

    def _load_pdf(self, path: str) -> List[Document]:
        PdfReader = _require_pypdf()
        reader = PdfReader(path)
        documents = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(page_content=text, metadata={"page": page_num})
                )
        if not documents:
            logger.warning("No text extracted from PDF: %s", path)
        return documents

    def _load_path(self, path: str, ext: str) -> List[Document]:
        if ext == ".pdf":
            return self._load_pdf(path)
        if ext == ".txt":
            return TextLoader(path, encoding="utf-8").load()
        if ext == ".docx":
            return Docx2txtLoader(path).load()
        if ext == ".md":
            return TextLoader(path, encoding="utf-8").load()
        raise ValueError(f"No loader for {ext}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_id"] = f"chunk_{i}"
            chunk.metadata["chunk_total"] = len(chunks)
        return chunks

    def process_upload(self, filename: str, data: bytes) -> List[Document]:
        return self.split_documents(self.load_bytes(filename, data))
