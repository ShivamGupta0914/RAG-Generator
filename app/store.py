import tempfile
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    ALLOWED_EXTENSIONS,
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    UPLOADS_DIR,
)
from app.llm import get_embeddings

_db = None
MAX_BYTES = 25 * 1024 * 1024


def get_db():
    global _db
    if _db is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _db = Chroma(
            collection_name="documents",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    return _db


def list_files() -> list[str]:
    if not UPLOADS_DIR.exists():
        return []
    names = []
    for path in sorted(UPLOADS_DIR.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            names.append(path.name)
    return names


def documents_from_upload(name: str, raw: bytes) -> list[Document]:
    suffix = Path(name).suffix.lower()
    if suffix == ".pdf":
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        try:
            handle.write(raw)
            handle.close()
            try:
                docs = PyPDFLoader(handle.name).load()
            except Exception as exc:
                raise ValueError(f"Could not read {name}.") from exc
        finally:
            Path(handle.name).unlink(missing_ok=True)
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Text files must be UTF-8.") from exc
        docs = [Document(page_content=text, metadata={})]

    for doc in docs:
        doc.metadata["source"] = name
    return docs


def index_uploads(files: list[tuple[str, bytes]]):
    if not files:
        raise ValueError("Choose a txt, md, or pdf file.")

    chosen: dict[str, bytes] = {}
    for filename, raw in files:
        name = Path(filename or "").name
        suffix = Path(name).suffix.lower()
        if not name or suffix not in ALLOWED_EXTENSIONS:
            raise ValueError("Upload txt, md, or pdf files.")
        if len(raw) > MAX_BYTES:
            raise ValueError(f"{name} is larger than 25 MB.")
        chosen[name] = raw

    loaded = []
    for name, raw in chosen.items():
        loaded.append((name, raw, documents_from_upload(name, raw)))

    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    db = get_db()
    docs = []
    names = []

    for name, raw, file_docs in loaded:
        try:
            already = db.get(where={"source": name})
            old_ids = already.get("ids") or []
        except Exception:
            old_ids = []
        if old_ids:
            db.delete(ids=old_ids)

        (UPLOADS_DIR / name).write_bytes(raw)
        docs.extend(file_docs)
        names.append(name)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs) if docs else []
    chunks = [chunk for chunk in chunks if chunk.page_content and chunk.page_content.strip()]
    if chunks:
        db.add_documents(chunks)
    return {"files": names, "chunks": len(chunks)}


def search_documents(query: str, k: int = 4):
    if not list_files():
        return []
    return get_db().similarity_search(query, k=k)
