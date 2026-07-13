"""
Rebuild the persistent Chroma vector store from documents under data/.

This script uses the production RAG pipeline classes from rag.pipeline:
- EmbeddingManager for embeddings
- VectorStore for Chroma persistence and metadata flattening

It intentionally avoids notebook code paths.
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
from pathlib import Path
from typing import List

from langchain_docling.loader import DoclingLoader, ExportType

from rag.config import COLLECTION_NAME, DATA_DIR, SUPPORTED_EXTENSIONS, VECTOR_STORE_DIR
from rag.pipeline import EmbeddingManager, VectorStore, load_env


logger = logging.getLogger(__name__)


def discover_documents(data_dir: Path) -> List[Path]:
    """Return all supported files under data_dir, recursively."""
    files: List[Path] = []
    seen: set[Path] = set()

    for pattern in SUPPORTED_EXTENSIONS:
        for path in data_dir.glob(f"**/{pattern}"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)

    return sorted(files)


def load_documents(file_path: Path, max_pages_per_doc: int | None = None):
    """Load a single file with Docling and add source metadata."""
    loader = DoclingLoader(
        str(file_path),
        export_type=ExportType.DOC_CHUNKS,
    )
    documents = loader.load()

    if max_pages_per_doc is not None:
        logger.warning(
            "--max-pages-per-doc is not supported by the installed DoclingLoader and will be ignored."
        )

    file_ext = file_path.suffix.lower().lstrip(".")
    for document in documents:
        document.metadata["source_file"] = file_path.name
        document.metadata["file_type"] = file_ext

    return documents


def flatten_documents(data_dir: Path, max_pages_per_doc: int | None = None):
    """Load every supported document under data_dir."""
    all_documents = []
    files = discover_documents(data_dir)

    logger.info("Found %d supported file(s) under %s", len(files), data_dir)

    for file_path in files:
        logger.info("Loading %s", file_path.relative_to(data_dir))
        try:
            documents = load_documents(file_path, max_pages_per_doc=max_pages_per_doc)
            all_documents.extend(documents)
            logger.info("Loaded %d chunk(s) from %s", len(documents), file_path.name)
        except Exception as exc:
            logger.exception("Error processing %s: %s", file_path, exc)
        finally:
            gc.collect()

    return all_documents


def rebuild_vector_store(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = VECTOR_STORE_DIR,
    collection_name: str = COLLECTION_NAME,
    max_pages_per_doc: int | None = None,
    wipe_existing: bool = True,
) -> int:
    """Delete the current store and rebuild it from data_dir.

    Returns the number of loaded LangChain documents/chunks.
    """
    load_env()

    if wipe_existing and persist_dir.exists():
        logger.info("Removing existing vector store at %s", persist_dir)
        shutil.rmtree(persist_dir)

    documents = flatten_documents(data_dir, max_pages_per_doc=max_pages_per_doc)
    if not documents:
        raise RuntimeError(f"No supported documents found under {data_dir}")

    logger.info("Building embeddings for %d document chunk(s)", len(documents))
    embedder = EmbeddingManager()
    embeddings = embedder.embed([doc.page_content for doc in documents], show_progress_bar=True)

    store = VectorStore(collection_name=collection_name, persist_directory=persist_dir)
    store.add_documents(documents, embeddings)

    logger.info(
        "Rebuild complete. Collection '%s' now contains %d document chunk(s).",
        collection_name,
        store.collection.count(),
    )
    return len(documents)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Delete and rebuild the Chroma vector store from documents under data/."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Root directory containing source documents (default: data/)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=VECTOR_STORE_DIR,
        help="Chroma persist directory (default: data/vector_store)",
    )
    parser.add_argument(
        "--collection-name",
        default=COLLECTION_NAME,
        help="Chroma collection name (default: pdf_documents)",
    )
    parser.add_argument(
        "--max-pages-per-doc",
        type=int,
        default=None,
        help="Optional page cap per document when loading with Docling.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not delete the existing vector store before rebuilding.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = build_arg_parser()
    args = parser.parse_args()

    rebuild_vector_store(
        data_dir=args.data_dir,
        persist_dir=args.persist_dir,
        collection_name=args.collection_name,
        max_pages_per_doc=args.max_pages_per_doc,
        wipe_existing=not args.keep_existing,
    )


if __name__ == "__main__":
    main()