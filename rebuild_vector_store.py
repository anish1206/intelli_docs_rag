"""
Rebuild the persistent Chroma vector store from documents under data/.

This script uses the production RAG pipeline classes from rag.pipeline:
- EmbeddingManager for embeddings
- VectorStore for Chroma persistence and metadata flattening

PDF loading uses a lightweight Docling converter (heavy ML features disabled,
batch sizes=1, one page at a time) to avoid bad memory allocation errors.
"""

from __future__ import annotations

import argparse
import gc
import logging
import shutil
from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_docling.loader import DoclingLoader, ExportType
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from rag.config import COLLECTION_NAME, DATA_DIR, SUPPORTED_EXTENSIONS, VECTOR_STORE_DIR
from rag.pipeline import EmbeddingManager, VectorStore, load_env


logger = logging.getLogger(__name__)


def build_light_pdf_converter() -> DocumentConverter:
    """
    Build a memory-efficient PDF converter with all heavy ML features disabled.
    Mirrors the notebook's build_pdf_converter() to prevent bad memory allocation.
    """
    pdf_options = PdfPipelineOptions(
        force_backend_text=True,
        do_ocr=False,
        do_table_structure=True,
        do_code_enrichment=False,
        do_formula_enrichment=False,
        do_picture_classification=False,
        do_picture_description=False,   # disabled (RAM Heavy)
        do_chart_extraction=False,
        generate_page_images=False,
        generate_picture_images=False,
        generate_table_images=False,
        generate_parsed_pages=False,
        document_timeout=120,
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        batch_polling_interval_seconds=0.1,
        queue_max_size=1,
        images_scale=1.0,
    )
    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)},
    )


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


def load_pdf_documents(file_path: Path, converter: DocumentConverter) -> list:
    """
    Load a PDF one page at a time to avoid memory spikes.
    Mirrors the notebook's load_pdf_documents() with page_chunk_size=1.
    """
    all_documents = []
    page_count = len(PdfReader(str(file_path)).pages)

    for page_no in range(1, page_count + 1):
        logger.info("  Page %d/%d", page_no, page_count)
        loader = DoclingLoader(
            str(file_path),
            converter=converter,
            export_type=ExportType.DOC_CHUNKS,
            convert_kwargs={"page_range": (page_no, page_no)},
        )
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = file_path.name
            doc.metadata["file_type"] = "pdf"
            doc.metadata["source_page_range"] = str(page_no)
        all_documents.extend(docs)
        gc.collect()

    return all_documents


def load_documents(file_path: Path, pdf_converter: DocumentConverter) -> list:
    """Load a single file with Docling and add source metadata."""
    file_ext = file_path.suffix.lower()

    if file_ext == ".pdf":
        return load_pdf_documents(file_path, pdf_converter)

    # Non-PDF files (txt, docx, pptx, etc.) — use the default converter
    loader = DoclingLoader(
        str(file_path),
        export_type=ExportType.DOC_CHUNKS,
    )
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = file_path.name
        doc.metadata["file_type"] = file_ext.lstrip(".")
    return documents


def flatten_documents(data_dir: Path) -> list:
    """Load every supported document under data_dir."""
    all_documents = []
    files = discover_documents(data_dir)
    logger.info("Found %d supported file(s) under %s", len(files), data_dir)

    # Build the lightweight PDF converter once and reuse it
    pdf_converter = build_light_pdf_converter()

    for file_path in files:
        logger.info("Loading %s", file_path.relative_to(data_dir))
        try:
            documents = load_documents(file_path, pdf_converter)
            all_documents.extend(documents)
            logger.info("  Loaded %d chunk(s) from %s", len(documents), file_path.name)
        except Exception as exc:
            logger.exception("Error processing %s: %s", file_path, exc)
        finally:
            gc.collect()

    return all_documents


def rebuild_vector_store(
    data_dir: Path = DATA_DIR,
    persist_dir: Path = VECTOR_STORE_DIR,
    collection_name: str = COLLECTION_NAME,
    wipe_existing: bool = True,
) -> int:
    """Delete the current store and rebuild it from data_dir.

    Returns the number of loaded LangChain documents/chunks.
    """
    load_env()

    if wipe_existing and persist_dir.exists():
        logger.info("Removing existing vector store at %s", persist_dir)
        shutil.rmtree(persist_dir)

    documents = flatten_documents(data_dir)
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
        wipe_existing=not args.keep_existing,
    )


if __name__ == "__main__":
    main()