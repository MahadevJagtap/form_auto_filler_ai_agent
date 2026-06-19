"""
Unified Document Extraction Module
===================================
Routes uploaded files to the appropriate extractor based on file type.
Supports multi-document upload with text merging across all documents.

Supported formats:
    - PDF  → pdfplumber
    - DOCX → docx2txt  
    - JPG/JPEG/PNG → EasyOCR
"""

import os
import tempfile
from typing import Union, List

from extractors.pdf_extractor import extract_pdf_text
from extractors.docx_extractor import extract_docx_text
from extractors.image_extractor import extract_image_text


# ---------------------------------------------------------------------------
# File-type → extractor mapping
# ---------------------------------------------------------------------------
EXTRACTOR_MAP = {
    ".pdf": extract_pdf_text,
    ".docx": extract_docx_text,
    ".jpg": extract_image_text,
    ".jpeg": extract_image_text,
    ".png": extract_image_text,
}

SUPPORTED_EXTENSIONS = list(EXTRACTOR_MAP.keys())


def _save_uploaded_file(uploaded_file) -> str:
    """
    Persist a Streamlit UploadedFile to a temporary path so that
    extractors can open it by filename.

    Returns:
        Absolute path to the temporary file.
    """
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        # Reset the stream position so the file can be re-read if needed
        uploaded_file.seek(0)
        return tmp.name


def extract_text(uploaded_file) -> dict:
    """
    Extract text from a single uploaded file.

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        dict with keys:
            - text (str): The extracted raw text.
            - metadata (dict): File-specific metadata (page count, etc.).
            - filename (str): Original filename.
            - status (str): "success" or "error".
            - error (str | None): Error message if status is "error".
    """
    filename = uploaded_file.name
    extension = os.path.splitext(filename)[1].lower()

    # Validate file type
    if extension not in EXTRACTOR_MAP:
        return {
            "text": "",
            "metadata": {},
            "filename": filename,
            "status": "error",
            "error": f"Unsupported file type: {extension}. "
                     f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}",
        }

    # Save to temp file and run the appropriate extractor
    tmp_path = None
    try:
        tmp_path = _save_uploaded_file(uploaded_file)
        extractor_fn = EXTRACTOR_MAP[extension]
        result = extractor_fn(tmp_path)
        result["filename"] = filename
        return result
    except Exception as exc:
        return {
            "text": "",
            "metadata": {},
            "filename": filename,
            "status": "error",
            "error": str(exc),
        }
    finally:
        # Clean up temporary file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def extract_from_multiple(uploaded_files: List) -> dict:
    """
    Extract text from multiple uploaded files and merge the results.

    Args:
        uploaded_files: List of Streamlit UploadedFile objects.

    Returns:
        dict with keys:
            - text (str): Merged text from all documents.
            - documents (list[dict]): Per-document extraction results.
            - status (str): "success" if at least one doc succeeded.
    """
    documents = []
    merged_texts = []

    for uploaded_file in uploaded_files:
        result = extract_text(uploaded_file)
        documents.append(result)
        if result["status"] == "success" and result["text"].strip():
            merged_texts.append(
                f"--- Document: {result['filename']} ---\n{result['text']}"
            )

    overall_status = "success" if merged_texts else "error"

    return {
        "text": "\n\n".join(merged_texts),
        "documents": documents,
        "status": overall_status,
    }
