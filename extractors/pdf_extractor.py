"""
PDF Text Extractor
==================
Extracts text from PDF documents using pdfplumber, with an
automatic OCR fallback for scanned / image-based PDFs.

Pipeline:
    1. Try pdfplumber.extract_text() (fast, for digital PDFs)
    2. If text is empty/minimal → render pages as images via PyMuPDF
    3. Run EasyOCR on the rendered images (handles Aadhaar, PAN, etc.)

Features:
    - Page-by-page text extraction with reading order preservation
    - Table detection and conversion to text
    - Automatic OCR fallback for scanned PDFs (Aadhaar, passports, etc.)
    - Metadata collection (page count, extraction method used)
    - Graceful handling of corrupted / encrypted PDFs
"""

import os
import tempfile
import logging

# pyrefly: ignore [missing-import]
import pdfplumber

logger = logging.getLogger(__name__)

# Minimum characters threshold — below this we assume the PDF is image-based
MIN_TEXT_LENGTH = 30


def _ocr_fallback(file_path: str) -> dict:
    """
    Fallback: render PDF pages as images using PyMuPDF, then OCR with EasyOCR.

    This handles scanned PDFs, Aadhaar cards, PAN cards, passports, and
    other image-based documents that pdfplumber cannot parse.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        dict with text, metadata, status, error keys.
    """
    try:
        # pyrefly: ignore [missing-import]
        import fitz  # PyMuPDF — used only for rendering pages as images
        from extractors.image_extractor import extract_image_text
    except ImportError as exc:
        return {
            "text": "",
            "metadata": {},
            "status": "error",
            "error": f"OCR fallback requires PyMuPDF: {exc}",
        }

    try:
        doc = fitz.open(file_path)
        page_count = len(doc)
        all_texts = []

        for page_num in range(page_count):
            page = doc[page_num]

            # Render page as a high-resolution image (300 DPI)
            # zoom=3 gives ~300 DPI from the default 72 DPI
            mat = fitz.Matrix(3, 3)
            pix = page.get_pixmap(matrix=mat)

            # Save to a temporary PNG file for EasyOCR
            tmp_img = tempfile.NamedTemporaryFile(
                delete=False, suffix=".png"
            )
            pix.save(tmp_img.name)
            tmp_img.close()

            try:
                # Run OCR on the rendered page image
                ocr_result = extract_image_text(tmp_img.name)
                if ocr_result["status"] == "success" and ocr_result["text"].strip():
                    all_texts.append(
                        f"[Page {page_num + 1}]\n{ocr_result['text']}"
                    )
            finally:
                # Clean up temp image
                try:
                    os.unlink(tmp_img.name)
                except OSError:
                    pass

        doc.close()

        full_text = "\n\n".join(all_texts)
        logger.info(
            "OCR fallback extracted %d chars from %d pages",
            len(full_text), page_count,
        )

        return {
            "text": full_text,
            "metadata": {
                "page_count": page_count,
                "extraction_method": "ocr_fallback",
            },
            "status": "success" if full_text.strip() else "error",
            "error": None if full_text.strip() else "OCR produced no text",
        }

    except Exception as exc:
        logger.error("OCR fallback failed: %s", exc)
        return {
            "text": "",
            "metadata": {},
            "status": "error",
            "error": f"OCR fallback failed: {exc}",
        }


def extract_pdf_text(file_path: str) -> dict:
    """
    Extract all text content from a PDF file.

    First tries pdfplumber (fast, for digital/text PDFs).
    If the result is empty or too short, automatically falls back
    to OCR via PyMuPDF + EasyOCR (for scanned documents like
    Aadhaar cards, PAN cards, passports, etc.).

    Args:
        file_path: Absolute path to the PDF file on disk.

    Returns:
        dict with keys:
            - text (str): Concatenated text from all pages.
            - metadata (dict): Contains ``page_count`` and ``extraction_method``.
            - status (str): ``"success"`` or ``"error"``.
            - error (str | None): Error description on failure.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            page_texts = []
            page_count = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                parts = []

                # ----- Regular text -----
                text = page.extract_text()
                if text:
                    parts.append(text.strip())

                # ----- Tables -----
                tables = page.extract_tables()
                for table in tables:
                    table_lines = []
                    for row in table:
                        cleaned = [
                            (cell or "").strip() for cell in row
                        ]
                        table_lines.append(" | ".join(cleaned))
                    parts.append("\n".join(table_lines))

                if parts:
                    page_texts.append(
                        f"[Page {page_num}]\n" + "\n".join(parts)
                    )

            full_text = "\n\n".join(page_texts)

            # ── Check if we got enough text ──
            # If text is too short, this is likely a scanned/image PDF
            if len(full_text.strip()) < MIN_TEXT_LENGTH:
                logger.info(
                    "pdfplumber extracted only %d chars — "
                    "falling back to OCR for scanned PDF",
                    len(full_text.strip()),
                )
                return _ocr_fallback(file_path)

            return {
                "text": full_text,
                "metadata": {
                    "page_count": page_count,
                    "extraction_method": "pdfplumber",
                },
                "status": "success",
                "error": None,
            }

    except Exception as exc:
        # If pdfplumber fails entirely (e.g. encrypted PDF), try OCR
        logger.warning(
            "pdfplumber failed (%s) — attempting OCR fallback", exc
        )
        return _ocr_fallback(file_path)
