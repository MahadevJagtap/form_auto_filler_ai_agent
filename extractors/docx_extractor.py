"""
DOCX Text Extractor
====================
Extracts text from Microsoft Word (.docx) documents using docx2txt.

Features:
    - Extracts paragraphs, tables, and headers in a single call
    - Lightweight — no complex XML parsing needed
    - Returns paragraph count metadata for UI preview
"""

# pyrefly: ignore [missing-import]
import docx2txt


def extract_docx_text(file_path: str) -> dict:
    """
    Extract all text content from a DOCX file.

    ``docx2txt.process()`` handles paragraphs, tables, and header/footer
    content automatically, returning a single unified string.

    Args:
        file_path: Absolute path to the DOCX file on disk.

    Returns:
        dict with keys:
            - text (str): Extracted document text.
            - metadata (dict): Contains ``paragraph_count``.
            - status (str): ``"success"`` or ``"error"``.
            - error (str | None): Error description on failure.
    """
    try:
        text = docx2txt.process(file_path)

        if not text:
            text = ""

        # Count non-empty paragraphs for metadata
        paragraphs = [p for p in text.split("\n") if p.strip()]
        paragraph_count = len(paragraphs)

        return {
            "text": text.strip(),
            "metadata": {"paragraph_count": paragraph_count},
            "status": "success",
            "error": None,
        }

    except Exception as exc:
        return {
            "text": "",
            "metadata": {},
            "status": "error",
            "error": f"DOCX extraction failed: {exc}",
        }
