"""
Image Text Extractor (OCR)
==========================
Extracts text from images (JPG, JPEG, PNG) using EasyOCR.

Features:
    - Zero system-level dependencies (no Tesseract install required)
    - Supports English text recognition out of the box
    - Spatial ordering of detected text blocks
    - EasyOCR Reader is cached to avoid costly re-initialization

Why EasyOCR over Tesseract?
    - pip-installable — no OS-level binary to manage
    - Better accuracy on noisy / low-contrast images
    - Built on PyTorch with deep-learning-based detection
"""

# pyrefly: ignore [missing-import]
import easyocr
import streamlit as st


@st.cache_resource(show_spinner=False)
def _get_reader():
    """
    Create and cache an EasyOCR Reader instance.

    The reader downloads model weights on first use (~100 MB) and keeps
    them cached locally.  ``@st.cache_resource`` ensures this happens
    only once per Streamlit server lifetime.
    """
    return easyocr.Reader(["en"], gpu=False)


def extract_image_text(file_path: str) -> dict:
    """
    Run OCR on an image file and return the detected text.

    EasyOCR returns a list of ``(bbox, text, confidence)`` tuples.
    We sort them top-to-bottom / left-to-right using the bounding-box
    coordinates so the output reads in a natural order.

    Args:
        file_path: Absolute path to the image file on disk.

    Returns:
        dict with keys:
            - text (str): Concatenated OCR text.
            - metadata (dict): Contains ``word_count`` and ``avg_ocr_confidence``.
            - status (str): ``"success"`` or ``"error"``.
            - error (str | None): Error description on failure.
    """
    try:
        reader = _get_reader()
        results = reader.readtext(file_path)

        if not results:
            return {
                "text": "",
                "metadata": {"word_count": 0, "avg_ocr_confidence": 0.0},
                "status": "success",
                "error": None,
            }

        # Sort by vertical position (top of bounding box), then horizontal
        # Each result is (bbox, text, confidence)
        # bbox is a list of 4 corner points: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        sorted_results = sorted(
            results,
            key=lambda r: (r[0][0][1], r[0][0][0]),  # sort by (y, x)
        )

        text_lines = [item[1] for item in sorted_results]
        confidences = [item[2] for item in sorted_results]

        full_text = "\n".join(text_lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return {
            "text": full_text,
            "metadata": {
                "word_count": len(full_text.split()),
                "avg_ocr_confidence": round(avg_confidence, 3),
            },
            "status": "success",
            "error": None,
        }

    except Exception as exc:
        return {
            "text": "",
            "metadata": {},
            "status": "error",
            "error": f"Image OCR failed: {exc}",
        }
