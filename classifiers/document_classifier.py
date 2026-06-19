"""
Document Classifier and Template Registry
=========================================
Uses Groq LLM to classify uploaded documents into predefined categories
and provides the form template associated with each category.
"""

import json
import logging
from typing import Dict, List, Optional
# pyrefly: ignore [missing-import]
from groq import Groq

logger = logging.getLogger(__name__)

# Predefined templates matching the Revalsys requirements
DOCUMENT_TEMPLATES = {
    "Resume": [
        "Full Name", "Email", "Phone", "Address",
        "Skills", "Education", "Experience", "Certifications"
    ],
    "Aadhaar Card": [
        "Full Name", "DOB", "Gender", "Aadhaar Number", "Address"
    ],
    "PAN Card": [
        "Full Name", "PAN Number", "DOB"
    ],
    "Passport": [
        "Full Name", "Passport Number", "DOB", "Nationality", "Address"
    ],
    "Invoice": [
        "Invoice Number", "Customer Name", "Invoice Date", "Amount", "GST Number"
    ],
    "Academic Certificate": [
        "Student Name", "University", "Degree", "Branch", "Year"
    ],
    "Project Report": [
        "Project Title", "Author", "Guide", "Technology Stack", "Abstract"
    ],
    "Employee Form": [
        "Employee Name", "Employee ID", "Department", "Designation", "Email", "Phone"
    ],
    "Generic Document": [
        "Document Title", "Author", "Date", "Summary"
    ]
}

DOCUMENT_CATEGORIES = list(DOCUMENT_TEMPLATES.keys())

class DocumentClassifier:
    """
    Classifies a document's text into one of the predefined categories.
    """
    
    MODEL = "llama-3.3-70b-versatile"
    
    SYSTEM_PROMPT = f"""You are an expert document classification assistant.
Classify the provided document text into exactly one of the following categories:
{', '.join(DOCUMENT_CATEGORIES)}

Return ONLY a valid JSON object with the key 'document_type' and the chosen category as the value.
Do not include any other text or explanation.

Example:
{{"document_type": "Resume"}}
"""

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def classify(self, text: str) -> str:
        """
        Classify document text using Groq LLM.
        Returns the category name or 'Generic Document' on failure.
        """
        # Truncate text if it's too long for classification context
        max_chars = 8000
        prompt_text = text[:max_chars] if len(text) > max_chars else text
        
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Document text:\n\n{prompt_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.0, # Deterministic classification
                max_tokens=150,
            )
            
            raw = response.choices[0].message.content
            data = json.loads(raw)
            doc_type = data.get("document_type", "Generic Document")
            
            if doc_type in DOCUMENT_CATEGORIES:
                logger.info(f"Classified document as: {doc_type}")
                return doc_type
            else:
                logger.warning(f"LLM returned unknown category: {doc_type}. Defaulting to Generic Document.")
                return "Generic Document"
                
        except Exception as exc:
            logger.error(f"Classification failed: {exc}. Defaulting to Generic Document.")
            return "Generic Document"

def get_template_for_type(doc_type: str) -> List[str]:
    """Retrieve the template fields for a given document type."""
    return DOCUMENT_TEMPLATES.get(doc_type, DOCUMENT_TEMPLATES["Generic Document"])
