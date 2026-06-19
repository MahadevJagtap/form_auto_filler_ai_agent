"""
Hybrid Template-Based Extraction Engine
=======================================
Implements a three-layer extraction pipeline driven by a predefined
form template. Result fusion and mathematical confidence scoring
are applied to the extracted data.

Architecture:
    Layer 1 — Regex Extractor:
        Extracts structured fields (email, phone, URLs, IDs, amounts).
        Confidence: 1.0

    Layer 2 — Rule-Based Extractor:
        Extracts semi-structured data (dates, certificate #, amounts).
        Confidence: 0.85

    Layer 3 — LLM Extractor:
        Extracts remaining semantic information relevant to the selected
        template. Confidence: 0.75 base.

    Result Merger:
        Priority: Regex > Rules > LLM. Tracks provenance.

    Confidence Engine:
        confidence = (source_reliability + validation_score + consistency_score) / 3
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

# pyrefly: ignore [missing-import]
from groq import Groq

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Layer 1 — Regex Extractor
# ═══════════════════════════════════════════════════════════════════════════

class RegexExtractor:
    """Extracts structured fields via regular expressions."""

    PATTERNS = {
        "Email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE),
        "Phone": re.compile(r"(?:\+?\d{1,3}[\s\-.]?)?(?:\(?\d{2,4}\)?[\s\-.]?)?\d{3,5}[\s\-.]?\d{3,5}"),
        "Aadhaar Number": re.compile(r"\b(\d{4}[\s\-]?\d{4}[\s\-]?\d{4})\b"),
        "PAN Number": re.compile(r"\b([A-Z]{5}\d{4}[A-Z])\b"),
        "Invoice Number": re.compile(r"(?i)(?:invoice|inv)[\s\-\#]*([A-Z0-9\-\/]+)"),
        "GST Number": re.compile(r"\b([0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b"),
        "Passport Number": re.compile(r"\b([A-Z]{1}[0-9]{7})\b"), # Indian passport format
    }

    def extract(self, text: str, template: List[str]) -> Dict[str, Any]:
        results = {}
        
        for field in template:
            if field in self.PATTERNS:
                matches = self.PATTERNS[field].findall(text)
                if matches:
                    # Special handling for Phone to avoid catching Aadhaar
                    if field == "Phone":
                        valid_phones = []
                        for m in matches:
                            digits_only = re.sub(r"\D", "", m)
                            if 10 <= len(digits_only) <= 15 and len(digits_only) != 12: # Avoid 12-digit Aadhaar
                                valid_phones.append(m)
                        if valid_phones:
                            results[field] = {
                                "value": valid_phones[0].strip(),
                                "confidence": 1.0,
                                "source": "regex"
                            }
                    else:
                        val = matches[0] if isinstance(matches[0], str) else matches[0][0]
                        results[field] = {
                            "value": str(val).strip(),
                            "confidence": 1.0,
                            "source": "regex"
                        }
        return results

# ═══════════════════════════════════════════════════════════════════════════
# Layer 2 — Rule-Based Extractor
# ═══════════════════════════════════════════════════════════════════════════

class RuleBasedExtractor:
    """Extracts semi-structured data based on deterministic rules."""

    def extract(self, text: str, template: List[str]) -> Dict[str, Any]:
        results = {}

        # DOB / Dates
        date_patterns = [
            re.compile(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b"),
            re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),
            re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b", re.IGNORECASE)
        ]
        
        date_fields = [f for f in template if "Date" in f or f == "DOB" or f == "Year"]
        for field in date_fields:
            if field == "DOB":
                # Look for context
                context_match = re.search(r"(?:DOB|Date\s+of\s+Birth|Birth\s*Date|D\.O\.B)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})", text, re.IGNORECASE)
                if context_match:
                    results[field] = {"value": context_match.group(1).strip(), "confidence": 0.85, "source": "rule_engine"}
                    continue
                    
            # Fallback to first date found
            for pattern in date_patterns:
                matches = pattern.findall(text)
                if matches:
                    results[field] = {"value": matches[0].strip(), "confidence": 0.85, "source": "rule_engine"}
                    break

        # Amounts
        amount_fields = [f for f in template if "Amount" in f]
        for field in amount_fields:
            # Matches ₹1,000.00, Rs. 500, etc.
            amt_match = re.search(r"(?:Rs\.?|INR|₹|\$)\s*([\d,]+(?:\.\d{2})?)", text, re.IGNORECASE)
            if amt_match:
                results[field] = {"value": amt_match.group(1).strip(), "confidence": 0.85, "source": "rule_engine"}

        # Gender
        if "Gender" in template:
            gender_match = re.search(r"\b(Male|Female|Transgender|OTHER|पुरुष|महिला)\b", text, re.IGNORECASE)
            if gender_match:
                raw_gender = gender_match.group(1).strip().capitalize()
                gender_map = {"पुरुष": "Male", "महिला": "Female"}
                results["Gender"] = {
                    "value": gender_map.get(raw_gender, raw_gender), 
                    "confidence": 0.85, 
                    "source": "rule_engine"
                }

        return results

# ═══════════════════════════════════════════════════════════════════════════
# Layer 3 — LLM Extractor
# ═══════════════════════════════════════════════════════════════════════════

class LLMExtractor:
    """Uses Groq LLM to extract semantic information based on the template."""

    MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)

    def extract(self, text: str, template: List[str], doc_type: str) -> Dict[str, Any]:
        if not template:
            return {}

        fields_json = json.dumps({field: "" for field in template})
        
        system_prompt = f"""You are an expert information extraction assistant.
Extract information from the provided {doc_type} document text based on the required fields.

Return ONLY a valid JSON object matching exactly these keys:
{fields_json}

Rules:
1. Extract ONLY information explicitly present in the text.
2. Do NOT invent or hallucinate information. If a field is not found, leave its value as an empty string ("").
3. For fields like 'Skills', 'Education', 'Experience', or 'Certifications', provide a comma-separated string if multiple items are found.
4. Return ONLY the JSON object. No markdown, no explanations.
"""
        max_chars = 12000
        prompt_text = text[:max_chars] if len(text) > max_chars else text

        try:
            response = self.client.chat.completions.create(
                model=self.MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Document text:\n\n{prompt_text}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=2000,
            )

            raw = response.choices[0].message.content
            data = json.loads(raw)
            
            results = {}
            for field in template:
                val = data.get(field)
                if val and str(val).strip():
                    results[field] = {
                        "value": str(val).strip(),
                        "confidence": 0.75, # Base LLM confidence
                        "source": "llm"
                    }
            return results

        except Exception as exc:
            logger.error(f"LLM extraction failed: {exc}")
            return {}

# ═══════════════════════════════════════════════════════════════════════════
# Result Merging & Confidence Engine
# ═══════════════════════════════════════════════════════════════════════════

class ExtractionOrchestrator:
    
    @staticmethod
    def calculate_validation_score(field: str, value: str) -> float:
        """Basic validation to contribute to confidence score."""
        val = str(value).strip().lower()
        if not val:
            return 0.50
        
        # Simple format checks
        if "email" in field.lower() and "@" in val and "." in val:
            return 1.0
        if "phone" in field.lower() and any(c.isdigit() for c in val) and len(re.sub(r"\D", "", val)) >= 10:
            return 1.0
        if "date" in field.lower() or field == "DOB" or field == "Year":
            if any(c.isdigit() for c in val):
                return 1.0
        
        # Default for non-empty generic fields
        return 0.75

    @staticmethod
    def process(text: str, template: List[str], doc_type: str, api_key: Optional[str]) -> Dict[str, Any]:
        """Runs the hybrid pipeline and merges results."""
        
        # 1. Extraction
        regex_res = RegexExtractor().extract(text, template)
        rule_res = RuleBasedExtractor().extract(text, template)
        
        llm_res = {}
        if api_key:
            llm_res = LLMExtractor(api_key).extract(text, template, doc_type)
            
        # 2. Result Merging
        merged = {}
        for field in template:
            candidates = []
            if field in regex_res: candidates.append(regex_res[field])
            if field in rule_res: candidates.append(rule_res[field])
            if field in llm_res: candidates.append(llm_res[field])
            
            if not candidates:
                merged[field] = {"value": "", "confidence": 0.0, "source": "none"}
                continue
                
            # Priority is implicit in the order we added them, but let's be explicit
            # Regex > Rule > LLM
            primary = None
            if field in regex_res: primary = regex_res[field]
            elif field in rule_res: primary = rule_res[field]
            elif field in llm_res: primary = llm_res[field]
            
            # 3. Confidence Calculation
            source_reliability = primary["confidence"] # 1.0 for regex, 0.85 for rules, 0.75 for llm
            val_score = ExtractionOrchestrator.calculate_validation_score(field, primary["value"])
            consistency_score = 1.0 if len(candidates) > 1 else 0.70
            
            final_conf = (source_reliability + val_score + consistency_score) / 3.0
            
            # Formatting sources for display
            sources = " + ".join([c["source"].replace("rule_engine", "rules") for c in candidates])
            
            merged[field] = {
                "value": primary["value"],
                "confidence": round(final_conf, 3),
                "source": sources
            }
            
        return merged
