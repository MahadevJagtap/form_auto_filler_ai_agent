"""
Dynamic Validation Module
=========================
Validates extracted form data based on field name patterns rather than
a hardcoded schema.
"""

import re
from typing import Dict, Any

def validate_all(form_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Validates form data dynamically.
    Returns a dictionary mapping field names to validation result dicts
    containing 'is_valid', 'message', and 'severity'.
    """
    results = {}

    for field, value in form_data.items():
        field_lower = field.lower()
        val_str = str(value).strip()
        
        # 1. Required fields check (basic heuristic)
        # Assume Name and ID fields are critical
        is_critical = "name" in field_lower or "number" in field_lower or "id" in field_lower
        if is_critical and not val_str:
            results[field] = {
                "is_valid": False,
                "message": f"{field} is usually required.",
                "severity": "warning"
            }
            continue

        if not val_str:
            continue # Skip format validation for empty optional fields

        # 2. Email Validation
        if "email" in field_lower:
            if not re.match(r"[^@]+@[^@]+\.[^@]+", val_str):
                results[field] = {
                    "is_valid": False,
                    "message": "Invalid email format.",
                    "severity": "error"
                }

        # 3. Phone Validation
        elif "phone" in field_lower or "mobile" in field_lower:
            digits = re.sub(r"\D", "", val_str)
            if len(digits) < 10 or len(digits) > 15:
                results[field] = {
                    "is_valid": False,
                    "message": "Phone number should be 10-15 digits.",
                    "severity": "warning"
                }

        # 4. Aadhaar Validation
        elif "aadhaar" in field_lower:
            digits = re.sub(r"\D", "", val_str)
            if len(digits) != 12:
                results[field] = {
                    "is_valid": False,
                    "message": "Aadhaar must be exactly 12 digits.",
                    "severity": "error"
                }

        # 5. PAN Validation
        elif "pan" in field_lower and "number" in field_lower:
            if not re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", val_str.upper().replace(" ", "")):
                results[field] = {
                    "is_valid": False,
                    "message": "PAN must be in ABCDE1234F format.",
                    "severity": "error"
                }

    return results
