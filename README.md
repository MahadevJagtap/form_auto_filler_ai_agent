# 🧠 Intelligent Form Auto-Filler (v3.1)

> **Revalsys AI/ML Internship Assignment** — An Intelligent Form Auto-Filler that classifies uploaded documents, selects the appropriate template, extracts information using a hybrid AI pipeline, and auto-fills a dynamic form for user review.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.3-green.svg)](https://groq.com)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Predefined Templates](#predefined-templates)
- [Setup & Installation](#setup--installation)
- [Usage Walkthrough](#usage-walkthrough)
- [Module Documentation](#module-documentation)

---
## Live Demo : https://formautofilleraiagent-ckscrqc9mpay6tunvyckjn.streamlit.app/
---
## Overview

This application demonstrates practical AI/ML engineering by implementing a structured, template-based extraction pipeline:

1. **Upload** documents (PDF, DOCX, JPG, JPEG, PNG).
2. **Classify** the document type (Resume, Aadhaar, Invoice, etc.) using LLM.
3. **Select Template** based on the detected document type.
4. **Extract** data using a 3-layer hybrid pipeline (Regex → Rules → LLM).
5. **Score** confidence mathematically.
6. **Auto-fill** a dynamically generated editable form.
7. **Export** final results to JSON and CSV.

---

## Architecture

```
User Upload (Multi-document)
       ↓
Document Processing (pdfplumber, docx2txt, EasyOCR)
       ↓
Document Classification (Groq LLM)
       ↓
Template Selection (e.g., Resume Template, Invoice Template)
       ↓
Hybrid Extraction Engine
├── Layer 1: Regex (Email, Phone, PAN, Aadhaar)
├── Layer 2: Rules (Dates, Amounts)
└── Layer 3: LLM (Contextual Semantic Data)
       ↓
Result Merging & Confidence Scoring
       ↓
Auto-Filled Form & Explainability Dashboard
       ↓
Validation (Format checks based on field names)
       ↓
User Review & JSON/CSV Export
```

---

## Predefined Templates

The system supports the following document types and templates:

*   **Resume**: Full Name, Email, Phone, Address, Skills, Education, Experience, Certifications
*   **Aadhaar Card**: Full Name, DOB, Gender, Aadhaar Number, Address
*   **PAN Card**: Full Name, PAN Number, DOB
*   **Passport**: Full Name, Passport Number, DOB, Nationality, Address
*   **Invoice**: Invoice Number, Customer Name, Invoice Date, Amount, GST Number
*   **Academic Certificate**: Student Name, University, Degree, Branch, Year
*   **Project Report**: Project Title, Author, Guide, Technology Stack, Abstract
*   **Employee Form**: Employee Name, Employee ID, Department, Designation, Email, Phone
*   **Generic Document**: Document Title, Author, Date, Summary

---

## Setup & Installation

### Prerequisites

- **Python 3.9+** (3.10 or 3.11 recommended)
- **Groq API Key** (free at [console.groq.com](https://console.groq.com))

### Step 1: Clone / Download

```bash
cd d:\revalsys
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set API Key

Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Step 5: Run the Application

```bash
streamlit run app.py
```
The app will open at `http://localhost:8501`.

---

## Usage Walkthrough

1.  **Step 1: Upload**: Upload one or more documents.
2.  **Step 2: Classify**: The system uses LLM to detect the document type and selects the corresponding template.
3.  **Step 3: Extract**: The hybrid pipeline extracts the template fields from the document text. The dashboard shows extracted values, confidence scores, and extraction sources (Regex, Rules, LLM).
4.  **Step 4: Review**: Review and edit the auto-filled form. Validation warnings appear for invalid formats (e.g., malformed email or Aadhaar number).
5.  **Step 5: Export**: Download the finalized data as JSON or CSV.

---

## Module Documentation

*   `app.py`: The main Streamlit 5-step wizard application.
*   `classifiers/document_classifier.py`: Uses Groq LLM to classify documents into predefined categories.
*   `extractors/dynamic_extractor.py`: The hybrid extraction engine (Regex, Rules, LLM) and result merger. Includes the mathematical confidence scoring engine.
*   `extractors/pdf_extractor.py`: Extracts text from PDFs using `pdfplumber`, with an automatic OCR fallback (`PyMuPDF` + `EasyOCR`) for scanned documents.
*   `extractors/image_extractor.py`: Extracts text from images using `EasyOCR`.
*   `extractors/docx_extractor.py`: Extracts text from DOCX files.
*   `validator/validator.py`: Performs dynamic validation based on field name patterns (e.g., validates any field containing "email" or "phone").
