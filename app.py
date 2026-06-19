"""
Intelligent Form Auto-Filler
============================
A production-quality Document Intelligence System that classifies uploaded
documents into predefined categories (Resume, Aadhaar, Invoice, etc.),
selects an appropriate template, extracts information using a hybrid
pipeline (Regex + Rules + LLM), and generates an editable form.
"""

import os
import json
import logging
from datetime import datetime

# pyrefly: ignore [missing-import]
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from extractors import extract_from_multiple
from classifiers.document_classifier import DocumentClassifier, get_template_for_type
from extractors.dynamic_extractor import ExtractionOrchestrator
from validator.validator import validate_all

# ---------------------------------------------------------------------------
# Logging & Page Config
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="AI Form Auto-Filler", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------------------
# Custom CSS for premium look (No sidebar, no footer)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; max-width: 1000px; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .step-indicator { display: flex; justify-content: center; gap: 0.5rem; margin-bottom: 2rem; }
    .step-badge { padding: 0.5rem 1.2rem; border-radius: 25px; font-size: 0.85rem; font-weight: 600; }
    .step-active { background: linear-gradient(135deg, #6C63FF, #4ECDC4); color: white; box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4); }
    .step-done { background: rgba(78, 205, 196, 0.2); color: #4ECDC4; border: 1px solid #4ECDC4; }
    .step-pending { background: rgba(255, 255, 255, 0.05); color: #888; border: 1px solid #333; }

    .conf-high { background: rgba(0, 200, 83, 0.15); color: #00C853; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .conf-medium { background: rgba(255, 171, 0, 0.15); color: #FFAB00; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .conf-low { background: rgba(255, 82, 82, 0.15); color: #FF5252; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }
    .source-tag { background: rgba(255,255,255,0.1); color: #ccc; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; }

    .metric-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.2rem; text-align: center; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #6C63FF; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State
# ---------------------------------------------------------------------------
def init_session_state():
    defaults = {
        "step": 1,
        "uploaded_files": [],
        "raw_text": "",
        "doc_type": "",
        "template": [],
        "extracted_data": {},
        "form_values": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state["groq_api_key"] = os.environ.get("GROQ_API_KEY", "")

init_session_state()

def set_step(step: int):
    st.session_state.step = step

def get_conf_html(conf: float) -> str:
    if conf >= 0.85: return f'<span class="conf-high">{int(conf*100)}%</span>'
    if conf >= 0.60: return f'<span class="conf-medium">{int(conf*100)}%</span>'
    return f'<span class="conf-low">{int(conf*100)}%</span>'

# ---------------------------------------------------------------------------
# Header & Steps
# ---------------------------------------------------------------------------
st.markdown("<h1 style='text-align:center; background: linear-gradient(135deg, #6C63FF, #4ECDC4); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🧠 Intelligent Form Auto-Filler</h1>", unsafe_allow_html=True)

steps = ["📄 Upload", "🏷️ Classify", "🔍 Extract", "📋 Review", "✅ Export"]
badges = []
for i, label in enumerate(steps, start=1):
    if i < st.session_state.step: badges.append(f'<span class="step-badge step-done">{label}</span>')
    elif i == st.session_state.step: badges.append(f'<span class="step-badge step-active">{label}</span>')
    else: badges.append(f'<span class="step-badge step-pending">{label}</span>')
st.markdown(f'<div class="step-indicator">{"".join(badges)}</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# UI Steps
# ═══════════════════════════════════════════════════════════════════════════

# --- STEP 1: UPLOAD ---
if st.session_state.step == 1:
    st.markdown("### 📄 Step 1: Upload Document")
    uploaded_files = st.file_uploader("Upload document (PDF, DOCX, JPG, PNG)", type=["pdf", "docx", "jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        if st.button("Process Document →", type="primary"):
            with st.spinner("Extracting text..."):
                res = extract_from_multiple(uploaded_files)
                if res["status"] == "success":
                    st.session_state.raw_text = res["text"]
                    set_step(2)
                    st.rerun()
                else:
                    st.error("Failed to extract text.")

# --- STEP 2: CLASSIFY ---
elif st.session_state.step == 2:
    st.markdown("### 🏷️ Step 2: Document Classification")
    if not st.session_state.doc_type:
        with st.spinner("Classifying document..."):
            classifier = DocumentClassifier(api_key=st.session_state.groq_api_key)
            doc_type = classifier.classify(st.session_state.raw_text)
            st.session_state.doc_type = doc_type
            st.session_state.template = get_template_for_type(doc_type)
            st.rerun()
            
    st.success(f"Detected Document Type: **{st.session_state.doc_type}**")
    st.info(f"Selected Template Fields: {', '.join(st.session_state.template)}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"): set_step(1); st.rerun()
    with col2:
        if st.button("Extract Information →", type="primary"): set_step(3); st.rerun()

# --- STEP 3: EXTRACT (DASHBOARD) ---
elif st.session_state.step == 3:
    st.markdown("### 🔍 Step 3: Extraction Dashboard")
    
    if not st.session_state.extracted_data:
        with st.spinner("Running hybrid extraction..."):
            data = ExtractionOrchestrator.process(
                st.session_state.raw_text, 
                st.session_state.template, 
                st.session_state.doc_type,
                st.session_state.groq_api_key
            )
            st.session_state.extracted_data = data
            # Init form values
            st.session_state.form_values = {k: v["value"] for k, v in data.items()}
            st.rerun()

    # Dashboard Table
    st.markdown("Field | Value | Confidence | Source")
    st.markdown("---|---|---|---")
    for field, info in st.session_state.extracted_data.items():
        val = str(info["value"])[:60] + ("..." if len(str(info["value"])) > 60 else "")
        conf = get_conf_html(info["confidence"])
        src = f'<span class="source-tag">{info["source"]}</span>'
        st.markdown(f"**{field}** | `{val}` | {conf} | {src}", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"): set_step(2); st.rerun()
    with col2:
        if st.button("Review Auto-Filled Form →", type="primary"): set_step(4); st.rerun()

# --- STEP 4: REVIEW FORM ---
elif st.session_state.step == 4:
    st.markdown(f"### 📋 Step 4: Review Form ({st.session_state.doc_type})")
    
    form_values = st.session_state.form_values
    extracted = st.session_state.extracted_data
    
    # Generate form dynamically based on template
    for field in st.session_state.template:
        info = extracted.get(field, {"confidence": 0.0, "source": "none"})
        conf = get_conf_html(info["confidence"])
        src = f'<span class="source-tag">{info["source"]}</span>'
        
        st.markdown(f"{conf} {src}", unsafe_allow_html=True)
        # Use text_area if value is long or list-like (skills, etc.)
        val = form_values.get(field, "")
        if len(str(val)) > 80 or "\n" in str(val) or field in ["Skills", "Education", "Experience"]:
            form_values[field] = st.text_area(field, value=val, key=f"input_{field}")
        else:
            form_values[field] = st.text_input(field, value=val, key=f"input_{field}")
            
    st.markdown("---")
    
    # Validation
    validation = validate_all(form_values)
    if validation:
        st.markdown("#### ⚠️ Validation Warnings")
        for f, res in validation.items():
            if not res["is_valid"]:
                st.warning(f"**{f}**: {res['message']}")
                
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back"): set_step(3); st.rerun()
    with col2:
        if st.button("Export Data →", type="primary"): set_step(5); st.rerun()

# --- STEP 5: EXPORT ---
elif st.session_state.step == 5:
    st.markdown("### ✅ Step 5: Export Data")
    
    final_data = st.session_state.form_values
    
    # Prepare export files
    os.makedirs("output", exist_ok=True)
    
    json_str = json.dumps(final_data, indent=2)
    with open("output/final_submission.json", "w") as f:
        f.write(json_str)
        
    df = pd.DataFrame([final_data])
    csv_str = df.to_csv(index=False)
    with open("output/final_submission.csv", "w") as f:
        f.write(csv_str)
        
    st.success("Form submitted successfully!")
    
    st.download_button("📥 Download JSON", data=json_str, file_name="submission.json", mime="application/json")
    st.download_button("📥 Download CSV", data=csv_str, file_name="submission.csv", mime="text/csv")
    
    st.markdown("---")
    if st.button("Start New Submission"):
        for key in ["uploaded_files", "raw_text", "doc_type", "template", "extracted_data", "form_values"]:
            del st.session_state[key]
        set_step(1)
        st.rerun()
