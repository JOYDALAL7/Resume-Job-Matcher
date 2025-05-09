import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import os

st.set_page_config(page_title="Resume vs Job Matcher", layout="wide")
st.title("📄 Resume vs Job Description Matcher")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

def compute_similarity(resume_text, job_desc):
    embeddings = model.encode([resume_text, job_desc], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return round(score * 100, 2)

with st.sidebar:
    st.header("Upload Inputs")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("Paste Job Description", height=300)

if resume_file and job_description:
    with st.spinner("Analyzing..."):
        resume_text = extract_text_from_pdf(resume_file)
        score = compute_similarity(resume_text, job_description)
        st.success(f"✅ Match Score: {score}%")
        if score > 75:
            st.markdown("🎯 **Great match! You’re ready to apply.**")
        elif score > 50:
            st.markdown("⚠️ **Partial match. Consider tailoring your resume.**")
        else:
            st.markdown("❌ **Low match. Consider revising your resume or applying elsewhere.**")

    with st.expander("🔍 Extracted Resume Text"):
        st.text(resume_text)
