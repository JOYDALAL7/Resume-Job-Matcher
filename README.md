# Resume vs Job Description Matcher

A smart Streamlit app that semantically compares a resume PDF to a job description using Sentence Transformers (SBERT).

## Features
- Upload your resume (PDF)
- Paste a job description
- Get a match score using semantic similarity
- Uses open-source `all-MiniLM-L6-v2` model
- No OpenAI or paid API

## Setup

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

## Model
Using SBERT: [all-MiniLM-L6-v2](https://www.sbert.net/docs/pretrained_models.html)
