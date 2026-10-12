# AI-Powered Career Intelligence & Interview Copilot

A beginner-friendly prototype that analyzes a resume against a job description, detects skills from a built-in vocabulary, calculates TF-IDF text similarity, highlights potential skill gaps, and generates interview practice questions.

## Current features

- Upload a resume as PDF or TXT, or paste resume text.
- Paste a job description.
- Extract known skills from both texts.
- Calculate TF-IDF + cosine similarity.
- Show overlapping skills and potential skill gaps.
- Generate role-specific interview practice questions.
- Download the questions as a text file.
- Runs locally; resume content is not sent to an external AI service.

## Setup on Windows

Install Python 3.11 or 3.12 from https://www.python.org/downloads/ and enable **Add Python to PATH** during installation.

Open Command Prompt in this folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The app should open in your browser. If it does not, visit `http://localhost:8501`.

## How it works

1. `pypdf` extracts text from PDF resumes.
2. A predefined skill vocabulary provides a simple baseline skill extractor.
3. `TfidfVectorizer` converts resume and job-description text into numerical vectors.
4. Cosine similarity measures textual similarity.
5. Set comparison highlights skills detected in the job description but not in the resume.
6. A question bank provides initial interview practice prompts.

## Important limitations

- The similarity score is not a probability of getting hired and is not model accuracy.
- The skill extractor uses a fixed vocabulary and may miss synonyms, context, and skills outside the list.
- The current question generator uses templates, not an LLM.
- A scanned/image-only PDF may require OCR, which is not included yet.
- Do not use this prototype as the sole basis for employment decisions.

## Planned next steps

1. Add a labeled dataset and evaluate the matching approach.
2. Add semantic matching with a pretrained Sentence Transformer.
3. Add RAG over trusted interview notes and role requirements.
4. Add optional LLM-powered feedback with safe API-key handling.
5. Separate processing endpoints into a FastAPI backend.
6. Add unit tests, sample data, and deployment instructions.
