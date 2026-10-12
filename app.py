import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Career Copilot", page_icon="🧭", layout="wide")

SKILL_BANK = [
    "Python", "Java", "C++", "JavaScript", "SQL", "MySQL", "HTML", "CSS",
    "React", "Node.js", "FastAPI", "Flask", "Django", "Spring Boot",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "Keras", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
    "Matplotlib", "Data Analysis", "Data Visualization", "Statistics",
    "Git", "GitHub", "Docker", "AWS", "Azure", "REST API", "MongoDB",
    "PostgreSQL", "Power BI", "Excel", "Generative AI", "LLM", "RAG",
    "Prompt Engineering", "Data Structures", "Algorithms", "OOP",
    "Problem Solving", "Communication", "Agile", "Linux"
]

QUESTION_BANK = {
    "Python": [
        "Explain the difference between a list and a tuple in Python.",
        "What are decorators and where might you use them?",
        "How would you handle exceptions in a Python application?"
    ],
    "Machine Learning": [
        "Explain the difference between supervised and unsupervised learning.",
        "What is overfitting, and how can you reduce it?",
        "Why do we split data into training and test sets?"
    ],
    "Deep Learning": [
        "What is a neural network, and what do its layers do?",
        "What is the role of an activation function?",
        "How are epochs, batches, and learning rate different?"
    ],
    "NLP": [
        "What is text vectorization, and why is it needed?",
        "Compare TF-IDF with contextual embeddings.",
        "What is the attention mechanism used by Transformers?"
    ],
    "Generative AI": [
        "What is an LLM, and what are its limitations?",
        "What makes a prompt clear and reliable?",
        "How would you reduce hallucinations in an AI application?"
    ],
    "SQL": [
        "Explain WHERE versus HAVING.",
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "When would you use GROUP BY?"
    ],
    "Web/API": [
        "What is a REST API?",
        "How would you validate input in an API?",
        "How would you handle an API failure gracefully?"
    ],
    "General": [
        "Walk me through a project you built and the problem it solves.",
        "Describe a technical challenge you faced and how you solved it.",
        "How would you test and monitor a machine-learning application?"
    ]
}

def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    if name.endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return ""

def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()

def find_skills(text):
    normalized = clean_text(text).lower()
    found = []
    for skill in SKILL_BANK:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(found, key=str.lower)

def calculate_match(resume_text, job_text):
    if not clean_text(resume_text) or not clean_text(job_text):
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([resume_text, job_text])
    return float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0] * 100)

def generate_questions(missing_skills, role):
    questions = []
    for skill in missing_skills[:4]:
        bank = QUESTION_BANK.get(skill, [])
        if bank:
            questions.append((skill, bank[0]))
        else:
            questions.append((skill, f"What is {skill}, and how would you use it in a project?"))
    questions.extend([
        ("Project", f"Explain a project that makes you a strong candidate for the {role} role."),
        ("Problem solving", "Tell me about a bug or difficult problem you solved. What steps did you take?")
    ])
    return questions

st.title("🧭 AI-Powered Career Intelligence & Interview Copilot")
st.caption("A beginner-friendly prototype: resume parsing, skill extraction, ML text matching, and interview practice.")

with st.sidebar:
    st.header("Candidate profile")
    role = st.text_input("Target role", "Applied AI Engineer")
    st.markdown("---")
    st.markdown("**Current modules**")
    st.write("✓ PDF/TXT resume parsing")
    st.write("✓ Skill extraction baseline")
    st.write("✓ TF-IDF + cosine similarity")
    st.write("✓ Interview question suggestions")
    st.info("This baseline runs locally. It does not send your resume to an external AI service.")

tab1, tab2, tab3 = st.tabs(["Resume & Job Match", "Interview Copilot", "How it works"])

with tab1:
    left, right = st.columns(2)
    with left:
        uploaded = st.file_uploader("Upload your resume (PDF or TXT)", type=["pdf", "txt"])
        resume_text_area = st.text_area("Or paste resume text", height=220, placeholder="Paste your resume text here...")
    with right:
        job_text = st.text_area("Paste the job description", height=300, placeholder="Paste the target job description here...")
    resume_from_file = extract_text(uploaded) if uploaded else ""
    resume_text = resume_text_area.strip() or resume_from_file

    if uploaded and resume_from_file and not resume_text_area.strip():
        st.success(f"Read text from {uploaded.name} ({len(resume_from_file)} characters).")

    if st.button("Analyze resume", type="primary", use_container_width=True):
        if not resume_text:
            st.error("Please upload a PDF/TXT resume or paste resume text.")
        elif not job_text.strip():
            st.error("Please paste a job description.")
        else:
            resume_skills = find_skills(resume_text)
            job_skills = find_skills(job_text)
            missing = sorted(set(job_skills) - set(resume_skills), key=str.lower)
            overlap = sorted(set(job_skills) & set(resume_skills), key=str.lower)
            score = calculate_match(resume_text, job_text)
            st.session_state["analysis"] = {
                "resume_text": resume_text,
                "job_text": job_text,
                "resume_skills": resume_skills,
                "job_skills": job_skills,
                "missing": missing,
                "overlap": overlap,
                "score": score,
                "role": role.strip() or "target role",
            }

    analysis = st.session_state.get("analysis")
    if analysis:
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Text similarity", f"{analysis['score']:.1f}%")
        c2.metric("Matched listed skills", len(analysis["overlap"]))
        c3.metric("Potential skill gaps", len(analysis["missing"]))
        st.caption("Text similarity is a TF-IDF similarity score, not a hiring probability or model accuracy.")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Skills detected in resume")
            st.write(", ".join(analysis["resume_skills"]) if analysis["resume_skills"] else "No listed skills detected. Try including standard skill names.")
            st.subheader("Skills that overlap")
            st.write(", ".join(analysis["overlap"]) if analysis["overlap"] else "No overlap detected from the built-in skill list.")
        with col2:
            st.subheader("Skills mentioned in the job description")
            st.write(", ".join(analysis["job_skills"]) if analysis["job_skills"] else "No listed skills detected in the job description.")
            st.subheader("Potential gaps to review")
            st.write(", ".join(analysis["missing"]) if analysis["missing"] else "No gaps detected from the built-in skill list.")
        if analysis["missing"]:
            st.subheader("Suggested learning priorities")
            st.dataframe(pd.DataFrame({
                "Skill to review": analysis["missing"],
                "Suggested next step": [f"Study the basics of {s}, then build a small demonstrable example." for s in analysis["missing"]]
            }), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Practice for your target role")
    st.write("Use the results from the Resume & Job Match tab, or enter skills manually.")
    manual_skills = st.text_input("Skills/topics to practice (comma-separated)", "Python, Machine Learning, Deep Learning, Generative AI, SQL")
    if st.button("Generate practice questions", type="primary"):
        skills = [s.strip() for s in manual_skills.split(",") if s.strip()]
        questions = []
        for skill in skills[:6]:
            bank = QUESTION_BANK.get(skill, [])
            if bank:
                questions.extend([(skill, q) for q in bank[:2]])
            else:
                questions.append((skill, f"Explain {skill} and describe one practical use case."))
        questions.extend(generate_questions([], role))
        st.session_state["questions"] = questions

    for i, (topic, question) in enumerate(st.session_state.get("questions", []), start=1):
        with st.expander(f"{i}. {topic}"):
            st.write(question)
            st.text_area("Your practice answer", key=f"answer_{i}", height=100, placeholder="Write your answer here...")
    if st.session_state.get("questions"):
        st.download_button(
            "Download interview questions",
            "\n".join(f"{i}. [{topic}] {question}" for i, (topic, question) in enumerate(st.session_state["questions"], start=1)),
            file_name="interview_practice_questions.txt",
            mime="text/plain",
            use_container_width=True,
        )

with tab3:
    st.subheader("Current pipeline")
    st.code(
        "Resume PDF/TXT + Job Description\n"
        "          ↓\n"
        "Text extraction and normalization\n"
        "          ↓\n"
        "Skill dictionary baseline + TF-IDF vectorization\n"
        "          ↓\n"
        "Cosine similarity + skill gap comparison\n"
        "          ↓\n"
        "Interview question suggestions",
        language="text",
    )
    st.markdown("**Next upgrades:** semantic embeddings with Sentence Transformers, a labeled evaluation dataset, a retrieval-augmented interview assistant (RAG), an optional LLM provider, and a FastAPI service.")
    st.warning("This is a learning prototype. Skill extraction depends on a predefined vocabulary and can miss synonyms or context. Do not use its score as the sole basis for hiring decisions.")
