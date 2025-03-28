import streamlit as st
import PyPDF2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Ensure NLTK stopwords are available
nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

# 🟢 Function to Extract Text from PDF
def extract_text_from_pdf(uploaded_file):
    """Extracts and cleans text from a PDF file."""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
        text = " ".join(text.split())  # Remove extra spaces
        return text if text else "No extractable text found."
    except PyPDF2.errors.PdfReadError:
        return "Error: Cannot extract text from encrypted or scanned PDF."
    except Exception as e:
        return f"Error reading PDF: {e}"

# 🟢 Function to Preprocess Text
def preprocess_text(text):
    """Cleans text by removing special characters and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.lower().strip()

# 🟢 Function to Rank Resumes using KNN & Cosine Similarity
def rank_resumes_knn(job_description, resumes, resume_names, k=10):
    """Ranks resumes based on TF-IDF, KNN, and Cosine Similarity."""
    vectorizer = TfidfVectorizer(
        stop_words="english", 
        lowercase=True, 
        ngram_range=(1, 3), 
        max_df=0.9, 
        min_df=2  # Adjusted to avoid rare words affecting results
    )

    # Preprocess job description & resumes
    corpus = [preprocess_text(job_description)] + [preprocess_text(resume) for resume in resumes]
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Apply NearestNeighbors
    knn = NearestNeighbors(n_neighbors=min(k, len(resumes)), metric="cosine")
    knn.fit(tfidf_matrix[1:])  # Fit KNN only on resumes

    # Find closest matches
    distances, indices = knn.kneighbors(tfidf_matrix[0], n_neighbors=min(k, len(resumes)))
    scores = 1 - distances.flatten()  # Convert distances to similarity scores

    ranked_results = sorted(zip(range(len(resumes)), scores), key=lambda x: x[1], reverse=True)
    ranked_resumes = [(resume_names[i], score) for i, score in ranked_results]

    return ranked_resumes, [sim for _, sim in ranked_resumes]

# 🟢 Function to Plot Resume Scores
def plot_resume_scores(resume_names, scores):
    """Generates a bar chart for resume ranking scores."""
    st.subheader("📊 Resume Score Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=scores, y=resume_names, palette="coolwarm", ax=ax)
    ax.set_xlabel("Similarity Score", fontsize=12)
    ax.set_ylabel("Resume Name", fontsize=12)
    ax.set_title("Resume Ranking Based on Job Description", fontsize=14)
    st.pyplot(fig)

# 🟢 Custom Styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #EBE9E1;
        color: #E43D12;
    }
    
    h1, h2, h3, h4 {
        color: #E43D12 !important;
        text-align: center;
        font-weight: bold;
    }

    .stTextInput, .stTextArea, .stFileUploader {
        background-color: #D6536D !important;
        color: white !important;
        border-radius: 5px;
    }

    .stButton>button {
        background-color: #FFA2B6 !important;
        color: white !important;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
    }

    .stButton>button:hover {
        background-color: #EFB11D !important;
        color: black !important;
    }

    .stTable {
        background-color: #D6536D !important;
        color: white !important;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 🟢 App Title & Description
st.title("AI-powered Resume Screening and Ranking System")
st.subheader("Upload resumes and rank candidates instantly!")

# 🎯 Job Description Input
job_description = st.text_area("Enter Job Description")

# 📂 File Upload Section
uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, type=["pdf"])

if st.button("Rank Resumes") and job_description and uploaded_files:
    with st.spinner("🔄 Processing resumes... Please wait."):
        time.sleep(3)  # Simulated delay
        
        resumes_text = []
        resume_names = []

        for uploaded_file in uploaded_files:
            text = extract_text_from_pdf(uploaded_file)
            resumes_text.append(text)
            resume_names.append(uploaded_file.name)

        ranked_resumes, scores = rank_resumes_knn(job_description, resumes_text, resume_names)

    st.success("✅ Resumes ranked successfully!")

    # 📜 Display Ranked Resumes
    st.subheader("📜 Ranked Resumes")
    ranked_results = pd.DataFrame({
        "Resume": [name for name, _ in ranked_resumes],  # Use name directly
        "Score": [round(score, 3) for _, score in ranked_resumes]  # Use score directly
    })
    st.table(ranked_results)

    # 📊 Plot Bar Chart
    plot_resume_scores(
        [name for name, _ in ranked_resumes],
        [score for _, score in ranked_resumes]
    )
