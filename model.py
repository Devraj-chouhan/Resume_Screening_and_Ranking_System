import spacy
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load NLP model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Remove stopwords, punctuation, and lemmatize text."""
    if not text:  # Handle empty text case
        return ""

    doc = nlp(text.lower())
    cleaned_text = " ".join(
        [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    )
    
    # Debug: Print preprocessed text
    print(f"🔍 Preprocessed Text (First 500 chars):\n{cleaned_text[:500]}")  

    return cleaned_text

def rank_resumes(job_desc, resumes):
    """Rank resumes based on similarity to job description using TF-IDF & cosine similarity."""
    
    # Handle empty input cases
    if not job_desc or not resumes:
        print("⚠️ Warning: Job description or resumes are empty.")
        return []

    # Preprocess job description & resumes
    all_texts = [preprocess_text(job_desc)] + [preprocess_text(resumes[name]) for name in resumes]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),  # Use unigrams & bigrams
        max_df=0.85,  # Ignore terms in >85% of resumes
        min_df=2,  # Ignore rare words
        sublinear_tf=True  # Apply logarithmic scaling
    )

    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Debugging: Print shape of the matrix
    print(f"📊 TF-IDF Matrix Shape: {tfidf_matrix.shape}")

    job_vec = tfidf_matrix[0].reshape(1, -1)
    resume_vecs = tfidf_matrix[1:]

    scores = cosine_similarity(job_vec, resume_vecs).flatten()
    
    # Normalize scores safely
    if np.max(scores) != np.min(scores):
        scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    else:
        scores = np.zeros_like(scores)  # Set all scores to zero if identical

    ranked_resumes = sorted(zip(resumes.keys(), scores), key=lambda x: x[1], reverse=True)

    # Debug: Print ranking
    print("\n🏆 Ranking Scores:")
    for name, score in ranked_resumes:
        print(f"📌 {name}: {score:.4f}")

    return ranked_resumes
