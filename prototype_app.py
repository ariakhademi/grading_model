import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------------
# Function: Count sentences
# -------------------------------
def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return len([s for s in sentences if s.strip()])

# -------------------------------
# Function: Compute similarity
# -------------------------------
def compute_similarity(vec1, vec2, method="Cosine"):
    if method == "Cosine":
        return cosine_similarity([vec1], [vec2])[0][0]
    elif method == "Euclidean":
        return -euclidean_distances([vec1], [vec2])[0][0]  # Negated for scoring
    elif method == "Manhattan":
        return -manhattan_distances([vec1], [vec2])[0][0]  # Negated for scoring
    else:
        raise ValueError("Unknown similarity method.")

# -------------------------------
# Function: Calculate score
# -------------------------------
def calculate_score(similarity, method="Cosine"):
    if method == "Cosine":
        return round(similarity * 5)
    else:
        scaled = max(min((similarity + 10) / 10, 1.0), 0.0)
        return round(scaled * 5)

# -------------------------------
# Function: Keyword feedback
# -------------------------------
def get_missing_keywords(ideal, candidate):
    ideal_keywords = set(re.findall(r'\b\w+\b', ideal.lower()))
    candidate_words = set(re.findall(r'\b\w+\b', candidate.lower()))
    missing = ideal_keywords - candidate_words
    return sorted(missing)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Automated Grading Prototype", layout="centered")
st.title("Automated Grading Prototype")
st.markdown("Grade short (1–3 sentence) free-text medical responses using embeddings.")

# Embedding model selection
model_name = st.selectbox("Select embedding model:", [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2"
])
model = SentenceTransformer(model_name)

# Similarity metric selection
similarity_method = st.selectbox("Similarity method:", ["Cosine", "Euclidean", "Manhattan"])

# Input fields
question = st.text_input("Question:", "")
ideal = st.text_area("Ideal Answer (1–3 sentences):", height=80)
candidate = st.text_area("Candidate Response (1–3 sentences):", height=80)

# Sentence count check
if ideal and count_sentences(ideal) > 3:
    st.error("Ideal answer exceeds 3 sentences.")
if candidate and count_sentences(candidate) > 3:
    st.error("Candidate response exceeds 3 sentences.")

# Grade button
if st.button("Grade Answer") and ideal and candidate:
    if count_sentences(ideal) > 3 or count_sentences(candidate) > 3:
        st.stop()

    # Compute embeddings and similarity
    ideal_vec = model.encode(ideal)
    candidate_vec = model.encode(candidate)
    similarity = compute_similarity(ideal_vec, candidate_vec, method=similarity_method)
    score = calculate_score(similarity, method=similarity_method)

    # Feedback
    missing_keywords = get_missing_keywords(ideal, candidate)
    feedback = f"Score: {score}/5"
    if missing_keywords:
        feedback += f". Missing keywords: {', '.join(missing_keywords)}."

    st.markdown("---")
    st.markdown(f"**Model Score:** {score}/5")
    st.markdown(f"**Feedback:** {feedback}")