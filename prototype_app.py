import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, cityblock
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Title
st.title("Short Answer Grading Prototype")

# Input fields
ideal_answer = st.text_area("Ideal Answer:", height=100)
user_answer = st.text_area("User Answer:", height=100)

# Choose similarity metric
similarity_method = st.selectbox(
    "Choose similarity method:",
    [
        "Cosine Similarity",
        "Euclidean Similarity",
        "Manhattan Similarity",
        "Jaccard Similarity (words)",
        "Levenshtein Similarity (normalized edit distance)",
    ]
)

def get_cosine_similarity(a, b):
    embeddings = model.encode([a, b])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

def get_euclidean_similarity(a, b):
    embeddings = model.encode([a, b])
    dist = euclidean(embeddings[0], embeddings[1])
    return 1 / (1 + dist)

def get_manhattan_similarity(a, b):
    embeddings = model.encode([a, b])
    dist = cityblock(embeddings[0], embeddings[1])
    return 1 / (1 + dist)

def get_jaccard_similarity(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    intersection = len(a_set & b_set)
    union = len(a_set | b_set)
    return intersection / union if union != 0 else 0

def get_levenshtein_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Evaluate
if st.button("Grade Answer"):
    if not ideal_answer.strip() or not user_answer.strip():
        st.warning("Please fill in both fields.")
    else:
        # Choose similarity function
        if similarity_method == "Cosine Similarity":
            score = get_cosine_similarity(ideal_answer, user_answer)
        elif similarity_method == "Euclidean Similarity":
            score = get_euclidean_similarity(ideal_answer, user_answer)
        elif similarity_method == "Manhattan Similarity":
            score = get_manhattan_similarity(ideal_answer, user_answer)
        elif similarity_method == "Jaccard Similarity (words)":
            score = get_jaccard_similarity(ideal_answer, user_answer)
        elif similarity_method == "Levenshtein Similarity (normalized edit distance)":
            score = get_levenshtein_similarity(ideal_answer, user_answer)
        else:
            score = 0

        # Scale to 0–5
        score_scaled = round(score * 5, 2)
        st.markdown(f"### Score: {score_scaled} / 5.00")

        # Feedback
        if score_scaled > 4.0:
            st.success("Excellent match.")
        elif score_scaled > 3.0:
            st.info("Good, minor differences.")
        elif score_scaled > 2.0:
            st.warning("Partially correct.")
        else:
            st.error("Low similarity. Review needed.")
