import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, cityblock
from difflib import SequenceMatcher
import numpy as np

# Preload models (load lazily when selected)
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

# App title
st.title("Short Answer Grading App")

# Input fields
ideal_answer = st.text_area("Ideal Answer:", height=100)
user_answer = st.text_area("User Answer:", height=100)

# Model selection
model_name = st.selectbox(
    "Choose Embedding Model:",
    [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
    ],
)

# Similarity method
similarity_method = st.selectbox(
    "Choose Similarity Method:",
    [
        "Cosine Similarity",
        "Euclidean Similarity",
        "Manhattan Similarity",
        "Jaccard Similarity (word overlap)",
        "Levenshtein Similarity (edit distance)",
    ]
)

# --- Similarity Functions ---

def get_cosine_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def get_euclidean_similarity(vec1, vec2):
    dist = euclidean(vec1, vec2)
    return 1 / (1 + dist)

def get_manhattan_similarity(vec1, vec2):
    dist = cityblock(vec1, vec2)
    return 1 / (1 + dist)

def get_jaccard_similarity(a, b):
    set1 = set(a.lower().split())
    set2 = set(b.lower().split())
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union else 0

def get_levenshtein_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Grading button
if st.button("Grade Answer"):
    if not ideal_answer.strip() or not user_answer.strip():
        st.warning("Please enter both answers.")
    else:
        # Load the selected model
        model = load_model(model_name)
        
        if similarity_method in [
            "Cosine Similarity",
            "Euclidean Similarity",
            "Manhattan Similarity"
        ]:
            # Compute embeddings
            vec1, vec2 = model.encode([ideal_answer, user_answer])
            
            # Apply the selected metric
            if similarity_method == "Cosine Similarity":
                score = get_cosine_similarity(vec1, vec2)
            elif similarity_method == "Euclidean Similarity":
                score = get_euclidean_similarity(vec1, vec2)
            elif similarity_method == "Manhattan Similarity":
                score = get_manhattan_similarity(vec1, vec2)
        else:
            # Text-level metrics
            if similarity_method == "Jaccard Similarity (word overlap)":
                score = get_jaccard_similarity(ideal_answer, user_answer)
            elif similarity_method == "Levenshtein Similarity (edit distance)":
                score = get_levenshtein_similarity(ideal_answer, user_answer)
        
        # Scale score to 0–5
        scaled_score = round(score * 5, 2)
        st.markdown(f"### Score: {scaled_score} / 5.00")

        # Feedback
        if scaled_score > 4.0:
            st.success("Excellent match.")
        elif scaled_score > 3.0:
            st.info("Good, but some differences.")
        elif scaled_score > 2.0:
            st.warning("Partially correct.")
        else:
            st.error("Poor match. Needs review.")
