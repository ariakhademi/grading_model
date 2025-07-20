import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import re
import numpy as np

# -------------------------------
# Sentence & word count checkers
# -------------------------------
def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return len([s for s in sentences if s.strip()])

def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

# -------------------------------
# Cached model loader
# -------------------------------
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

# -------------------------------
# Similarity computation
# -------------------------------
def compute_similarity(vec1, vec2, method="Cosine"):
    if method == "Cosine":
        raw = cosine_similarity([vec1], [vec2])[0][0]
        raw_min, raw_max = -1.0, 1.0
    elif method == "Euclidean":
        raw = euclidean_distances([vec1], [vec2])[0][0]
        raw_min, raw_max = 0.0, 2.0  # Approximate max dist in unit-normalized 384D space
    elif method == "Manhattan":
        raw = manhattan_distances([vec1], [vec2])[0][0]
        raw_min, raw_max = 0.0, 100.0  # Rough empirical estimate
    else:
        raise ValueError("Unknown similarity method.")

    # Normalize to [0, 1] (higher = better)
    if method == "Cosine":
        norm_score = (raw - raw_min) / (raw_max - raw_min)
    else:
        norm_score = 1 - (raw - raw_min) / (raw_max - raw_min)
        norm_score = max(0.0, min(1.0, norm_score))  # Clamp

    return norm_score, raw

# -------------------------------
# Keyword scoring penalty
# -------------------------------
def get_missing_keywords(ideal, candidate):
    ideal_keywords = set(re.findall(r'\b\w+\b', ideal.lower()))
    candidate_words = set(re.findall(r'\b\w+\b', candidate.lower()))
    missing = ideal_keywords - candidate_words
    return sorted(missing), len(missing), len(ideal_keywords)

# -------------------------------
# Final score scaling
# -------------------------------
def calculate_score(similarity_score, num_missing_keywords, total_keywords):
    keyword_penalty = (num_missing_keywords / max(total_keywords, 1)) * 0.4  # Up to 40% penalty
    penalized = max(similarity_score - keyword_penalty, 0.0)
    return round(penalized * 5), penalized  # Scaled score out of 5

# -------------------------------
# Example bank
# -------------------------------
examples = {
    "": {"question": "", "ideal": "", "candidate": ""},
    "Hypertension Diagnosis": {
        "question": "What criteria are used to diagnose hypertension?",
        "ideal": "Hypertension is diagnosed when blood pressure readings are consistently above 130/80 mmHg on at least two separate occasions.",
        "candidate": "High blood pressure is diagnosed if it stays above 130 over 80 multiple times."
    },
    "Diabetes Management": {
        "question": "How is type 2 diabetes managed?",
        "ideal": "Management includes lifestyle changes like diet and exercise, along with medications such as metformin.",
        "candidate": "Patients take metformin and try to eat better and exercise."
    }
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Automated Grading Prototype", layout="centered")
st.title("Automated Grading Prototype")
st.markdown("Grade short (1–3 sentence) medical responses using embeddings and similarity scoring.")

# Example selector
example_choice = st.selectbox("Try an example:", list(examples.keys()))
example = examples[example_choice]

# Model selection
model_name = st.selectbox("Select embedding model:", [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "pritamdeka/S-PubMedBERT-MS-MARCO"
])
model = load_model(model_name)

# Similarity metric
similarity_method = st.selectbox("Similarity method:", ["Cosine", "Euclidean", "Manhattan"])

# Input fields
question = st.text_input("Question:", example["question"])
ideal = st.text_area("Ideal Answer (1–3 sentences):", value=example["ideal"], height=80)
candidate = st.text_area("Candidate Response (1–3 sentences):", value=example["candidate"], height=80)

# Input checks
if ideal and count_sentences(ideal) > 3:
    st.warning("Ideal answer exceeds 3 sentences.")
if candidate and count_sentences(candidate) > 3:
    st.warning("Candidate response exceeds 3 sentences.")
if candidate and count_words(candidate) < 3:
    st.warning("Candidate answer is too short (<3 words).")

# Grade
if st.button("Grade Answer") and ideal and candidate:
    if count_sentences(ideal) > 3 or count_sentences(candidate) > 3:
        st.stop()

    # Embeddings
    ideal_vec = model.encode(ideal)
    candidate_vec = model.encode(candidate)

    # Similarity
    normalized_score, raw_score = compute_similarity(ideal_vec, candidate_vec, method=similarity_method)

    # Keyword comparison
    missing_keywords, num_missing, total_keywords = get_missing_keywords(ideal, candidate)

    # Final score
    scaled_score, penalized_score = calculate_score(normalized_score, num_missing, total_keywords)

    # Feedback
    st.markdown("---")
    st.markdown(f"**Similarity Method:** {similarity_method}")
    st.markdown(f"**Raw Similarity Score:** {round(raw_score, 4)}")
    st.markdown(f"**Normalized Score (pre-penalty):** {round(normalized_score, 4)}")
    st.markdown(f"**Missing Keywords ({num_missing}/{total_keywords}):** {', '.join(missing_keywords) if missing_keywords else 'None'}")
    st.markdown(f"**Final Score (with keyword penalty):** {scaled_score}/5")
