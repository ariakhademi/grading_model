import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------------
# Sentence count checker
# -------------------------------
def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return len([s for s in sentences if s.strip()])

# -------------------------------
# Similarity computation
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
# Score scaling
# -------------------------------
def calculate_score(similarity, method="Cosine"):
    if method == "Cosine":
        return round(similarity * 5)
    else:
        scaled = max(min((similarity + 10) / 10, 1.0), 0.0)
        return round(scaled * 5)

# -------------------------------
# Missing keyword detection
# -------------------------------
def get_missing_keywords(ideal, candidate):
    ideal_keywords = set(re.findall(r'\b\w+\b', ideal.lower()))
    candidate_words = set(re.findall(r'\b\w+\b', candidate.lower()))
    missing = ideal_keywords - candidate_words
    return sorted(missing)

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

# Embedding model selection
model_name = st.selectbox("Select embedding model:", [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "pritamdeka/S-PubMedBERT-MS-MARCO"
])

model = SentenceTransformer(model_name)

# Similarity method
similarity_method = st.selectbox("Similarity method:", ["Cosine", "Euclidean", "Manhattan"])

# Input fields
question = st.text_input("Question:", example["question"])
ideal = st.text_area("Ideal Answer (1–3 sentences):", value=example["ideal"], height=80)
candidate = st.text_area("Candidate Response (1–3 sentences):", value=example["candidate"], height=80)

# Sentence constraint warnings
if ideal and count_sentences(ideal) > 3:
    st.error("Ideal answer exceeds 3 sentences.")
if candidate and count_sentences(candidate) > 3:
    st.error("Candidate response exceeds 3 sentences.")

# Grade response
if st.button("Grade Answer") and ideal and candidate:
    if count_sentences(ideal) > 3 or count_sentences(candidate) > 3:
        st.stop()

    # Embedding + similarity
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
    st.markdown(f"**Similarity ({similarity_method}):** {round(similarity, 4)}")
    st.markdown(f"**Feedback:** {feedback}")
