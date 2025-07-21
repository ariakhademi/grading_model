import streamlit as st
from graphviz import Digraph
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances
import re

# 1-3 sentence constraint
def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    return len([s for s in sentences if s.strip()])

# counts whole words in a sentence
def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

# cached model loader
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

# similarity measures
def compute_similarity(vec1, vec2, method="Cosine"):
    if method == "Cosine":
        raw = cosine_similarity([vec1], [vec2])[0][0] # extracts raw number
        raw_min, raw_max = -1.0, 1.0
    elif method == "Euclidean":
        raw = euclidean_distances([vec1], [vec2])[0][0] # extracts raw number
        raw_min, raw_max = 0.0, 2.0  # theoretical upper bound under normalization
    elif method == "Manhattan":
        raw = manhattan_distances([vec1], [vec2])[0][0] # extracts raw number
        raw_min, raw_max = 0.0, 100.0  # rough empirical estimate
    else:
        raise ValueError("Unknown similarity method.")

    # normalize to [0, 1] 
    if method == "Cosine":
        norm_score = (raw - raw_min) / (raw_max - raw_min)
    else:
        norm_score = 1 - (raw - raw_min) / (raw_max - raw_min)
        norm_score = max(0.0, min(1.0, norm_score))  # clamps to [0,1]

    return norm_score, raw

# penalty for missing words
def get_missing_keywords(ideal, candidate):
    ideal_keywords = set(re.findall(r'\b\w+\b', ideal.lower()))
    candidate_words = set(re.findall(r'\b\w+\b', candidate.lower()))
    missing = ideal_keywords - candidate_words
    return sorted(missing), len(missing), len(ideal_keywords)

# final score after penalty
def calculate_score(similarity_score, num_missing_keywords, total_keywords):
    keyword_penalty = (num_missing_keywords / max(total_keywords, 1)) * 0.4  # 40% penalty
    penalized_score = max(similarity_score - keyword_penalty, 0.0)

    # add interpretation based on thresholds
    if penalized_score >= 0.8:
        label = "Excellent answer, good job."
    elif penalized_score >= 0.5:
        label = "Fair answer, doing good."
    else:
        label = "Needs improvement, alert the board of trustees."

    return penalized_score, label

# bank of questions, ideal answers, and candidate answers
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
    },
    "Red Blood Cells": {
        "question": "What is the main function of red blood cells?",
        "ideal": "Red blood cells carry oxygen from the lungs to tissues throughout the body. They use hemoglobin to bind oxygen.",
        "candidate": "They transport oxygen using hemoglobin."
    },
    "Insulin Functionality": {
        "question": "How does insulin regulate blood sugar?",
        "ideal": "Insulin lowers blood glucose by helping cells absorb sugar from the bloodstream. It stimulates the liver to store glucose as glycogen. This keeps blood sugar levels within a healthy range.",
        "candidate": "Insulin helps cells absorb glucose. It also stores sugar in the liver."
    },
    "Allergic Reaction": {
        "question": "What happens during an allergic reaction?",
        "ideal": "The immune system identifies a harmless substance as a threat. It releases histamine and other chemicals. This causes symptoms like swelling, itching, and sneezing.",
        "candidate": "Histamines are released. You get symptoms like hives or sneezing."
    },
    "Vaccination": {
        "question": "What is the purpose of a vaccine?",
        "ideal": "Vaccines help the immune system recognize and fight specific pathogens.",
        "candidate": "Vaccines train the body to fight infections. They don’t actually cause the disease."
    },
    "Antibiotics Functionality": {
        "question": "How do antibiotics work?",
        "ideal": "Antibiotics target bacteria and either kill them or prevent them from multiplying. They interfere with bacterial cell walls or proteins. They are not effective against viruses.",
        "candidate": "They kill bacteria but don’t work on viruses."
    }
}

# user interface with streamlit
st.set_page_config(page_title="Automated Grading Prototype", layout="centered")
st.title("Automated Grading Prototype")
st.markdown("Grade short (1–3 sentence) medical responses using embeddings and similarity scoring.")

# select an example
example_choice = st.selectbox("Try an example:", list(examples.keys()))
example = examples[example_choice]

# select a semantic embedding model
model_name = st.selectbox("Select embedding model:", [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "paraphrase-MiniLM-L6-v2",
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "pritamdeka/S-PubMedBERT-MS-MARCO"
])

# load the model
model = load_model(model_name)

# choose a similarity metric
similarity_method = st.selectbox("Similarity method:", ["Cosine", "Euclidean", "Manhattan"])

# input fields
question = st.text_input("Question:", example["question"])
ideal = st.text_area("Ideal Answer (1–3 sentences):", value=example["ideal"], height=80)
candidate = st.text_area("Candidate Response (1–3 sentences):", value=example["candidate"], height=80)

# input checks for constraints
if ideal and count_sentences(ideal) > 3:
    st.warning("Ideal answer exceeds 3 sentences.")
if candidate and count_sentences(candidate) > 3:
    st.warning("Candidate response exceeds 3 sentences.")

# grade answers
if st.button("Grade Answer") and ideal and candidate:
    if count_sentences(ideal) > 3 or count_sentences(candidate) > 3:
        st.stop()

    # obtain embeddings of answers
    ideal_vec = model.encode(ideal)
    candidate_vec = model.encode(candidate)

    # check similarity of embeddings
    normalized_score, raw_score = compute_similarity(ideal_vec, candidate_vec, method=similarity_method)

    # keyword comparison
    missing_keywords, num_missing, total_keywords = get_missing_keywords(ideal, candidate)

    # final score
    final_normalized_score, interpretation = calculate_score(normalized_score, num_missing, total_keywords)

    # feedback
    st.markdown("---")
    st.markdown(f"**Similarity Method:** {similarity_method}")
    st.markdown(f'**Transformer Embedding Method:** {model_name}')
    st.markdown(f"**Normalized Score (pre-penalty):** {round(normalized_score, 4)}")
    st.markdown(f"**Missing Keywords ({num_missing}/{total_keywords}):** {', '.join(missing_keywords) if missing_keywords else 'None'}")
    st.markdown(f"**Final Normalized Score (after penalty):** {round(final_normalized_score, 4)}")
    st.subheader("🔍 Final Score")
    st.progress(final_normalized_score)
    # optional: Color-coded label for qualitative interpretation
    if final_normalized_score >= 0.85:
        label = "✅ Excellent"
        color = "green"
    elif final_normalized_score >= 0.5:
        label = "⚠️ Fair"
        color = "orange"
    else:
        label = "❌ Needs Improvement"
        color = "red"

    st.markdown(f"**Interpretation (Excellent, fair, bad):** {interpretation}")