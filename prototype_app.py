import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, cityblock
from difflib import SequenceMatcher
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load and cache models
@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

# Similarity functions
def cosine_sim(v1, v2): return cosine_similarity([v1], [v2])[0][0]
def euclidean_sim(v1, v2): return 1 / (1 + euclidean(v1, v2))
def manhattan_sim(v1, v2): return 1 / (1 + cityblock(v1, v2))
def jaccard_sim(a, b):
    s1, s2 = set(a.lower().split()), set(b.lower().split())
    return len(s1 & s2) / len(s1 | s2) if s1 | s2 else 0
def levenshtein_sim(a, b): return SequenceMatcher(None, a, b).ratio()

# Title
st.title("Short Answer Grading with Multi-Metric and Visualization")

# Inputs
ideal = st.text_area("Ideal Answer:", height=100)
user = st.text_area("User Answer:", height=100)

# Model selector
model_name = st.selectbox("Embedding Model", [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
])

# Similarity options
metrics_selected = st.multiselect(
    "Choose similarity metrics (select 1 or more):",
    ["Cosine", "Euclidean", "Manhattan", "Jaccard", "Levenshtein"],
    default=["Cosine"]
)

# Grade button
if st.button("Grade Answer"):
    if not ideal.strip() or not user.strip():
        st.warning("Both fields are required.")
    else:
        model = load_model(model_name)
        vec1, vec2 = model.encode([ideal, user])

        scores = {}
        for metric in metrics_selected:
            if metric == "Cosine":
                scores["Cosine"] = cosine_sim(vec1, vec2)
            elif metric == "Euclidean":
                scores["Euclidean"] = euclidean_sim(vec1, vec2)
            elif metric == "Manhattan":
                scores["Manhattan"] = manhattan_sim(vec1, vec2)
            elif metric == "Jaccard":
                scores["Jaccard"] = jaccard_sim(ideal, user)
            elif metric == "Levenshtein":
                scores["Levenshtein"] = levenshtein_sim(ideal, user)

        # Scale each to 0–5 and average
        scaled_scores = {k: round(v * 5, 2) for k, v in scores.items()}
        avg_score = round(np.mean(list(scaled_scores.values())), 2)

        # Display scores
        st.subheader("Similarity Scores")
        for k, v in scaled_scores.items():
            st.write(f"**{k} Similarity**: {v} / 5.00")

        st.markdown(f"### 🧮 **Final Averaged Score**: {avg_score} / 5.00")

        # Feedback
        if avg_score > 4.0:
            st.success("Excellent match.")
        elif avg_score > 3.0:
            st.info("Good, but some differences.")
        elif avg_score > 2.0:
            st.warning("Partially correct.")
        else:
            st.error("Poor match. Needs review.")

        # --- Embedding Visualization ---
        st.subheader("📊 Embedding Visualization (2D)")
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform([vec1, vec2])
        labels = ['Ideal Answer', 'User Answer']

        fig, ax = plt.subplots()
        ax.scatter(reduced[:, 0], reduced[:, 1], color=["green", "blue"])
        for i, txt in enumerate(labels):
            ax.annotate(txt, (reduced[i, 0]+0.01, reduced[i, 1]+0.01))
        ax.set_title("2D PCA of Sentence Embeddings")
        st.pyplot(fig)
