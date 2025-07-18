import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler

# Load pre-trained sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Title
st.title("Short Answer Grading Prototype")

# Input fields
ideal_answer = st.text_area("Enter the ideal answer (provided by exam board):", height=100)
user_answer = st.text_area("Enter the candidate's answer:", height=100)

# Button to evaluate
if st.button("Grade Answer"):
    if not ideal_answer.strip() or not user_answer.strip():
        st.warning("Please enter both the ideal and user answers.")
    else:
        # Encode both answers
        embeddings = model.encode([ideal_answer, user_answer], convert_to_tensor=True)
        
        # Calculate cosine similarity
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        
        # Normalize to a 0–5 scale
        scaler = MinMaxScaler((0, 5))
        scaled_score = scaler.fit_transform([[0], [1]])[1][0] * similarity
        
        st.markdown(f"### Score: {scaled_score:.2f} / 5.00")
        
        # Add qualitative feedback
        if scaled_score > 4.0:
            st.success("Excellent match with the ideal answer.")
        elif scaled_score > 3.0:
            st.info("Good, but minor gaps detected.")
        elif scaled_score > 2.0:
            st.warning("Partial understanding.")
        else:
            st.error("Poor match. Review required.")

