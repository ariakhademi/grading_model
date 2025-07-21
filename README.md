# Grading model for the American Board of Anesthesiology

Scenario: You are working for a medical certification board exploring the use of short-answer questions for continuing certification exams. You are asked to build a prototype grading model that can automatically score short, free-text responses (1-3 sentences) given a reference “ideal” answer.
Objective: Please share and explain your approach to reach the goal.

## How to run
To run the Automated Grading Prototype, ensure you have all required Python packages installed (e.g., streamlit, sentence-transformers, scikit-learn, etc.), and then simply execute the included Bash script. To do this, open a terminal, navigate to the project directory, and run ./prototype_runner.sh. This script launches the prototype_app.py Streamlit application, which provides a web interface for grading short free-text answers using sentence embeddings and similarity metrics. Make sure the script has execute permissions (chmod +x run_grader.sh) before running it.

## Approach
In our approach, we use sentence embeddings to convert both the ideal answer and the candidate's response into fixed-length numerical vectors using a pre-trained transformer model. These embeddings capture the semantic meaning of the sentences beyond just surface-level word matching. Once we have these vector representations, we compute similarity scores between them using metrics such as cosine similarity, Euclidean distance, or Manhattan distance. Cosine similarity measures the angle between the two vectors, highlighting directional similarity, while the distance-based metrics quantify how far apart the vectors are in space. These similarity scores are then scaled and converted into a 0–5 grading score, offering an interpretable and automated way to evaluate the semantic closeness between a candidate's response and the expected answer.

## Transformer models
This project leverages transformer-based sentence embedding models to evaluate the semantic similarity between a candidate response and an ideal medical answer. It supports general-purpose models such as all-MiniLM-L6-v2, all-MiniLM-L12-v2, and paraphrase-MiniLM-L6-v2, all of which are lightweight yet powerful variants of Sentence-BERT fine-tuned for tasks like paraphrase detection and semantic textual similarity. Additionally, it includes two domain-specific models— BioBERT (pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb) and PubMedBERT (pritamdeka/S-PubMedBERT-MS-MARCO)—which are pre-trained and fine-tuned on biomedical and clinical corpora, making them well-suited for medical language understanding. All models generate fixed-length sentence embeddings using mean pooling, and similarity scores between candidate and reference answers are calculated using cosine similarity, Euclidean distance, or Manhattan distance for robust evaluation.

## File structure
<pre><code>
text grading_model/ 
├── README.md # Project overview and usage instructions 
├── prototype_app.py # Streamlit app for grading prototype 
├── prototype_runner.sh # Shell script to run the prototype 
├── sample_QA.txt # Sample question-answer pairs for testing 

</code></pre>

## List of demo functionalities
- Compares a candidate answer with a given ideal answer
- Provides a normalized score, in [0,1], for measuring similarity of the answers
- Outputs words in candidate answers that are missing, according to comparison with the ideal answer 
- Gives feedback through a visualized bar
- Categorizes quality of candidate answers into three groups of excellent, fair, and poor.

## Limitations and future work
Not clinically fine-tuned.
Incorporate human scoring as feedback loop.

## Citations:

### Models
BioBERT: https://huggingface.co/pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb<br>
PubMedBERT: https://huggingface.co/pritamdeka/S-PubMedBert-MS-MARCO<br>
MiniLM-L6: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2<br>
MiniLM-L12: https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2<br>
paraphrase-MiniLM-L6: https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L6-v2
