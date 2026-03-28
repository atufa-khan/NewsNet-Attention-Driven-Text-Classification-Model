import streamlit as st
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

st.title("📰 News Classification App")

st.write("Enter a news article and classify it")

text = st.text_area("Enter news text here")

if st.button("Classify"):
    if text:
        embedding = model.encode([text])
        st.success(f"Prediction: {embedding.shape}")
    else:
        st.warning("Please enter text")
