import streamlit as st
import joblib
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load model and vectorizer
@st.cache_resource

def load_model_and_vectoriser():
    model = joblib.load(os.path.join(script_dir, "../Models/final_model.pkl"))
    vectoriser = joblib.load(os.path.join(script_dir, "../Models/final_vectoriser.pkl"))
    return model, vectoriser

model, vectoriser = load_model_and_vectoriser()

# App title
st.set_page_config(page_title="Fake News Detector")
st.title("Fake News Detection App")
st.markdown("Enter a news article or headline below to check if it's **Real** or **Fake**.")

# Input box
user_input = st.text_area("Paste your news text here", height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        input_vector = vectoriser.transform([user_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0].max()

        # Show result
        if prediction == 1:
            st.success(f"This news is **Real** ({proba*100:.2f}% confidence)")
        else:
            st.error(f"This news is **Fake** ({proba*100:.2f}% confidence)")