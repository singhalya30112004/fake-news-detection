import streamlit as st
import joblib
import os
import pandas as pd

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../Models/final_model.pkl")
vectoriser_path = os.path.join(script_dir, "../Models/final_vectoriser.pkl")
FEEDBACK_FILE = os.path.join(script_dir, "../Dataset/Feedback_Data.csv")
os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)

# Load model and vectorizer
@st.cache_resource
def load_model_and_vectoriser():
    model = joblib.load(model_path)
    vectoriser = joblib.load(vectoriser_path)
    return model, vectoriser

model, vectoriser = load_model_and_vectoriser()

# Title
st.set_page_config(page_title="Fake News Detector")
st.title("Fake News Detection App")
st.markdown("Enter a news article or headline to check if it's **Real** or **Fake**.")

# Initialize session state
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
    st.session_state.proba = None
    st.session_state.user_input = ""

# User input
user_input = st.text_area("Paste your news text here", value=st.session_state.user_input, height=200)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vector = vectoriser.transform([user_input])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector)[0].max()

        st.session_state.prediction = prediction
        st.session_state.proba = proba
        st.session_state.user_input = user_input

# Show result
if st.session_state.prediction is not None:
    if st.session_state.prediction == 1:
        st.success(f"This news is **Real** ({st.session_state.proba * 100:.2f}% confidence)")
    else:
        st.error(f"This news is **Fake** ({st.session_state.proba * 100:.2f}% confidence)")

    # Feedback
    st.markdown("---")
    st.subheader("Was this prediction correct?")
    feedback = st.radio("Your answer:", ["Yes", "No"])

    if feedback == "Yes":
        if st.button("Submit Feedback"):
            new_data = pd.DataFrame([[st.session_state.user_input, st.session_state.prediction]], columns=["text", "label"])
            if os.path.exists(FEEDBACK_FILE):
                new_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
            else:
                new_data.to_csv(FEEDBACK_FILE, index=False)
            st.success("Feedback recorded. Thank you!")

    elif feedback == "No":
        actual = st.radio("What should it be?", ["Fake", "Real"])
        if st.button("Submit Correct Label"):
            correct_label = 0 if actual == "Fake" else 1
            new_data = pd.DataFrame([[st.session_state.user_input, correct_label]], columns=["text", "label"])
            if os.path.exists(FEEDBACK_FILE):
                new_data.to_csv(FEEDBACK_FILE, mode='a', header=False, index=False)
            else:
                new_data.to_csv(FEEDBACK_FILE, index=False)
            st.success("Correction recorded. Thank you!")