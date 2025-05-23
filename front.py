import streamlit as st
import requests

st.title("Sentiment & Platform Predictor")

text = st.text_area("Enter text:")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text to predict.")
    else:
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json={"text": text})
            if response.status_code == 200:
                result = response.json()
                sentiment = result.get('sentiment', 'Unknown')
                platform = result.get('platform', 'Unknown')
                st.success(f"Sentiment: {sentiment}")
                st.info(f"Platform: {platform}")
            else:
                st.error("Failed to get prediction from server.")
        except Exception as e:
            st.error(f"Error connecting to prediction server: {e}")
