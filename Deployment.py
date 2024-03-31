import streamlit as st
import numpy as np
from src.utils.utils import load_object, load_GRU

# Load preprocessor and model
preprocess = load_object('artifacts/GRU_Preprocessor.pkl')
loaded_model = load_GRU('artifacts/GRU_Model.h5')

# Function to preprocess input text and predict sentiment
def predict_sentiment(input_text):
    input_padded = preprocess.transform([input_text])
    result = np.argmax(loaded_model.predict(input_padded), axis=1)[0]
    sentiment_labels = {
        0: {'label': 'Terrible Experience', 'emoji': '😡'},
        1: {'label': 'Bad Experience', 'emoji': '😞'},
        2: {'label': 'Decent Experience', 'emoji': '😐'},
        3: {'label': 'Good Experience', 'emoji': '😊'},
        4: {'label': 'Excellent Experience', 'emoji': '😍'}
    }
    sentiment = sentiment_labels[result]
    return sentiment


# Streamlit app
# Streamlit app
def main():
    st.markdown("<h2 style= color: white; white-space: nowrap;'>Hotel Review Sentiment Classification</h2>", unsafe_allow_html=True)

    # User input
    input_text = st.text_area("Enter your hotel review:", "")

    if st.button("Predict Sentiment"):
        if input_text.strip() == "":
            st.warning("Please enter a review.")
        else:
            # Predict sentiment
            sentiment = predict_sentiment(input_text)
            st.markdown(f"<h5 style='color: white;'>{sentiment['label']} {sentiment['emoji']}</h5>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()