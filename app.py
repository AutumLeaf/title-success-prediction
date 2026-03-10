import streamlit as st
import joblib
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already present
nltk.download('stopwords')
stop = set(stopwords.words('english'))

# --- 1. LOAD MODELS ---
@st.cache_resource
def load_models():
    retweet_model = joblib.load('retweet_model.pkl')
    likes_model = joblib.load('likes_model.pkl')
    claps_model = joblib.load('claps_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return retweet_model, likes_model, claps_model, vectorizer

retweet_model, likes_model, claps_model, vectorizer = load_models()

# --- 2. PREPROCESSING FUNCTIONS ---
def clean_title(text):
    # Remove twitter users
    text = re.sub('@[_A-Za-z0-9]+', '@', str(text))
    # Basic cleaning as per notebook
    text = text.lower()
    text = re.sub(r'[()!@#$:"?,./+]', '', text)
    return text

def get_features(text):
    # Transform text using the saved vectorizer
    tfidf_features = vectorizer.transform([text]).toarray()
    
    # Calculate additional features used in notebook
    text_len = len(text)
    word_count = len(text.split())
    
    # Combine features: TF-IDF + Length + Word Count
    features = np.append(tfidf_features, [[text_len]], axis=1)
    features = np.append(features, [[word_count]], axis=1)
    
    return features

# --- 3. STREAMLIT UI ---
st.title("🚀 Article Performance Predictor")
st.markdown("""
Predict how many **Retweets**, **Likes**, and **Claps** your technical article title might get on Twitter and Medium.
""")

user_title = st.text_input("Enter your article title:", "How to build a React app in 10 minutes")

if st.button("Predict Performance"):
    if user_title:
        # Preprocess
        cleaned = clean_title(user_title)
        final_features = get_features(cleaned)
        
        # Predict
        pred_rt = retweet_model.predict(final_features)[0]
        pred_likes = likes_model.predict(final_features)[0]
        pred_claps = claps_model.predict(final_features)[0]
        
        # Display Results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Retweets (Twitter)", pred_rt)
        with col2:
            st.metric("Likes (Twitter)", pred_likes)
        with col3:
            st.metric("Claps (Medium)", pred_claps)
            
        st.success("Prediction complete!")
    else:
        st.warning("Please enter a title.")
