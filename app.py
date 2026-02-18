import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Book Recommender", layout="wide")

@st.cache_resource
def load_model():
    df = pickle.load(open("model/products.pkl", "rb"))

    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(
        df["name"] + " " + df["category"] + " " + df["description"]
    )

    similarity = cosine_similarity(vectors)

    return df, similarity

df, similarity = load_model()
