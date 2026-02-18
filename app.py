import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Book Recommender", layout="wide")

# ---------- Load Model Safely ----------
@st.cache_resource
def load_model():
    df = pickle.load(open("model/products.pkl", "rb"))

    # Safety cleaning
    df = df.dropna(subset=["name", "description"])
    df["category"] = df["category"].fillna("book")

    df["features"] = df["name"] + " " + df["category"] + " " + df["description"]

    tfidf = TfidfVectorizer(stop_words="english")
    vectors = tfidf.fit_transform(df["features"])

    similarity = cosine_similarity(vectors)

    return df, similarity

df, similarity = load_model()

# ---------- UI ----------
st.title("📚 Smart Book Recommender")

search_query = st.text_input("🔍 Search a book")

filtered_books = df[df["name"].str.contains(search_query, case=False, na=False)] if search_query else df

selected_book = st.selectbox("Select a book", filtered_books["name"].values)

# ---------- Recommendation Function ----------
def recommend(book):
    index = df[df["name"] == book].index[0]
    distances = similarity[index]
    books = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names = []
    images = []

    for i in books:
        names.append(df.iloc[i[0]]["name"])
        images.append(df.iloc[i[0]]["image_url"])

    return names, images

# ---------- Button ----------
if st.button("Recommend"):
    names, images = recommend(selected_book)

    st.subheader("✨ You may also like")

    cols = st.columns(5)

    for idx in range(5):
        with cols[idx]:
            st.image(images[idx], width=150)
            st.caption(names[idx])
