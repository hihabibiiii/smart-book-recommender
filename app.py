import streamlit as st
import pickle
import pandas as pd

# Load data
df = pickle.load(open("model/products.pkl", "rb"))
similarity = pickle.load(open("model/similarity.pkl", "rb"))

st.set_page_config(page_title="Smart Book Recommender", layout="wide")

# ---------- Title ----------
st.markdown(
    "<h1 style='text-align: center;'>📚 Smart Book Recommender</h1>",
    unsafe_allow_html=True
)

# ---------- Search Box ----------
search_query = st.text_input("🔍 Search a book")

filtered_books = df[df["name"].str.contains(search_query, case=False, na=False)] if search_query else df

selected_book = st.selectbox(
    "Select a book",
    filtered_books["name"].values
)


# ---------- Recommendation Function ----------
def recommend(book):
    index = df[df["name"] == book].index[0]
    distances = similarity[index]
    books = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    names = []
    images = []

    for i in books:
        names.append(df.iloc[i[0]]["name"])        # ✅ FIXED
        images.append(df.iloc[i[0]]["image_url"])  # ✅ FIXED

    return names, images

# ---------- Button ----------
if st.button("Recommend"):
    names, images = recommend(selected_book)

    st.subheader("✨ You may also like")

    cols = st.columns(5)

    for idx in range(5):
        with cols[idx]:
            st.image(images[idx], width=150)  # ✅ fixed deprecation
            st.markdown(
                f"<p style='text-align: center; font-size:14px;'>{names[idx]}</p>",
                unsafe_allow_html=True
            )
