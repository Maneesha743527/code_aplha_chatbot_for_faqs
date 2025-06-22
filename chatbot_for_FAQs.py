import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("🤖 FAQ Chatbot")

# Sample FAQ data
faq_data = {
    "What is your return policy?": "You can return any product within 30 days of delivery.",
    "How do I track my order?": "You can track your order using the tracking ID sent via email.",
    "What payment methods are accepted?": "We accept Credit/Debit Cards, UPI, and Net Banking.",
    "How to contact customer support?": "You can email us at support@example.com or call 1800-000-000.",
    "Do you ship internationally?": "Yes, we ship to over 100 countries worldwide."
}

questions = list(faq_data.keys())
answers = list(faq_data.values())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

user_query = st.text_input("Ask your question:")

if user_query:
    user_vec = vectorizer.transform([user_query])
    similarities = cosine_similarity(user_vec, X)
    index = np.argmax(similarities)
    st.markdown(f"**Answer:** {answers[index]}")
