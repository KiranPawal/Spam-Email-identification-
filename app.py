import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import streamlit.components.v1 as components

components.html("""
<!-- Google tag (gtag.js) -->
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-7NPBHPXX0Y"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-7NPBHPXX0Y');
</script>    
""", height=0)

# Download NLTK data (only first time)
nltk.download("stopwords")

# Load model and vectorizer
model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# NLP tools
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Text cleaning function
def clean_message(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

# ================= UI =================

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="📧",
    layout="centered"
)

st.title("📧 Spam Email Detection")
st.write("Check whether an email message is **Spam** or **Not Spam**.")

# Text input
message = st.text_area(
    "Enter email text:",
    height=150,
    placeholder="Type or paste the email message here..."
)

# Predict button
if st.button("Check Email"):
    if message.strip() == "":
        st.warning("⚠️ Please enter an email message.")
    else:
        cleaned_message = clean_message(message)
        vectorized_input = vectorizer.transform([cleaned_message])
        prediction = model.predict(vectorized_input)[0]

        if prediction == 1:
            st.error("🚫 Spam ❌")
        else:
            st.success("✅ Not Spam")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:13px;'>Designed by <b style='color:#4CAF50;'>Kiran Pawal</b></p>",
    unsafe_allow_html=True
)
