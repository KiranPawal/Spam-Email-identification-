from flask import Flask, render_template, request
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

model = pickle.load(open("spam_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def clean_message(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text)

    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]

    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    message = ""

    if request.method == "POST":
        message = request.form.get("email")

        if not message or message.strip() == "":
            result = "⚠️ Please enter an email message."
        else:
            cleaned_message = clean_message(message)
            vectorized_input = vectorizer.transform([cleaned_message])
            prediction = model.predict(vectorized_input)[0]

            result = "🚫 Spam ❌" if prediction == 1 else "✅ Not Spam"

    return render_template("index.html", result=result, message=message)


if __name__ == "__main__":
    app.run(debug=True)
