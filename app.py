"""
app.py
------
Flask backend that:
 1. Loads your trained model (sentiment_model.h5) + tokenizer (tokenizer.pkl)
 2. Serves a simple HTML page (templates/index.html)
 3. Exposes a /predict API that the frontend JS calls
 4. (Optional GenAI add-on) Exposes /explain which uses Groq's free LLM API
    to generate a one-line natural-language explanation of the prediction.
    This is what turns a plain ML/DL project into an "AI + GenAI" project.
    It's optional - app works fine without a GROQ_API_KEY set.
"""

import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dotenv import load_dotenv

load_dotenv() 

# ---------- CONFIG ----------
MAX_LEN = 200
VOCAB_SIZE = 10000
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # set this on Render as an env var (optional)

app = Flask(__name__)

# ---------- LOAD MODEL + TOKENIZER (once, at startup) ----------
print("Loading model...")
model = load_model("sentiment_model.h5")

print("Loading tokenizer/word index...")
with open("tokenizer.pkl", "rb") as f:
    word_index = pickle.load(f)

# IMDB word_index: word -> index, but indices are offset by 3
# (0,1,2 reserved for padding/start/unknown). We rebuild that mapping here.
def text_to_sequence(text):
    words = text.lower().split()
    seq = []
    for w in words:
        idx = word_index.get(w, 2) + 3  # 2 = <UNK>, +3 offset matches imdb.load_data()
        if idx < VOCAB_SIZE:
            seq.append(idx)
        else:
            seq.append(2)
    return seq


def predict_sentiment(text):
    seq = text_to_sequence(text)
    padded = pad_sequences([seq], maxlen=MAX_LEN, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])
    label = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if label == "Positive" else 1 - prob
    return label, confidence


# ---------- ROUTES ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter some text."}), 400

    label, confidence = predict_sentiment(text)
    return jsonify({
        "label": label,
        "confidence": round(confidence * 100, 2)
    })


@app.route("/explain", methods=["POST"])
def explain():
    """Optional GenAI layer: uses Groq (free tier) to explain the result in plain English.
    Skips gracefully if no API key is configured."""
    if not GROQ_API_KEY:
        return jsonify({"explanation": "(GenAI explanation disabled — set GROQ_API_KEY to enable)"})

    data = request.get_json()
    text = data.get("text", "")
    label = data.get("label", "")

    try:
        import requests
        resp = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "You are a concise assistant. In one short sentence, explain why a movie review might be classified as the given sentiment."},
                    {"role": "user", "content": f"Review: \"{text}\"\nPredicted sentiment: {label}\nExplain briefly:"}
                ],
                "max_tokens": 60,
            },
            timeout=15,
        )
        result = resp.json()
        explanation = result["choices"][0]["message"]["content"]
    except Exception as e:
        explanation = f"(explanation unavailable: {e})"

    return jsonify({"explanation": explanation})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
