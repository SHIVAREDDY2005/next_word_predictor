import os
import numpy as np
import tensorflow as tf
import pickle
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ===== CREATE APP FIRST =====
app = FastAPI(title="Next Word Predictor")

# ===== CONFIG =====
API_KEY = os.getenv("API_KEY")
MAX_LEN = 10
GENERATE_WORDS = 10

# ===== LOAD TOKENIZER & MODEL =====
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}
model = tf.keras.models.load_model("lstm_model.keras")

# ===== AUTH CHECK =====
def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

# ===== FRONTEND =====
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head><title>Next Word Predictor</title></head>
    <body style="font-family:Arial; padding:40px;">
        <h2>🔥 Next Word Predictor</h2>
        <textarea id="prompt" rows="4" cols="60"
            placeholder="Type something..."></textarea><br><br>
        <button onclick="generate()">Generate</button>
        <p id="output"></p>

        <script>
        async function generate() {
            const text = document.getElementById("prompt").value;
            const res = await fetch(
                `/generate?prompt=${encodeURIComponent(text)}`,
                {
                    method: "POST",
                    headers: { "x-api-key": "bro-this-is-my-ml-api" }
                }
            );
            const data = await res.json();
            document.getElementById("output").innerText = data.generated_text;
        }
        </script>
    </body>
    </html>
    """

# ===== API ENDPOINT =====
@app.post("/generate")
def generate_text(prompt: str, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    text = prompt
    for _ in range(GENERATE_WORDS):
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre")
        pred = model.predict(padded, verbose=0)
        pos = np.argmax(pred)
        text += " " + index_word.get(pos, "")

    return {
        "prompt": prompt,
        "generated_text": text
    }
