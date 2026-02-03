import os
import numpy as np
import tensorflow as tf
import pickle

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 🔥 MUST BE FIRST
app = FastAPI(title="Next Word Predictor")

API_KEY = os.getenv("API_KEY")
MAX_LEN = 10
GENERATE_WORDS = 10

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}
model = tf.keras.models.load_model("lstm_model.keras")

def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body style="font-family:Arial; padding:40px;">
        <h2>🔥 Next Word Predictor</h2>

        <label>API Key:</label><br>
        <input id="apikey" type="text" style="width:400px;"><br><br>

        <label>Enter Text:</label><br>
        <textarea id="prompt" rows="4" cols="60"></textarea><br><br>

        <button onclick="generate()">Generate</button>
        <p id="output"></p>

        <script>
        async function generate() {
            const text = document.getElementById("prompt").value;
            const apiKey = document.getElementById("apikey").value;

            const res = await fetch(
                `/generate?prompt=${encodeURIComponent(text)}`,
                {
                    method: "POST",
                    headers: { "x-api-key": apiKey }
                }
            );

            if (!res.ok) {
                document.getElementById("output").innerText =
                    "❌ Invalid API Key or error";
                return;
            }

            const data = await res.json();
            document.getElementById("output").innerText = data.generated_text;
        }
        </script>
    </body>
    </html>
    """


@app.post("/generate")
def generate_text(prompt: str, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    text = prompt
    for _ in range(GENERATE_WORDS):
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="pre")
        pred = model.predict(padded, verbose=0)
        text += " " + index_word.get(np.argmax(pred), "")

    return {"generated_text": text}

