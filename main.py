import numpy as np
import tensorflow as tf
import pickle
from fastapi import FastAPI, Header, HTTPException
from tensorflow.keras.preprocessing.sequence import pad_sequences

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>LSTM Text Generator</title>
        <style>
            body { font-family: Arial; background:#0f172a; color:white; padding:40px; }
            textarea { width:100%; height:100px; font-size:16px; }
            button { padding:10px 20px; margin-top:10px; font-size:16px; }
            .box { max-width:700px; margin:auto; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🔥 LSTM Text Generator</h1>
            <textarea id="prompt" placeholder="Type something..."></textarea>
            <br>
            <button onclick="generate()">Generate</button>
            <h3>Output:</h3>
            <p id="output"></p>
        </div>

        <script>
        async function generate() {
            const text = document.getElementById("prompt").value;
            const res = await fetch(
                `/generate?prompt=${encodeURIComponent(text)}`,
                {
                    method: "POST",
                    headers: {
                        "x-api-key": "bro-this-is-my-ml-api"
                    }
                }
            );
            const data = await res.json();
            document.getElementById("output").innerText = data.generated_text;
        }
        </script>
    </body>
    </html>
    """

# ====== CONFIG ======
import os
API_KEY = os.getenv("API_KEY")
MAX_LEN = 10
GENERATE_WORDS = 10

# ====== LOAD MODEL & TOKENIZER ======
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}
model = tf.keras.models.load_model("lstm_model.keras")

# ====== APP ======
app = FastAPI(title="LSTM Text Generator API")

# ====== AUTH ======
def check_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ====== ENDPOINT ======
@app.post("/generate")
def generate_text(prompt: str, x_api_key: str = Header(...)):
    check_api_key(x_api_key)

    text = prompt

    for _ in range(GENERATE_WORDS):
        token_text = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(token_text, maxlen=MAX_LEN, padding="pre")

        pred = model.predict(padded, verbose=0)
        pos = np.argmax(pred)

        next_word = index_word.get(pos, "")
        text += " " + next_word

    return {
        "prompt": prompt,
        "generated_text": text
    }

