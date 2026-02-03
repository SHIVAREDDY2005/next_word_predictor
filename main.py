import numpy as np
import tensorflow as tf
import pickle
from fastapi import FastAPI, Header, HTTPException
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
