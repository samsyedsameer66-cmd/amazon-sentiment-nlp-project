import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers

st.set_page_config(page_title="Amazon Fine Food Reviews — Sentiment Classifier", page_icon="🍜", layout="centered")

MAX_LEN = 40


# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("|", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---- Custom layers required to load the Transformer model ----
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.maxlen = maxlen
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=self.maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": self.token_emb.input_dim,
            "embed_dim": self.token_emb.output_dim,
        })
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


CUSTOM_OBJECTS = {
    "TokenAndPositionEmbedding": TokenAndPositionEmbedding,
    "TransformerBlock": TransformerBlock,
}

MODEL_FILES = {
    "Logistic Regression (TF-IDF)": "logreg_model.pkl",
    "SimpleRNN": "simplernn_model.h5",
    "LSTM": "lstm_model.h5",
    "GRU": "gru_model.h5",
    "Transformer": "transformer_model.keras",
}


@st.cache_resource
def load_shared_artifacts():
    with open("label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return le, tokenizer


@st.cache_resource
def load_selected_model(model_name):
    path = MODEL_FILES[model_name]
    if model_name.startswith("Logistic"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        with open("tfidf_vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        return {"type": "sklearn", "model": model, "vectorizer": vectorizer}
    elif model_name == "Transformer":
        model = load_model(path, custom_objects=CUSTOM_OBJECTS)
        return {"type": "keras", "model": model}
    else:
        model = load_model(path)
        return {"type": "keras", "model": model}


le, tokenizer = load_shared_artifacts()


def predict(model_name, text):
    cleaned = clean_text(text)
    bundle = load_selected_model(model_name)
    if bundle["type"] == "sklearn":
        X = bundle["vectorizer"].transform([cleaned])
        probs = bundle["model"].predict_proba(X)[0]
    else:
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs = bundle["model"].predict(padded, verbose=0)[0]

    order = np.argsort(probs)[::-1]
    top_labels = le.inverse_transform(order)
    top_probs = probs[order]
    return top_labels, top_probs


# ---------------- Streamlit UI ----------------
st.title("🍜 Amazon Fine Food Reviews — Sentiment Classifier")
st.write(
    "Predicts review sentiment (Negative / Neutral / Positive) from the review's "
    "summary + text. Pick a model to compare how each one performs on the same input."
)

model_name = st.selectbox("Model", list(MODEL_FILES.keys()), index=0)
summary = st.text_input("Review summary", placeholder="e.g. Great taste but pricey")
review_text = st.text_area("Review text", height=150, placeholder="e.g. Bought this for my dog and she loves it, will order again...")

if st.button("Predict Sentiment"):
    combined = f"{summary} {review_text}".strip()
    if combined == "":
        st.warning("Please enter a review summary and/or review text first.")
    else:
        with st.spinner(f"Running {model_name}..."):
            top_labels, top_probs = predict(model_name, combined)

        st.success(f"**Predicted: {top_labels[0]}** (confidence: {top_probs[0]:.2%})")
        st.caption("All class probabilities")
        df_top = pd.DataFrame({"Sentiment": top_labels, "Probability": top_probs})
        st.bar_chart(df_top.set_index("Sentiment"))
        st.dataframe(df_top.style.format({"Probability": "{:.2%}"}), hide_index=True, use_container_width=True)
