"""
Amazon Fine Food Reviews — Sentiment Classifier
Trains 5 models on the SAME cleaned text input and exports artifacts
in the same style as the reference Naukri project:

    Logistic Regression (TF-IDF)  -> logreg_model.pkl + tfidf_vectorizer.pkl
    SimpleRNN                     -> simplernn_model.h5
    LSTM                          -> lstm_model.h5
    GRU                           -> gru_model.h5
    Transformer                   -> transformer_model.keras
    (shared)                      -> tokenizer.pkl, label_encoder.pkl

Target label
------------
The Reviews.csv file only has a 1-5 star `Score`, not a ready-made class
column, so this script derives a 3-class Sentiment label the same way the
Amazon Fine Food Reviews dataset is conventionally used:

    Score 1-2  -> "Negative"
    Score 3    -> "Neutral"
    Score 4-5  -> "Positive"

If you'd rather predict the raw 1-5 Score as a 5-class problem instead,
set TARGET = "score" below.

Usage
-----
    pip install -r requirements.txt
    python train_and_export_models.py --csv data/Reviews.csv --sample 40000 --epochs 4

Drop --sample to train on the full ~568k rows (needs more time/RAM;
a GPU runtime such as Google Colab is recommended for that).
"""

import argparse
import pickle
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

TARGET = "sentiment"  # "sentiment" (3-class) or "score" (5-class)
MAX_LEN = 40
MAX_VOCAB = 20000
EMBED_DIM = 64


# ---------------------------------------------------------------------------
# Cleaning — kept identical between training and the Streamlit app
# ---------------------------------------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = text.replace("|", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(csv_path, sample):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["Text", "Score"])
    # This dataset is known to contain many duplicate reviews (same user,
    # same text, re-posted under different ProductIds) — drop them.
    df = df.drop_duplicates(subset=["UserId", "ProfileName", "Time", "Text"])

    if TARGET == "sentiment":
        def to_sentiment(score):
            if score <= 2:
                return "Negative"
            elif score == 3:
                return "Neutral"
            return "Positive"

        df["label"] = df["Score"].apply(to_sentiment)
    else:
        df["label"] = df["Score"].astype(str)

    df["Summary"] = df["Summary"].fillna("")
    df["combined_text"] = (df["Summary"] + " " + df["Text"]).apply(clean_text)
    df = df[df["combined_text"].str.len() > 0]

    if sample:
        per_class = sample // df["label"].nunique()
        parts = [
            g.sample(min(len(g), per_class), random_state=42)
            for _, g in df.groupby("label")
        ]
        df = pd.concat(parts, ignore_index=True)
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Custom Transformer layers (same design as the reference project so the
# saved .keras file loads the same way in app.py)
# ---------------------------------------------------------------------------
def build_transformer_layers():
    import tensorflow as tf
    from tensorflow.keras import layers

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

    return TokenAndPositionEmbedding, TransformerBlock


def train_deep_models(X_train_pad, y_train_cat, X_test_pad, y_test_cat, vocab_size, num_classes, epochs):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    results = {}

    def compile_fit(model, name):
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(
            X_train_pad, y_train_cat,
            validation_data=(X_test_pad, y_test_cat),
            epochs=epochs, batch_size=128, verbose=2,
        )
        results[name] = model
        return model

    # SimpleRNN
    rnn = models.Sequential([
        layers.Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        layers.SimpleRNN(64),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    compile_fit(rnn, "simplernn")

    # LSTM
    lstm = models.Sequential([
        layers.Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        layers.LSTM(64),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    compile_fit(lstm, "lstm")

    # GRU
    gru = models.Sequential([
        layers.Embedding(vocab_size, EMBED_DIM, input_length=MAX_LEN),
        layers.GRU(64),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    compile_fit(gru, "gru")

    # Transformer
    TokenAndPositionEmbedding, TransformerBlock = build_transformer_layers()
    inputs = layers.Input(shape=(MAX_LEN,))
    x = TokenAndPositionEmbedding(MAX_LEN, vocab_size, EMBED_DIM)(inputs)
    x = TransformerBlock(EMBED_DIM, num_heads=2, ff_dim=64)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    transformer = models.Model(inputs, outputs)
    compile_fit(transformer, "transformer")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/Reviews.csv")
    parser.add_argument("--sample", type=int, default=40000, help="rows to use (balanced per class); 0 = full dataset")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--skip-deep", action="store_true", help="only train the TF-IDF + Logistic Regression baseline")
    args = parser.parse_args()

    print(f"Loading {args.csv} ...")
    df = load_data(args.csv, args.sample)
    print(f"Rows after cleaning/sampling: {len(df)}")
    print(df["label"].value_counts())

    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["combined_text"], y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Logistic Regression baseline ----
    print("\nTraining TF-IDF + Logistic Regression ...")
    tfidf = TfidfVectorizer(max_features=MAX_VOCAB, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train_tfidf, y_train)
    print(classification_report(y_test, logreg.predict(X_test_tfidf), target_names=le.classes_))

    with open("logreg_model.pkl", "wb") as f:
        pickle.dump(logreg, f)
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print("Saved logreg_model.pkl, tfidf_vectorizer.pkl, label_encoder.pkl")

    if args.skip_deep:
        print("\n--skip-deep set: not training SimpleRNN/LSTM/GRU/Transformer.")
        return

    # ---- Deep learning models ----
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.utils import to_categorical

    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)
    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)

    print(f"\nTraining SimpleRNN / LSTM / GRU / Transformer for {args.epochs} epochs each ...")
    trained = train_deep_models(X_train_pad, y_train_cat, X_test_pad, y_test_cat, vocab_size, num_classes, args.epochs)

    trained["simplernn"].save("simplernn_model.h5")
    trained["lstm"].save("lstm_model.h5")
    trained["gru"].save("gru_model.h5")
    trained["transformer"].save("transformer_model.keras")
    print("Saved simplernn_model.h5, lstm_model.h5, gru_model.h5, transformer_model.keras")

    print("\nAll artifacts saved in the current directory — copy them next to app.py before running Streamlit.")


if __name__ == "__main__":
    main()
