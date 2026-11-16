# Amazon Fine Food Reviews — Sentiment Classifier (RNN / LSTM / GRU / Transformer)

Text classification project on the **Amazon Fine Food Reviews** dataset (568k reviews, `Reviews.csv`). It predicts a review's **Sentiment** (Negative / Neutral / Positive) from its summary + review text, using five models trained on the same cleaned input:

| Model | File |
|---|---|
| Logistic Regression (TF-IDF) | `logreg_model.pkl` + `tfidf_vectorizer.pkl` |
| SimpleRNN | `simplernn_model.h5` |
| LSTM | `lstm_model.h5` |
| GRU | `gru_model.h5` |
| Transformer (custom token+position embedding, attention block) | `transformer_model.keras` |

Shared artifacts: `tokenizer.pkl` (Keras tokenizer for the DL models),
`label_encoder.pkl` (maps class index ↔ Negative/Neutral/Positive).

## Dataset

- Source: [Amazon Fine Food Reviews](https://snap.stanford.edu/data/web-FineFoods.html) — `Reviews.csv`, ~568,454 rows.
- Label derivation: `Score` 1–2 → **Negative**, 3 → **Neutral**, 4–5 → **Positive**.
  (You can switch to a raw 5-class Score prediction instead by setting
  `TARGET = "score"` at the top of `train_and_export_models.py`.)
- Input text: `Summary` + `Text`, lower-cased, HTML tags and punctuation stripped
  (same `clean_text()` used in training and in the app, so they never drift apart).
- Exact duplicate reviews (same user/profile/time/text, re-posted under a
  different product) are dropped before splitting.

## Project structure

```
.
├── app.py                        # Streamlit demo — pick a model, type a review, get a prediction
├── train_and_export_models.py    # Full training pipeline for all 5 models
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md                 # put Reviews.csv here (not committed — see below)
│   └── Reviews.csv               # (you provide this; git-ignored)
└── README.md
```

After training, the five model files + `tokenizer.pkl` + `label_encoder.pkl`
+ `tfidf_vectorizer.pkl` land in the project root, next to `app.py` —
exactly where `app.py` expects to find them.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train the models

Put `Reviews.csv` in `data/`, then:

```bash
# Quick run on a balanced 30k-row sample (a few minutes on CPU)
python train_and_export_models.py --csv data/Reviews.csv --sample 30000 --epochs 4

# Full dataset (568k rows) — slow on CPU; a GPU runtime (e.g. Google Colab) is recommended
python train_and_export_models.py --csv data/Reviews.csv --sample 0 --epochs 5
```

This prints a classification report for each model and saves every
artifact listed above into the project folder.

If you only want the fast baseline (no TensorFlow needed) while you set
up the deep-learning environment, add `--skip-deep`:

```bash
python train_and_export_models.py --csv data/Reviews.csv --sample 30000 --skip-deep
```

## 2. Run the app

```bash
streamlit run app.py
```

Pick a model from the dropdown, type a review summary + text, and click
**Predict Sentiment** to see the predicted class and the probability for
every class.

## Uploading to GitHub

`Reviews.csv` (~300MB) and `database.sqlite` (~370MB) are git-ignored —
GitHub rejects pushes with files over 100MB without Git LFS. Only the
code, `requirements.txt`, `.gitignore`, and (once trained) the model
artifact files need to be committed:

```bash
git init
git add .
git commit -m "Amazon Fine Food Reviews sentiment classifier (RNN/LSTM/GRU/Transformer)"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

`lstm_model.h5`, `gru_model.h5`, `simplernn_model.h5` and
`transformer_model.keras` are typically a few MB each at this model size
and push fine; if a full-dataset run produces a much bigger embedding and
you hit the 100MB limit, use Git LFS for that one file.
