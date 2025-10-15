import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load preprocessed data
train_df = pd.read_pickle("data/train_preprocessed.pkl")
X_train = train_df["text"]

# Fit a TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)

# Vocabulary size
vocab_size = len(vectorizer.vocabulary_)
print(f"Vocabulary size: {vocab_size}")

# train_df = pd.read_pickle("data/train_preprocessed.pkl")
# test_df = pd.read_pickle("data/test_preprocessed.pkl")

# print(train_df.head())

results_df = pd.read_pickle("results_extra/y_test.pkl")
print(results_df)

data = np.load("results_extra/y_pred_dt_bigrams.npy", allow_pickle=True)
print(data)