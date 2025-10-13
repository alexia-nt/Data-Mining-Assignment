import pandas as pd
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