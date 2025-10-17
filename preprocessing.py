import pandas as pd
import string
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

RESULTS_EXTRA_DIR = "results_extra"

for pkg in ("stopwords", "wordnet", "punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

translator = str.maketrans ("", "", string.punctuation)

lemmatizer = WordNetLemmatizer()
train_df = pd.read_pickle("data/train.pkl")
test_df = pd.read_pickle("data/test.pkl")


def processeddata(text):
    text = text.lower() # lowercasing
    text = str(text).strip() # removing white spaces
    text = text.translate(translator) # removing punctuation
    tokens = word_tokenize(text) # tokenize
    tokens =[w for w in tokens if w not in stop_words] # remove stopwords (unimportant words)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  # makes dogs into dog and running into ran etc
    return " ".join(tokens)

train_df = pd.read_pickle("data/train.pkl")
test_df = pd.read_pickle("data/test.pkl")


train_df["text"] = train_df["text"].astype(str).apply(processeddata)
test_df["text"]  = test_df["text"].astype(str).apply(processeddata)

# Save preprocessed data
train_df.to_pickle("data/train_preprocessed.pkl")
test_df.to_pickle("data/test_preprocessed.pkl")

# Save true labels (once)
y_test = test_df["label"]
y_test.to_pickle(os.path.join(RESULTS_EXTRA_DIR, "y_test.pkl"))