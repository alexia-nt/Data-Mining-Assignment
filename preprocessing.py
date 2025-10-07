import pandas as pd
import string
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os

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
    text = text.translate(translator) #removing punctuation
    tokens = word_tokenize(text) #tokenize
    tokens =[w for w in tokens if w not in stop_words] #remove stopwords (unimportant words)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]  #makes dogs into dog and running into ran etc
    return " ".join(tokens)

train_df = pd.read_pickle("data/train.pkl")
test_df = pd.read_pickle("data/test.pkl")


train_df["text"] = train_df["text"].astype(str).apply(processeddata)
test_df["text"]  = test_df["text"].astype(str).apply(processeddata)

# Save preprocessed data
train_df.to_pickle("data/train_preprocessed.pkl")
test_df.to_pickle("data/test_preprocessed.pkl")

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(train_df["text"])
X_test = vectorizer.transform(test_df["text"])
tfidf_df = pd.DataFrame(X_train.toarray(), 
                        columns=vectorizer.get_feature_names_out())
print("TF-IDF train shape:", X_train.shape)


# Picking one external test fold d
if "fold" in test_df.columns and len(test_df["fold"].unique()) >= 1:
    _test_fold_vals = sorted(test_df["fold"].unique().tolist())
    TEST_FOLD = _test_fold_vals[0]

print(f"Using the  fold={TEST_FOLD} as the external test fold.") 


#Cross-validation  (cv) over the training folds (no model), For each fold f (excluding TEST_FOLD), fit TF-IDF on the
#remaining training folds and transform the validation fold.

all_folds = sorted(train_df["fold"].unique())
cv_folds  = [f for f in all_folds if f != TEST_FOLD]

for f in cv_folds:
    print(f"\n[CV] Fold {f} as validation (external test fold = {TEST_FOLD})")
    train_idx = (train_df["fold"] != f) & (train_df["fold"] != TEST_FOLD)
    val_idx   = (train_df["fold"] == f)

    # Fit vectorizer only on the training split of this fold
    vectorizer = TfidfVectorizer() 
    X_train_cv = vectorizer.fit_transform(train_df.loc[train_idx, "text"])
    X_val_cv   = vectorizer.transform(train_df.loc[val_idx, "text"])

    # Showing us the shape so i know whether i worked
    print("  TF-IDF train shape:", X_train_cv.shape, " | TF-IDF val shape:", X_val_cv.shape)

# Here we combine as a last training fold all the training folds together to train.
vectorizer = TfidfVectorizer()
X_all_train = vectorizer.fit_transform(train_df.loc[train_df["fold"] != TEST_FOLD, "text"])
X_test_hold = vectorizer.transform(test_df["text"])
print(f"\n[HOLD-OUT TEST] all-train shape: {X_all_train.shape} | test shape: {X_test_hold.shape}")


