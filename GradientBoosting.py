import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_pickle("data/train_preprocessed.pkl")
test_df = pd.read_pickle("data/test_preprocessed.pkl")

# Prepare features and labels
X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# Compare different n-gram ranges
ngram_ranges = [(1, 1), (1, 2), (2, 2)]

for ngram_range in ngram_ranges:
    print(f"\n------------ Testing ngram_range={ngram_range} ------------")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=ngram_range)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    
    cv_scores = cross_val_score(pipeline, X_train, y_train, 
                               cv=5, scoring='accuracy', n_jobs=-1)
    
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

    # For final evaluation: train on full training set, test on test set
    print("\n=== Final Training and Test Evaluation ===")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    print(f"Test set accuracy: {test_accuracy:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
