import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

# Make results folder if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

train_df = pd.read_pickle("data/train_preprocessed.pkl")
test_df = pd.read_pickle("data/test_preprocessed.pkl")

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# Compare different n-gram ranges
ngram_ranges = [(1, 1), (1, 2)]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

for ngram_range in ngram_ranges:
    print(f"\n------------ Testing ngram_range={ngram_range} (BoW) ------------")

    pipeline = Pipeline([
        ('bow', CountVectorizer()),
        ('lr', LogisticRegressionCV(
            Cs=C_GRID,
            penalty='l1',
            solver='liblinear',
            cv=5,
            scoring='accuracy',
            max_iter=1000,
            n_jobs=-1,
            refit=True,
            random_state=42
        ))
    ])

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"Mean CV accuracy: {cv_scores.mean():.4f}")

    pipeline.fit(X_train, y_train)
    best_C = float(np.ravel(pipeline.named_steps['lr'].C_)[0])
    print(f"Best C found: {best_C}")

    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Test accuracy: {test_acc:.4f}")
    print("\n=== Classification Report ===")
    print(class_report)

    # Save results
    file_name = f"logistic_regression_{'unigrams' if ngram_range == (1, 1) else 'bigrams'}_bow.txt"
    file_path = os.path.join(RESULTS_DIR, file_name)
    with open(file_path, "w") as f:
        f.write(f"Best C: {best_C}\n")
        f.write(f"Mean CV accuracy: {cv_scores.mean():.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"Results saved to {file_name}")
