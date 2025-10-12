import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
RESULTS_EXTRA_DIR = "results_extra"
CONF_MATRIX_DIR = "confusion_matrices"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_EXTRA_DIR, exist_ok=True)
os.makedirs(CONF_MATRIX_DIR, exist_ok=True)

train_df = pd.read_pickle("data/train_preprocessed.pkl")
test_df = pd.read_pickle("data/test_preprocessed.pkl")

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

ngram_ranges = [(1, 1), (1, 2)]
C_GRID = [0.001, 0.01, 0.1, 1, 10, 100]

for ngram_range in ngram_ranges:
    print(f"\n------------ Testing ngram_range={ngram_range} ------------")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=ngram_range)),
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

    pipeline.fit(X_train, y_train)
    best_C = float(np.ravel(pipeline.named_steps['lr'].C_)[0])
    print(f"Best C found: {best_C}")

    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print(f"Test set accuracy: {test_accuracy:.4f}")
    print("\n=== Classification Report ===")
    print(class_report)

    # Save predictions
    pred_file = f"y_pred_lr_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.npy"
    np.save(os.path.join(RESULTS_EXTRA_DIR, pred_file), y_pred)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Deceptive", "Truthful"])
    cm_file = f"confusion_matrix_lr_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.npy"
    np.save(os.path.join(CONF_MATRIX_DIR, cm_file), cm)

    # Optional: plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Deceptive", "Truthful"])
    disp.plot(cmap="Blues")
    plt.title(f"LR Confusion Matrix ({'unigrams' if ngram_range==(1,1) else 'bigrams'})")
    plt.savefig(os.path.join(CONF_MATRIX_DIR, f"confusion_matrix_lr_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.png"))
    plt.close()

    # Save classification report and accuracy
    file_name = f"logistic_regression_{'unigrams' if ngram_range==(1,1) else 'bigrams'}_tfidf.txt"
    file_path = os.path.join(RESULTS_DIR, file_name)
    with open(file_path, "w") as f:
        f.write(f"Best C: {best_C}\n")
        f.write(f"Test set accuracy: {test_accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    # Extract feature names and importance
    vec = pipeline.named_steps['tfidf']
    lr = pipeline.named_steps['lr']

    feature_names = np.array(vec.get_feature_names_out())

    coefs = lr.coef_.ravel()
    abs_coefs = np.abs(coefs)
    top_idx = np.argsort(abs_coefs)[::-1][:10]
    top_features = feature_names[top_idx]
    top_weights = coefs[top_idx]

    print("Top 10 most important features:")
    for feat, w in zip(top_features, top_weights):
        print(f"{feat}: {w:.4f}")

    # Save to text file
    with open(file_path, "a") as f:
        f.write("\nTop 10 most important features (by |weight|):\n")
        for feat, w in zip(top_features, top_weights):
            f.write(f"{feat}: {w:.4f}\n")

    print(f"Results saved to {file_name}, predictions and confusion matrix saved.")