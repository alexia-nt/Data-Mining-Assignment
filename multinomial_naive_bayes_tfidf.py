import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Make results folder if it doesn't exist
RESULTS_DIR = "results"
RESULTS_EXTRA_DIR = "results_extra"
CONF_MATRIX_DIR = "confusion_matrices"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(RESULTS_EXTRA_DIR, exist_ok=True)
os.makedirs(CONF_MATRIX_DIR, exist_ok=True)

# Load preprocessed data
train_df = pd.read_pickle("data/train_preprocessed.pkl")
test_df = pd.read_pickle("data/test_preprocessed.pkl")

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# N-gram ranges to test
ngram_ranges = [(1, 1), (1, 2)]

for ngram_range in ngram_ranges:
    print(f"\n------------ Testing ngram_range={ngram_range} ------------")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('nb', MultinomialNB())
    ])

    param_grid = {
        'tfidf__ngram_range': [ngram_range],
        'tfidf__max_features': [3000, 5000],
        'nb__alpha': [0.075, 0.1,]  
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    print("Best params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)
    print(f"Test set accuracy: {test_accuracy:.4f}")
    print("\n=== Classification Report ===")
    print(class_report)

    # Save predictions
    pred_file = f"y_pred_nb_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.npy"
    np.save(os.path.join(RESULTS_EXTRA_DIR, pred_file), y_pred)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["Deceptive", "Truthful"])
    cm_file = f"confusion_matrix_nb_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.npy"
    np.save(os.path.join(CONF_MATRIX_DIR, cm_file), cm)

    # Optional: save confusion matrix plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Deceptive", "Truthful"])
    disp.plot(cmap="Blues")
    plt.title(f"GB Confusion Matrix ({'unigrams' if ngram_range==(1,1) else 'bigrams'})")
    plt.savefig(os.path.join(CONF_MATRIX_DIR, f"confusion_matrix_nb_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.png"))
    plt.close()

    # Save results to file
    file_name = f"multinomialnb_{'unigrams' if ngram_range==(1,1) else 'bigrams'}_tfidf.txt"
    file_path = os.path.join(RESULTS_DIR, file_name)
    with open(file_path, "w") as f:
        f.write(f"Best params: {grid_search.best_params_}\n")
        f.write(f"Best CV accuracy: {grid_search.best_score_:.4f}\n")
        f.write(f"Test set accuracy: {test_accuracy:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"Results saved to {file_path}")