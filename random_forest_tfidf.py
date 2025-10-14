from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import os
import pandas as pd
import numpy as np

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

for ngram_range in ngram_ranges:
    print(f"\n------------ Testing ngram_range={ngram_range} ------------")
    
    pipeline = Pipeline([
        ('vec', TfidfVectorizer()),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'vec__ngram_range': [ngram_range],
        'vec__max_features': [3000, 5000],
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [None, 10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print("Best params:", grid_search.best_params_)
    print("Best CV accuracy:", grid_search.best_score_)
    print(f"Test accuracy: {test_acc:.4f}")
    print(class_report)
    
    # Save predictions
    pred_file = f"y_pred_rf_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.npy"
    np.save(os.path.join(RESULTS_EXTRA_DIR, pred_file), y_pred)

    # Save confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_test, y_pred, labels=["Deceptive", "Truthful"])
    cm_file = f"confusion_matrix_rf_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.npy"
    np.save(os.path.join(CONF_MATRIX_DIR, cm_file), cm)
    
    # Optional: plot
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Deceptive", "Truthful"])
    disp.plot(cmap="Blues")
    plt.title(f"RF Confusion Matrix ({'unigrams' if ngram_range==(1,1) else 'bigrams'})")
    plt.savefig(os.path.join(CONF_MATRIX_DIR, f"confusion_matrix_rf_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.png"))
    plt.close()
    
    # Save classification report and accuracy
    file_name = f"random_forest_{'unigrams' if ngram_range==(1,1) else 'bigrams'}_tfidf.txt"
    file_path = os.path.join(RESULTS_DIR, file_name)
    with open(file_path, "w") as f:
        f.write(f"Best params: {grid_search.best_params_}\n")
        f.write(f"Best CV accuracy: {grid_search.best_score_:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"Results saved to {file_name}, predictions and confusion matrix saved.")

    # Extract feature names and importance
    vec = best_pipeline.named_steps['vec']
    clf = best_pipeline.named_steps['rf']

    feature_names = vec.get_feature_names_out()
    importances = clf.feature_importances_

    # Sort features by importance (descending)
    top_idx = np.argsort(importances)[::-1]
    top_features = feature_names[top_idx][:10]  # top 10 features overall

    print("Top 10 most important features:", top_features)

    # Save to the same txt file
    with open(file_path, "a") as f:
        f.write("\nTop 10 most important features:\n")
        f.write(", ".join(top_features) + "\n")