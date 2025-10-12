import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Make results folder if it doesn't exist
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load preprocessed data
train_df = pd.read_pickle("data/train_preprocessed.pkl")
test_df = pd.read_pickle("data/test_preprocessed.pkl")

X_train = train_df["text"]
y_train = train_df["label"]
X_test = test_df["text"]
y_test = test_df["label"]

# Compare different n-gram ranges
ngram_ranges = [(1, 1), (1, 2)]

for ngram_range in ngram_ranges:
    print(f"\n------------ Testing ngram_range={ngram_range} (BOW) ------------")
    
    pipeline = Pipeline([
        ('bow', CountVectorizer()),  # Bag-of-Words instead of TF-IDF
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'bow__ngram_range': [ngram_range],
        'bow__max_features': [3000, 5000],
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
    
    # Save results
    file_name = f"random_forest_bow_{'unigrams' if ngram_range==(1,1) else 'bigrams'}.txt"
    with open(os.path.join(RESULTS_DIR, file_name), "w") as f:
        f.write(f"Best params: {grid_search.best_params_}\n")
        f.write(f"Best CV accuracy: {grid_search.best_score_:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(class_report)

    print(f"Results saved to {file_name}")
