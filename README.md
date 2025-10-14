# Opinion Spam Detection - Data Mining 2025

This project implements several machine learning models for classifying hotel reviews as **truthful** or **deceptive**, based on [this dataset](https://myleott.com/op-spam.html).

## Requirements
- Python 3.11.5
- Packages listed in requirements.txt

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/opinion-spam-detection.git
   cd opinion-spam-detection
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   - On Windows (PowerShell):
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Workflow

### 1. Load and save raw data (optional)
Run `load_data.py`:
- Reads raw `.txt` files from `op_spam_v1.4/`
- Merges them into train/test DataFrames
- Saves `train.pkl` and `test.pkl` in `data/`

### 2. Preprocess the data (optional)
Run `preprocessing.py`:
- Reads `train.pkl` and `test.pkl`
- Performs:
  - Lowercasing
  - Punctuation removal
  - Tokenization
  - Stopwords removal
  - Lemmatization
- Saves preprocessed data as `train_preprocessed.pkl` and `test_preprocessed.pkl`

### 3. Experimentation and model training
For each model, there are separate scripts depending on the feature representation: **TF-IDF** or **Bag-of-Words (BoW)**.
| Model                   | TF-IDF Script                      | BoW Script                       |
| ----------------------- | ---------------------------------- | -------------------------------- |
| Multinomial Naive Bayes | `multinomial_naive_bayes_tfidf.py` | `multinomial_naive_bayes_bow.py` |
| Logistic Regression     | `logistic_regression_tfidf.py`     | `logistic_regression_bow.py`     |
| Classification Tree     | `classification_tree_tfidf.py`     | `classification_tree_bow.py`     |
| Random Forest           | `random_forest_tfidf.py`           | `random_forest_bow.py`           |
| Gradient Boosting       | `gradient_boosting_tfidf.py`       | `gradient_boosting_bow.py`       |

Each script performs the following steps:
- Loads train_preprocessed.pkl.
- Performs feature extraction (TF-IDF or BoW) with unigrams and bigrams.
- Runs cross-validation on the training folds.
- Evaluates performance on validation data.
- Trains on the full training set and evaluates on the external test set.
- Saves predictions, confusion matrices, and a detailed classification report.
