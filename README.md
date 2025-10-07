# Opinion Spam Detection – Data Mining 2025

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
Run `baive_bayes.py`, `logistic_regression.py`, `classification_tree.py`, `random_forest.py`, `gradient_boosting.py`:
- Each script:
  - Loads `train_preprocessed.pkl`
  - Performs TF-IDF vectorization (unigrams / bigrams)
  - Runs cross-validation on training folds
  - Evaluates performance
  - Trains on full training set and evaluates on the external test fold
  - Saves final model results
