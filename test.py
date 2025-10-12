import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

RESULTS_EXTRA_DIR = "results_extra"

# File pairs for unigrams and bigrams
pred_files = {
    "unigrams": ("y_pred_gb_unigrams.npy", "y_pred_rf_unigrams.npy"),
    "bigrams": ("y_pred_gb_bigrams.npy", "y_pred_rf_bigrams.npy")
}

# Load true labels once
y_true = np.load(f"{RESULTS_EXTRA_DIR}/y_test.pkl", allow_pickle=True)

for ngram_name, (gb_file, rf_file) in pred_files.items():
    print(f"\n=== McNemar Test for {ngram_name} ===")
    
    y_pred_gb = np.load(f"{RESULTS_EXTRA_DIR}/{gb_file}", allow_pickle=True)
    y_pred_rf = np.load(f"{RESULTS_EXTRA_DIR}/{rf_file}", allow_pickle=True)
    
    # Complete contingency table
    a = np.sum((y_pred_gb == y_true) & (y_pred_rf == y_true))  # Both correct
    b = np.sum((y_pred_gb == y_true) & (y_pred_rf != y_true))  # GB correct, RF wrong
    c = np.sum((y_pred_gb != y_true) & (y_pred_rf == y_true))  # GB wrong, RF correct
    d = np.sum((y_pred_gb != y_true) & (y_pred_rf != y_true))  # Both wrong

    # Print table
    print(f"Contingency Table (GB vs RF):")
    print(f"                RF Correct | RF Wrong")
    print(f"GB Correct:      {a:4d}      |   {b:4d}")
    print(f"GB Wrong:        {c:4d}      |   {d:4d}")
    
    # Accuracies
    acc_gb = np.mean(y_pred_gb == y_true)
    acc_rf = np.mean(y_pred_rf == y_true)
    print(f"\nGradient Boosting Accuracy: {acc_gb:.4f}")
    print(f"Random Forest Accuracy:    {acc_rf:.4f}")
    
    # Run McNemar test
    table = [[a, b],
             [c, d]]
    result = mcnemar(table, exact=True)
    print(f"\nMcNemar's test p-value: {result.pvalue:.5f}")
    
    if result.pvalue < 0.05:
        better_model = "Gradient Boosting" if acc_gb > acc_rf else "Random Forest"
        print(f"✓ SIGNIFICANT: {better_model} performs significantly better (p < 0.05)")
    else:
        print("× NOT SIGNIFICANT: Performance difference could be due to chance")
