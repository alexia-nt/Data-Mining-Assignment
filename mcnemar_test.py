import os
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

# === CONFIG ===
RESULTS_EXTRA_DIR = "results_extra"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load true labels
y_true = np.load(os.path.join(RESULTS_EXTRA_DIR, "y_test.pkl"), allow_pickle=True)

# Define models and feature types
models = ["nb", "lr", "dt", "rf", "gb"]
feature_sets = ["unigrams", "bigrams"]

# Helper to load predictions
def load_pred(model, feature_set):
    fname = f"y_pred_{model}_{feature_set}.npy"
    return np.load(os.path.join(RESULTS_EXTRA_DIR, fname), allow_pickle=True)

# === Prepare all tests ===
tests = []

# 5.1 – NB vs LR
for feature_set in feature_sets:
    tests.append(("5.1", f"NB vs LR ({feature_set})", "nb", "lr", feature_set))

# 5.2 – Tree vs Linear
pairs = [("nb", "rf"), ("nb", "gb"), ("lr", "rf"), ("lr", "gb")]
for feature_set in feature_sets:
    for m1, m2 in pairs:
        tests.append(("5.2", f"{m1.upper()} vs {m2.upper()} ({feature_set})", m1, m2, feature_set))

# 5.3 – Unigrams vs Bigrams (same model)
for model in models:
    tests.append(("5.3", f"{model.upper()} unigrams vs bigrams", model, model, None))

# === Run McNemar tests ===
results = []
print("\n=== Running McNemar Tests ===")

for q, desc, m1, m2, feature_set in tests:
    if feature_set:
        y1 = load_pred(m1, feature_set)
        y2 = load_pred(m2, feature_set)
        name = f"{desc}"
    else:
        y1 = load_pred(m1, "unigrams")
        y2 = load_pred(m2, "bigrams")
        name = f"{desc}"
    
    # Contingency counts
    a = np.sum((y1 == y_true) & (y2 == y_true))  # both correct
    b = np.sum((y1 == y_true) & (y2 != y_true))  # m1 correct, m2 wrong
    c = np.sum((y1 != y_true) & (y2 == y_true))  # m1 wrong, m2 correct
    d = np.sum((y1 != y_true) & (y2 != y_true))  # both wrong

    # Accuracies
    acc1 = np.mean(y1 == y_true)
    acc2 = np.mean(y2 == y_true)

    # McNemar test
    table = [[a, b], [c, d]]
    result = mcnemar(table, exact=True)
    p = result.pvalue

    results.append([q, name, acc1, acc2, a, b, c, d, p])

# === Make dataframe and correct for multiple tests ===
df = pd.DataFrame(results, columns=[
    "Question", "Comparison", "Acc_Model1", "Acc_Model2", "a_both_correct", "b_m1_correct", "c_m2_correct", "d_both_wrong", "p_value"
])

# FDR correction (Benjamini-Hochberg)
df["p_adj"] = multipletests(df["p_value"], alpha=0.05, method='fdr_bh')[1]
df["Significant (FDR<0.05)"] = df["p_adj"] < 0.05
df["Better_Model"] = np.where(df["Acc_Model1"] > df["Acc_Model2"], "Model 1", "Model 2")

# === Save and print ===
out_path = os.path.join(RESULTS_DIR, "mcnemar_summary.txt")
with open(out_path, "w") as f:
    for _, row in df.iterrows():
        f.write(f"\n=== {row['Question']} : {row['Comparison']} ===\n")
        f.write(f"Acc1: {row['Acc_Model1']:.4f} | Acc2: {row['Acc_Model2']:.4f}\n")
        f.write(f"Contingency Table:\n")
        f.write(f"   a (both correct): {row['a_both_correct']}\n")
        f.write(f"   b (Model1 correct only): {row['b_m1_correct']}\n")
        f.write(f"   c (Model2 correct only): {row['c_m2_correct']}\n")
        f.write(f"   d (both wrong): {row['d_both_wrong']}\n")
        f.write(f"p-value: {row['p_value']:.5f} | FDR-adjusted p: {row['p_adj']:.5f}\n")
        if row['Significant (FDR<0.05)']:
            f.write(f"SIGNIFICANT — {row['Better_Model']} performs better\n")
        else:
            f.write("NOT SIGNIFICANT\n")

print("\n✅ All McNemar tests completed.")
print(f"Results saved to: {out_path}")

# Optionally display summary table
print("\nSummary (first few rows):")
print(df.head())
