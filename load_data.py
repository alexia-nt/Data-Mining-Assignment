import os
import glob
import re
import pandas as pd

DATA_DIR = "op_spam_v1.4"
OUTPUT_DIR = "data"


def read_opspam(data_dir: str, subset: str = "negative") -> pd.DataFrame:
    """
    Reads the Opinion Spam v1.4 dataset into a DataFrame.

    Parameters
    ----------
    data_dir : str
        Path to the op_spam_v1.4 directory.
    subset : str
        Which polarity subset to load ("negative" or "positive").

    Returns
    -------
    pd.DataFrame
        Columns: text, label (t/d), polarity (neg/pos), fold
    """
    subset_dir = os.path.join(
        data_dir,
        f"{subset}_polarity"
    )

    rows = []

    for label_name, label in [
        ("deceptive_from_MTurk", "Deceptive"),
        ("truthful_from_Web" if subset == "negative" else "truthful_from_TripAdvisor", "Truthful"),
    ]:
        label_dir = os.path.join(subset_dir, label_name)

        for fold in os.listdir(label_dir):
            fold_dir = os.path.join(label_dir, fold) # e.g., fold1, fold2, ...

            for filepath in glob.glob(os.path.join(fold_dir, "*.txt")):
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read().strip()

                rows.append({
                    "text": text,
                    "label": label,     # 'Truthful' or 'Deceptive'
                    "polarity": subset, # 'negative' or 'positive'
                    "fold": fold
                })

    return pd.DataFrame(rows)


def make_train_test(df: pd.DataFrame, test_fold: str = "fold5"):
    """
    Splits dataset into train and test sets, using fold5 as test.

    Returns
    -------
    (train_df, test_df)
    """
    train = df[df["fold"] != test_fold].reset_index(drop=True)
    test = df[df["fold"] == test_fold].reset_index(drop=True)

    return train, test


if __name__ == "__main__":
    DATA_DIR = "op_spam_v1.4"

    # Load only negative reviews
    df = read_opspam(DATA_DIR, subset="negative")
    train_df, test_df = make_train_test(df, test_fold="fold5")

    # Create the folder if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save as pickles
    train_df.to_pickle(os.path.join(OUTPUT_DIR, "train.pkl"))
    test_df.to_pickle(os.path.join(OUTPUT_DIR, "test.pkl"))

    print(f"Saved train ({len(train_df)}) and test ({len(test_df)}) dataframes in '{OUTPUT_DIR}'")