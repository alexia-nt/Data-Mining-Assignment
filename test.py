import pandas as pd

train_df = pd.read_pickle("data/train_preprocessed.pkl")
test_df = pd.read_pickle("data/test_preprocessed.pkl")

print(train_df.head())
print(test_df.head())