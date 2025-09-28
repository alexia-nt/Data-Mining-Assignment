import pandas as pd

train_df = pd.read_pickle("data/train.pkl")
test_df = pd.read_pickle("data/test.pkl")

print(f"Train set: {len(train_df)} samples")
print(train_df.head())