import pandas as pd

df = pd.read_csv("results/core_unmatched_live_with_bt_decisions.csv")

print("=== df shape ===")
print(df.shape)
print()

print("=== bt_reason value counts ===")
print(df["bt_reason"].value_counts())
print()

m = df[df["bt_reason"] == "meta_prob"].copy()

print("=== meta_prob subset shape ===")
print(m.shape)
print()

print("=== Columns ===")
print(list(df.columns))
print()

print("=== First 10 meta_prob rows ===")
print(m.head(10))
print()

print("Hint: Look in the 'Columns' list above for a column that looks like probability.")
print("If you tell ChatGPT that column name, we can inspect its distribution next.")
