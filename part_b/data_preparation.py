import pandas as pd

df = pd.read_csv("data/train.tsv", sep="\t")
df_small = df.sample(n=150, random_state=42)
df_small.to_csv("data/train_small.tsv", sep="\t", index=False)
