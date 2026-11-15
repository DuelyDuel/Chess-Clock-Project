import pandas as pd

df = pd.read_csv("C:/Users/Duelz/positions_full.csv")

print(len(df))
print(df["phase"].value_counts())
print(df.describe())
print(df.groupby("phase")["move_number"].mean())
strong = df[(df["white_elo"] >= 2000) & (df["black_elo"] >= 2000)]
print(len(strong))
print(df["result"].value_counts())