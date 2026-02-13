import pandas as pd

df = pd.read_csv("Coffee Shop Sales.csv")

df["Total_Price"] = df["transaction_qty"] * df["unit_price"]

print(df.head())
