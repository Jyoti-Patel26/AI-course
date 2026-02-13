import pandas as pd

j = pd.read_csv("Coffee Shop Sales.csv")

j["Total_Price"] = j["transaction_qty"] * j["unit_price"]
print("Total Price:", total_price)

print(j.head())
total_sales = j["Total_Price"].sum()

print("Total Sales:", total_sales)
