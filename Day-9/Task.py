import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

df = pd.read_csv("coffee_shop_sales.csv")

sales_by_category = df.groupby("product_category")["unit_price"].sum()

plt.figure(figsize=(10, 6))
sales_by_category.plot(kind="bar")

plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()
