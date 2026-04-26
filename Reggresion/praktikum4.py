import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# load dataset dari file CSV
data = pd.read_csv("Advertising.csv")
print("--- 5 data teratas ---")
print(data.head())

# Eksplorasi data
print("\n--- Info Data ---")
data.info()
print("\n--- Deskripsi Data ---")
print(data.describe())

# Tentukan variabel independen (X) dan dependen (y)
X = data[['Advertising_Budget_USD']]
y = data['Product_Sales_Units']

# Visualisasi data dengan scatter plot
plt.scatter(X, y)
plt.xlabel("Advertising Budget (USD)")
plt.ylabel("Product Sales (Units)")
plt.title("Advertising Budget vs Product Sales")
plt.show() # pop up

# Buat dan train model Linear Regression
model = LinearRegression()
model.fit(X, y)

# Cetak nilai Intercept dan Slope
print("\n--- Hasil Model ---")
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Visualisasi hasil regresi
y_pred = model.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, color='red') # warna merah untuk garis prediksi agar lebih jelas
plt.xlabel("Advertising Budget (USD)")
plt.ylabel("Product Sales (Units)")
plt.title("Linear Regression Result")
plt.show()

# Prediksi jumlah penjualan berdasarkan budget spesifik
prediksi = model.predict([[2500]])
print("\nPrediksi penjualan (Budget 2500):", prediksi[0])
