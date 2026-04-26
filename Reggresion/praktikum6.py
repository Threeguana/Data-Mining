import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("fuel_efficiency_speed.csv")

# Eksplorasi dataset
print("--- Info Dataset ---")
print(data.info())
print("\n--- Deskripsi Dataset ---")
print(data.describe())

# Pisahkan variabel independen (X) dan dependen (y)
X = data[['speed_kmh']]
y = data['fuel_consumption']

# Buat visualisasi awal dengan scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y)
plt.xlabel("Kecepatan (km/h)")
plt.ylabel("Konsumsi BBM (L/100km)")
plt.title("Hubungan Kecepatan dan Konsumsi BBM")
plt.show()

# Bagi data menjadi training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transformasi data ke bentuk polynomial degree 2
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Bangun model regresi linear menggunakan fitur polinomial
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Lakukan prediksi pada data training dan testing
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Evaluasi model dengan menghitung Root Mean Square Error (RMSE)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
print(f"\nRMSE Training: {rmse_train}")
print(f"RMSE Testing: {rmse_test}")

# Visualisasi hasil prediksi model vs data asli
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Asli')

# Mengurutkan X agar garis prediksi bisa digambar dengan mulus
X_sorted = np.sort(X.values, axis=0)
X_sorted_poly = poly.transform(X_sorted)
y_sorted_pred = model.predict(X_sorted_poly)

plt.plot(X_sorted, y_sorted_pred, color='red', label='Polynomial Regression')
plt.xlabel("Kecepatan Kendaraan (km/jam)")
plt.ylabel("Konsumsi BBM (Liter/100km)")
plt.title("Polynomial Regression")
plt.legend()
plt.show()

# Uji coba prediksi pada data baru (kecepatan 120 km/jam)
new_speed = np.array([[120]])
new_speed_poly = poly.transform(new_speed)
prediction = model.predict(new_speed_poly)
print(f"\nPrediksi konsumsi BBM pada kecepatan 120 km/jam: {prediction[0]}")
