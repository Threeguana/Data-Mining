import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("newYork.csv")

# Mengambil data hari ke- berapa dan harga rata-rata
data['Harga_Rata'] = (data['Low Price'] + data['High Price']) / 2
data['Tanggal'] = pd.to_datetime(data['Date'])
data['Hari_ke'] = data['Tanggal'].dt.dayofyear

# Eksplorasi dataset
print("--- Info Dataset ---")
print(data[['Hari_ke', 'Harga_Rata']].info())
print("\n--- Deskripsi Dataset ---")
print(data[['Hari_ke', 'Harga_Rata']].describe())

# Pisahkan variabel independen (X) dan dependen (y)
X = data[['Hari_ke']]
y = data['Harga_Rata']

# Buat visualisasi awal dengan scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y)
plt.xlabel("Hari ke- (dalam setahun)")
plt.ylabel("Harga Rata-rata ($)")
plt.title("Hubungan Waktu dan Harga Labu")
plt.show()

# Bagi data menjadi training (66.7%) dan testing (33.3%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

# Transformasi data ke bentuk polynomial degree
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
print(f"\nRMSE Training: {rmse_train:.2f}")
print(f"RMSE Testing: {rmse_test:.2f}")

# Visualisasi hasil prediksi model vs data asli
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='blue', label='Data Asli')

# Mengurutkan X agar garis prediksi bisa digambar dengan mulus
X_sorted = np.sort(X.values, axis=0)

# Jadikan DataFrame kembali agar nama kolomnya dikenali
X_sorted_df = pd.DataFrame(X_sorted, columns=['Hari_ke'])

# Gunakan data yang sudah diperbaiki untuk transformasi dan prediksi
X_sorted_poly = poly.transform(X_sorted_df)
y_sorted_pred = model.predict(X_sorted_poly)

plt.plot(X_sorted, y_sorted_pred, color='red', label='Polynomial Regression')
plt.xlabel("Hari ke- (dalam setahun)")
plt.ylabel("Harga Rata-rata ($)")
plt.title("Prediksi Harga Labu (Polynomial Regression)")
plt.legend()
plt.show()
