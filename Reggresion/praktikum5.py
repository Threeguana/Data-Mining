import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('voice.csv')

print("--- 5 Baris Pertama Dataset ---")
print(df.head())

# Eksplorasi dataset
print("\n--- Info Dataset ---")
df.info()

print("\n--- Deskripsi Dataset ---")
print(df.describe())

print("\n--- Jumlah Data per Kelas ---")
print(df['label'].value_counts())

# Pisahkan fitur (X) dan label (y)
X = df.drop('label', axis=1)
y = df['label']

# Encoding: Mengubah label teks (male/female) menjadi angka numerik
le = LabelEncoder()
y = le.fit_transform(y)

# Normalisasi data agar semua fitur berada pada skala yang sama
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Bagi dataset menjadi 80% data training dan 20% data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Ukuran Data ---")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Bangun dan latih model Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# Lakukan prediksi menggunakan data testing
y_pred = model.predict(X_test)

# Lakukan evaluasi model dengan menghitung nilai akurasi
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Performa Model ---")
print("Accuracy:", accuracy)
