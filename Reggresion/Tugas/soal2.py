import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('abalone.data.csv')

print("--- 5 Data Pertama ---")
print(df.head())

print("\n--- Jumlah Data Tiap Jenis Kelamin ---")
print(df['Sex'].value_counts())

# Buang kategori bayi ('I') biar Logistic Regression fokus nebak 2 pilihan aja (M atau F)
df = df[df['Sex'] != 'I']

# Pisahin mana data ukuran fisiknya (X) dan mana target jenis kelaminnya (y)
X = df.drop(['Sex'], axis=1)
y = df['Sex']

# Ubah huruf 'M' dan 'F' jadi angka biar algoritmanya nggak bingung
le = LabelEncoder()
y = le.fit_transform(y)

# Ratakan skala angkanya biar nggak ada atribut yang jomplang nilainya
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333, random_state=42)

print("\n--- Pembagian Data ---")
print("Jumlah data training :", X_train.shape)
print("Jumlah data testing  :", X_test.shape)

# Panggil modelnya dan mulai proses belajarnya
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Kasih soal ujian ke model dari data test
y_pred = model.predict(X_test)

# Cek seberapa pinter tebakan modelnya
akurasi = accuracy_score(y_test, y_pred)
print(f"\nAkurasi Model: {akurasi:.2f}")
