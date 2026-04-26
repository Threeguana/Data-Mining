# Dokumentasi Praktikum Data Mining: Regression
Repositori ini berisi hasil praktikum mata kuliah Data Mining yang berfokus pada implementasi model regresi menggunakan Python.

## Praktikum 4: Linear Regression
**Tujuan:** Memprediksi jumlah penjualan produk (`Product Sales Unit`) berdasarkan anggaran iklan (`Advertising Budget`).

**Detail Implementasi:**
- **Dataset:** `Advertising.csv`.
- **Variabel Independen (X):** `Advertising_Budget_USD`.
- **Variabel Dependen (y):** `Product_Sales_Units`.
- **Metode:** `LinearRegression`.
- **Hasil:** Model menghasilkan persamaan garis lurus dengan nilai _intercept_ dan _slope_ (koefisien regresi) untuk memprediksi target. Berdasarkan pengujian, anggaran iklan sebesar 2500 diprediksi menghasilkan penjualan sebanyak 162.78 unit.

---

## Praktikum 5: Logistic Regression
**Tujuan:** Melakukan klasifikasi biner untuk membedakan sampel suara pria (_male_) dan wanita (_female_).

**Detail Implementasi:**
- **Dataset:** `voice.csv`. Terdapat total 3168 data dengan distribusi kelas seimbang (1584 pria, 1584 wanita).
- **Pra-pemrosesan:**
    - Pemutakhiran label teks menjadi format numerik menggunakan `LabelEncoder`.
    - Penyesuaian skala fitur (normalisasi) menggunakan `StandardScaler`.
    - Pembagian data latih (80%) dan data uji (20%).
- **Metode:** `LogisticRegression`.
- **Hasil:** Evaluasi performa model menggunakan metrik akurasi (`accuracy_score`) mencatatkan hasil sebesar 0.981 (98.1%) pada data pengujian.

---

## Praktikum 6: Polynomial Regression
**Tujuan:** Menganalisis hubungan non-linear antara kecepatan kendaraan (`speed_kmh`) dengan tingkat konsumsi bahan bakar (`fuel_consumption`).

**Detail Implementasi:**
- **Dataset:** `fuel_efficiency_speed.csv`.
- **Karakteristik Data:** Pemetaan scatter plot menunjukkan bahwa persebaran data membentuk kurva U (non-linear), sehingga tidak cocok diukur dengan regresi linear standar.
- **Pra-pemrosesan:** \* Pembagian data latih (80%) dan data uji (20%).
    - Transformasi fitur asli ke dalam bentuk polinomial derajat 2 menggunakan `PolynomialFeatures`.
- **Metode:** Menerapkan algoritma `LinearRegression` pada fitur yang sudah ditransformasi menjadi polinomial.
- **Hasil:** Evaluasi error menggunakan _Root Mean Square Error_ (RMSE) menghasilkan nilai 0.169 pada data latih dan 0.561 pada data uji. Model ini mampu mengakomodasi pola kurva data asli untuk memprediksi konsumsi bahan bakar, misalnya pada kecepatan 120 km/jam.
