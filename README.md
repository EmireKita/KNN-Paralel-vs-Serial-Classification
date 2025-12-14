# KNN-Paralel-vs-Serial-Classification
Program yang membandingkan sistem serial dengan sistem paralel untuk klasifikasi menggunakan KNN pada dataset Crop_Recommendation.csv
# README – Implementasi KNN Serial dan Paralel (Python)

## 1. Gambaran Umum

Proyek ini mengimplementasikan algoritma **K-Nearest Neighbor (KNN)** dalam dua pendekatan:

1. **KNN Serial** → sebagai *baseline* (eksekusi berurutan)
2. **KNN Paralel** → menggunakan `multiprocessing` untuk mempercepat eksekusi

Kedua implementasi:

* Menggunakan dataset yang sama
* Menggunakan preprocessing yang sama
* Menggunakan nilai `k` yang sama

Dengan demikian, perbedaan hasil hanya terletak pada **waktu eksekusi**, bukan akurasi.

---

## 2. KNN Serial

### 2.1 Konsep Dasar

Pada KNN serial, setiap data uji diproses **satu per satu** secara berurutan:

* Ambil satu data uji
* Hitung jarak ke seluruh data latih
* Ambil `k` tetangga terdekat
* Tentukan label mayoritas
* Simpan hasil prediksi

Tidak ada pembagian tugas atau proses paralel.

---

### 2.2 Alur Eksekusi Serial

```
X_test[0] → KNN → pred[0]
X_test[1] → KNN → pred[1]
X_test[2] → KNN → pred[2]
   ...
X_test[n] → KNN → pred[n]
```

Atau dalam bentuk alur:

```
X_test
  ↓
Loop satu data uji
  ↓
Hitung jarak ke seluruh X_train
  ↓
Voting k tetangga terdekat
  ↓
Simpan prediksi
```

---

### 2.3 Karakteristik KNN Serial

* Menggunakan satu core CPU
* Implementasi sederhana
* Waktu eksekusi relatif lebih lama
* Digunakan sebagai *baseline* perbandingan

---

## 3. KNN Paralel (Multiprocessing)

### 3.1 Konsep Paralelisasi

Pada KNN paralel, percepatan dilakukan dengan cara:

* **Data latih (`X_train`, `y_train`) dibagikan ke semua proses**
* **Data uji (`X_test`) dibagi menjadi beberapa batch**
* Setiap proses mengklasifikasikan satu batch data uji
* Hasil dari semua proses digabung kembali

Karena setiap data uji bersifat **independen**, pendekatan ini sangat cocok untuk KNN.

---

### 3.2 Pendistribusian Tugas (Dengan Nomor Baris)

Cuplikan kode berikut menunjukkan proses pembagian data uji:

```python
1  num_processes = mp.cpu_count()
2
3  batch_size = len(X_test) // num_processes
4  tasks = []
5
6  for i in range(num_processes):
7      start = i * batch_size
8      end = len(X_test) if i == num_processes - 1 else (i + 1) * batch_size
9      tasks.append((X_train, y_train, X_test[start:end], k))
```

Penjelasan singkat:

* Baris 1: Menentukan jumlah proses berdasarkan core CPU
* Baris 3: Menentukan ukuran batch data uji
* Baris 6–9: Membagi data uji ke beberapa proses

---

### 3.3 Graf Paralelisasi

```
                    X_train, y_train
                           |
        ------------------------------------------------
        |              |              |               |
     Process 1      Process 2      Process 3       Process N
        |              |              |               |
  X_test[0:a]    X_test[a:b]    X_test[b:c]    X_test[...]
        |              |              |               |
   KNN Serial     KNN Serial     KNN Serial      KNN Serial
        ------------------------------------------------
                           |
                   Gabung Hasil Prediksi
```

Makna graf:

* Garis horizontal menunjukkan proses berjalan **paralel**
* Setiap proses bekerja independen
* Sinkronisasi hanya terjadi saat penggabungan hasil

---

### 3.4 Karakteristik KNN Paralel

* Memanfaatkan multi-core CPU
* Waktu eksekusi lebih cepat
* Ada overhead pembuatan dan manajemen proses
* Akurasi sama dengan versi serial

---

## 4. Perbandingan Singkat

| Aspek          | KNN Serial | KNN Paralel  |
| -------------- | ---------- | ------------ |
| Model eksekusi | Berurutan  | Paralel      |
| Core CPU       | 1          | Banyak       |
| Waktu eksekusi | Lebih lama | Lebih cepat  |
| Kompleksitas   | Rendah     | Lebih tinggi |
| Akurasi        | Sama       | Sama         |

---

## 5. Kesimpulan

* KNN serial digunakan sebagai **baseline performa**
* KNN paralel mempercepat proses klasifikasi tanpa mengubah hasil
* Paralelisasi efektif karena KNN tidak memiliki dependensi antar data uji
* Cocok untuk studi perbandingan **serial vs paralel** dalam sistem terdistribusi
