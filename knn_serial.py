
import time
import numpy as np
from collections import Counter
from preprocess import load_and_preprocess


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_serial(X_train, y_train, x_test, k=5):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    labels = [label for _, label in k_nearest]
    return Counter(labels).most_common(1)[0][0]


if __name__ == "__main__":
    # =========================
    # 🔹 LOAD & PREPROCESS
    # =========================
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "Crop_recommendation.csv"
    )

    k = 5
    predictions = []

    # =========================
    # 🔹 EKSEKUSI SERIAL
    # =========================
    start_time = time.perf_counter()

    for x in X_test:
        pred = knn_serial(X_train, y_train, x, k)
        predictions.append(pred)

    end_time = time.perf_counter()

    # =========================
    # 🔹 OUTPUT HASIL
    # =========================
    print("\n=== Program Klasifikasi menggunakan KNN Serial ===")
    print("Kelompok Syncro")
    print("Anggota Kelompok:")
    print("Emire Kita           (1152700031)")
    print("Rafi Muhammad Akbar  (1152700017)")

    print("\n=== HASIL KNN SERIAL ===")
    print("Jumlah data latih:", len(X_train))
    print("Jumlah data uji:", len(X_test))
    print("Waktu total KNN Serial:", end_time - start_time, "detik")

    print("\nContoh hasil prediksi (10 data pertama):")
    print("No | Label Asli | Prediksi")
    print("-" * 35)

    for i in range(10):
        print(f"{i+1:2d} | {y_test[i]:11s} | {predictions[i]}")

    # =========================
    # 🔹 AKURASI
    # =========================
    correct = sum(1 for i in range(len(y_test)) if y_test[i] == predictions[i])
    accuracy = correct / len(y_test)

    print("\nAkurasi KNN Serial:", round(accuracy * 100, 2), "%")

    # =========================
    # 🔹 DISTRIBUSI LABEL
    # =========================
    print("\nDistribusi hasil prediksi (Top 5):")
    counter = Counter(predictions)
    for label, count in counter.most_common(5):
        print(f"{label}: {count}")
