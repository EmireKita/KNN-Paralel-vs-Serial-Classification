
import time
import multiprocessing as mp
from preprocess import load_and_preprocess
from knn_serial import knn_serial
from collections import Counter


def knn_batch(args):
    X_train, y_train, X_tests, k = args
    predictions = []

    for x in X_tests:
        pred = knn_serial(X_train, y_train, x, k)
        predictions.append(pred)

    return predictions


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess(
        "Crop_recommendation.csv"
    )

    k = 5
    num_processes = mp.cpu_count()

    # Bagi data uji
    batch_size = len(X_test) // num_processes
    tasks = []

    for i in range(num_processes):
        start = i * batch_size
        end = len(X_test) if i == num_processes - 1 else (i + 1) * batch_size
        tasks.append((X_train, y_train, X_test[start:end], k))

    start_time = time.perf_counter()

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(knn_batch, tasks)

    end_time = time.perf_counter()

    # Gabungkan hasil
    predictions = []
    for part in results:
        predictions.extend(part)

    # =========================
    # 🔹 TAMPILKAN HASIL KNN
    # =========================

    print("\nJumlah data uji:", len(X_test))
    print("Waktu total KNN Paralel:", end_time - start_time, "detik")

    print("\nContoh hasil prediksi (10 data pertama):")
    print("No | Label Asli | Prediksi")
    print("-" * 35)

    for i in range(10):
        print(f"{i+1:2d} | {y_test[i]:11s} | {predictions[i]}")

    # Hitung akurasi
    correct = sum(1 for i in range(len(y_test)) if y_test[i] == predictions[i])
    accuracy = correct / len(y_test)

    print("\nAkurasi KNN:", round(accuracy * 100, 2), "%")

    # Distribusi hasil prediksi
    print("\nDistribusi hasil prediksi:")
    counter = Counter(predictions)
    for label, count in counter.most_common(5):
        print(f"{label}: {count}")
