# --- 1. IMPORT SEMUA LIBRARY YANG DIBUTUHKAN ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Diperlukan untuk plt.plot
import joblib # Untuk menyimpan scaler (praktek baik, meskipun tidak digunakan di sini)

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- 2. PERSIAPAN DATA DAN SCALING ---

# Pastikan file 'processed_kelulusan.csv' sudah ada
try:
    df = pd.read_csv("processed_kelulusan.csv")
except FileNotFoundError:
    print("ERROR: File 'processed_kelulusan.csv' tidak ditemukan. Pastikan sudah ada.")
    exit()

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Scaling data menggunakan StandardScaler (PENTING untuk Neural Network)
sc = StandardScaler()
Xs = sc.fit_transform(X)
# Simpan scaler untuk deployment
# joblib.dump(sc, "scaler.pkl")

# Split 1: Train (70%) vs Temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    Xs, y, test_size=0.3, stratify=y, random_state=42
)

# Split 2: Temp (30%) dibagi menjadi Validation (15%) dan Test (15%)
# MENGGUNAKAN TRY-EXCEPT UNTUK MENGATASI VALUEERROR DARI DATA SANGAT KECIL
try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print("Pemisahan Val/Test: Stratified berhasil.")
except ValueError:
    # Jika gagal stratify (karena kelas minoritas hanya 1), jalankan pembagian non-stratified
    print("\nPERHATIAN: Stratifikasi Val/Test gagal (kelas minoritas terlalu sedikit).")
    print("Melanjutkan dengan pemisahan non-stratified.")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

print("--- SHAPE DATA ---")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print("-" * 20)

# --- 3. DEFINISI MODEL NEURAL NETWORK (ANN) ---

print("--- 3. MEMBANGUN MODEL NEURAL NETWORK ---")
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid") # Klasifikasi biner
])

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="binary_crossentropy",
              metrics=["accuracy","AUC"])
model.summary()
print("-" * 20)

# --- 4. PELATIHAN MODEL ---

# Early Stopping untuk mencegah overfitting
es = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

print("--- 4. MELATIH MODEL ---")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[es],
    verbose=1
)
print("-" * 20)

# --- 5. EVALUASI AKHIR (TEST SET) & VISUALISASI ---

print("--- 5. EVALUASI AKHIR DI TEST SET ---")
# Try-except untuk mencegah crash saat evaluasi metrik pada set data yang sangat kecil
try:
    loss, acc, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Akurasi: {acc:.4f}, Test AUC: {auc:.4f}")

    # Prediksi Probabilitas dan Konversi ke Kelas (0 atau 1)
    y_proba = model.predict(X_test).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report (Test):")
    # Menambahkan zero_division=0 untuk menangani pembagian nol jika hanya ada satu kelas di y_test
    print(classification_report(y_test, y_pred, digits=3, zero_division=0)) 

except Exception as e:
    print(f"\n[ERROR PADA EVALUASI METRIK]: Gagal menghitung metrik karena data terlalu kecil atau masalah kelas tunggal. Error: {e}")
    # Masih coba plot meskipun evaluasi metrik gagal
    pass 

# Plot Learning Curve
try:
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
    plt.title("Learning Curve (ANN)")
    plt.tight_layout()
    plt.savefig("learning_curve.png", dpi=120)
    print("\nLearning Curve tersimpan ke learning_curve.png")
except Exception as e:
    print(f"\nGagal membuat plot Learning Curve: {e}")

# --- 6. SIMPAN MODEL KERAS (OPSIONAL) ---
try:
    model.save("ann_model.h5")
    print("\nModel Keras tersimpan sebagai ann_model.h5")
except Exception as e:
    print(f"Gagal menyimpan model Keras: {e}")
