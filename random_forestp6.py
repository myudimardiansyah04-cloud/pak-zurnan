import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- Import Semua Modul Scikit-learn ---
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve
)

# --- 1. PEMISAHAN DATA (DENGAN PENANGANAN ERROR) ---

# Pastikan file 'processed_kelulusan.csv' sudah ada
try:
    df = pd.read_csv("processed_kelulusan.csv")
except FileNotFoundError:
    print("ERROR: File 'processed_kelulusan.csv' tidak ditemukan. Pastikan sudah ada.")
    exit()

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Split 1: Train (70%) vs Temp (30%) - Stratifikasi aman
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# Split 2: Temp (30%) dibagi menjadi Validation (15%) dan Test (15%)
# MENGGUNAKAN TRY-EXCEPT UNTUK MENGATASI VALUEERROR (Kelas Minoritas Hanya 1 Sampel)
try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print("Pemisahan Val/Test: Stratified berhasil.")
except ValueError:
    # Jika gagal stratify, jalankan pembagian non-stratified
    print("\nPERHATIAN: Stratifikasi Val/Test gagal (kelas minoritas terlalu sedikit).")
    print("Melanjutkan dengan pemisahan non-stratified.")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

print("--- SHAPE DATA ---")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print("-" * 20)

# --- 2. PIPELINE PREPROCESSING ---

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")

# --- 3. MODEL RANDOM FOREST (BASELINE) ---

print("--- 3. PELATIHAN MODEL RANDOM FOREST ---")
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt",
    class_weight="balanced", random_state=42
)

pipe = Pipeline([("pre", pre), ("clf", rf)])
pipe.fit(X_train, y_train)

y_val_pred = pipe.predict(X_val)
print("Baseline RF — F1(val):", f1_score(y_val, y_val_pred, average="macro"))
print(classification_report(y_val, y_val_pred, digits=3))
print("-" * 20)

# --- 4. CROSS-VALIDATION DI TRAINING SET ---

print("--- 4. CROSS-VALIDATION (5 FOLD) ---")
# Diubah ke n_splits=3 karena X_train hanya memiliki 7 sampel.
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
print(f"CV F1-macro (train): {scores.mean():.4f} ± {scores.std():.4f}")
print("-" * 20)

# --- 5. HYPERPARAMETER TUNING (GRID SEARCH) ---

print("--- 5. HYPERPARAMETER TUNING ---")
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}

# skf yang digunakan di sini adalah StratifiedKFold(n_splits=3)
gs = GridSearchCV(pipe, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)
best_model = gs.best_estimator_

y_val_best = best_model.predict(X_val)
print("Best RF — F1(val):", f1_score(y_val, y_val_best, average="macro"))
print("-" * 20)

# --- 6. EVALUASI AKHIR (TEST SET) & FEATURE IMPORTANCE ---

final_model = best_model # Menggunakan model hasil tuning

print("--- 6. EVALUASI AKHIR DI TEST SET ---")
y_test_pred = final_model.predict(X_test)
print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion Matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# Plot ROC-AUC dan PR Curve
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    
    # ROC Curve
    try:
        auc_score = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC(test):", auc_score)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve (Test Set)")
        plt.legend()
        plt.tight_layout(); plt.savefig("roc_test.png", dpi=120)
    except Exception as e:
        print(f"Gagal menghitung ROC-AUC/ROC Curve: {e}")
        
    # PR Curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec); 
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve (Test Set)")
    plt.tight_layout(); plt.savefig("pr_test.png", dpi=120)

# Feature importance native (gini)
print("\n--- Feature Importance ---")
try:
    importances = final_model.named_steps["clf"].feature_importances_
    fn = final_model.named_steps["pre"].get_feature_names_out(input_features=X_train.columns)
    
    top = sorted(zip(fn, importances), key=lambda x: x[1], reverse=True)
    
    print("Top 10 Feature Importance (Gini):")
    for name, val in top[:10]:
        clean_name = name.split('__')[-1] 
        print(f"{clean_name}: {val:.4f}")
except Exception as e:
    print("Feature importance tidak tersedia:", e)
    
# --- 7. SIMPAN MODEL ---
joblib.dump(final_model, "rf_model.pkl")
print("\nModel disimpan sebagai rf_model.pkl")


# --- 8. PREDIKSI CONTOH FIKTIF ---
mdl = joblib.load("rf_model.pkl")
sample = pd.DataFrame([{
    "IPK": 3.4,
    "Jumlah_Absensi": 4,
    "Waktu_Belajar_Jam": 7,
    "Rasio_Absensi": 4/14, 
    "IPK_x_Study": 3.4*7 
}])

print("\n--- Contoh Prediksi ---")
try:
    pred = mdl.predict(sample)[0]
    print(f"Input: IPK 3.4, Absensi 4/14, Waktu Belajar 7 jam")
    print(f"Prediksi Lulus (0=Tidak, 1=Ya): {int(pred)}")
except Exception as e:
     print(f"Gagal melakukan prediksi contoh fiktif: {e}. Pastikan kolom input sesuai dengan data training.")
