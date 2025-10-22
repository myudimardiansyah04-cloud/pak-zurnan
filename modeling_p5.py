import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib
from flask import Flask, request, jsonify # Import Flask harus di sini agar terdeteksi Pylance

# --- 1. PEMISAHAN DATA (Membuat X_val dan X_test) ---

# Pastikan file 'processed_kelulusan.csv' sudah ada
try:
    df = pd.read_csv("processed_kelulusan.csv")
except FileNotFoundError:
    print("ERROR: File 'processed_kelulusan.csv' tidak ditemukan. Pastikan sudah dibuat.")
    exit()

X = df.drop("Lulus", axis=1)
y = df["Lulus"]

# Pemisahan Pertama: Train (70%) vs Temp (30%)
# Stratify menggunakan 'y' penuh
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# Pemisahan Kedua: Temp (30%) dibagi dua menjadi Validation (15%) dan Test (15%)
# Hapus stratify=y_temp jika ada kelas minoritas (hanya 1 sampel), jika tidak ada error, biarkan.
try:
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
except ValueError:
    print("\nPERHATIAN: Stratifikasi gagal karena kelas minoritas (hanya 1 sampel) terlalu sedikit.")
    print("Melanjutkan tanpa stratifikasi pada pemisahan Val/Test.")
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)
    
print("--- SHAPE DATA ---")
print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
print("-" * 20)

# --- 2. PREPROCESSING PIPELINE ---

num_cols = X_train.select_dtypes(include="number").columns

pre = ColumnTransformer([
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ]), num_cols),
], remainder="drop")


# --- 3. BASELINE MODEL: LOGISTIC REGRESSION ---

print("--- 3. LOGISTIC REGRESSION (Baseline) ---")
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
pipe_lr = Pipeline([("pre", pre), ("clf", logreg)])

pipe_lr.fit(X_train, y_train)
y_val_pred_lr = pipe_lr.predict(X_val) # Rename y_val_pred to y_val_pred_lr
print("Baseline (LogReg) F1(val):", f1_score(y_val, y_val_pred_lr, average="macro"))
print(classification_report(y_val, y_val_pred_lr, digits=3))
print("-" * 20)


# --- 4. MODEL COMPLEX: RANDOM FOREST ---

print("--- 4. RANDOM FOREST ---")
rf = RandomForestClassifier(
    n_estimators=300, max_features="sqrt", class_weight="balanced", random_state=42
)
pipe_rf = Pipeline([("pre", pre), ("clf", rf)])

pipe_rf.fit(X_train, y_train)
y_val_rf = pipe_rf.predict(X_val)
print("RandomForest F1(val):", f1_score(y_val, y_val_rf, average="macro"))
print("-" * 20)


# --- 5. HYPERPARAMETER TUNING (Random Forest) ---

print("--- 5. HYPERPARAMETER TUNING (Random Forest) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param = {
    "clf__max_depth": [None, 12, 20, 30],
    "clf__min_samples_split": [2, 5, 10]
}
gs = GridSearchCV(pipe_rf, param_grid=param, cv=skf,
                  scoring="f1_macro", n_jobs=-1, verbose=1)

# PERHATIAN: GridSearchCV hanya di-fit ke X_train dan y_train
gs.fit(X_train, y_train)
print("Best params:", gs.best_params_)
print("Best CV F1:", gs.best_score_)

best_rf = gs.best_estimator_
y_val_best = best_rf.predict(X_val)
print("Best RF F1(val):", f1_score(y_val, y_val_best, average="macro"))
print("-" * 20)


# --- 6. EVALUASI AKHIR (TEST SET) ---

# Pilih model terbaik berdasarkan metrik F1 di Validation Set
if f1_score(y_val, y_val_best, average="macro") > f1_score(y_val, y_val_pred_lr, average="macro"):
    final_model = best_rf
    print("Model Akhir: Best Tuned Random Forest")
else:
    final_model = pipe_lr
    print("Model Akhir: Logistic Regression (Baseline)")
    
print("--- 6. EVALUASI DI TEST SET ---")
y_test_pred = final_model.predict(X_test)

print("F1(test):", f1_score(y_test, y_test_pred, average="macro"))
print(classification_report(y_test, y_test_pred, digits=3))
print("Confusion matrix (test):")
print(confusion_matrix(y_test, y_test_pred))

# ROC-AUC Plot dan Skor
if hasattr(final_model, "predict_proba"):
    y_test_proba = final_model.predict_proba(X_test)[:,1]
    # Handle error jika hanya ada 1 kelas di y_test (walaupun jarang terjadi)
    try:
        auc = roc_auc_score(y_test, y_test_proba)
        print("ROC-AUC(test):", auc)
        
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve (Test Set)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_test.png", dpi=120)
        print("ROC Curve disimpan sebagai roc_test.png")
    except ValueError as e:
         print(f"Tidak dapat menghitung ROC-AUC/ROC Curve: {e}")
         print("Mungkin hanya ada satu kelas di Test Set (perlu data lebih banyak)")
    except Exception as e:
         print(f"Error saat menghitung ROC-AUC: {e}")

# --- 7. SIMPAN MODEL ---
joblib.dump(final_model, "model.pkl")
print("\nModel tersimpan ke model.pkl")


# --- 8. FLASK DEPLOYMENT (API) ---

# Inisialisasi Flask, pastikan ada di luar fungsi/blok if __name__
app = Flask(__name__) 

# Muat model yang baru saja disimpan
try:
    # Menggunakan global MODEL agar bisa diakses oleh fungsi predict()
    global MODEL
    MODEL = joblib.load("model.pkl")
    print("Model 'model.pkl' berhasil dimuat untuk deployment.")
except FileNotFoundError:
    print("ERROR: Model 'model.pkl' belum tersimpan, tidak bisa menjalankan Flask API.")
    MODEL = None

@app.route("/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    try:
        data = request.get_json(force=True)  # dict fitur
        
        # Pastikan data yang masuk adalah list of dicts atau single dict
        if isinstance(data, dict):
             X = pd.DataFrame([data])
        else:
             return jsonify({"error": "Input data must be a single JSON object (dictionary) of features"}), 400
             
        yhat = MODEL.predict(X)[0]
        proba = None
        if hasattr(MODEL, "predict_proba"):
            # Ambil probabilitas untuk kelas positif (indeks 1)
            proba_value = MODEL.predict_proba(X)[:,1][0]
            # Konversi numpy float ke standard Python float
            proba = float(proba_value) 
            
        return jsonify({"prediction": int(yhat), "proba": proba})

    except Exception as e:
        # Jika ada error saat prediksi (misal: kolom kurang/lebih)
        return jsonify({"error": str(e), "message": "Check your input features matching training columns"}), 400

if __name__ == "__main__":
    if MODEL is not None:
        print("\n--- API FLASK SIAP ---")
        print("Akses: http://127.0.0.1:5000/predict (POST)")
        # Tambahkan debug=False saat deployment untuk keamanan, tapi saat development boleh True
        app.run(port=5000, debug=False)
