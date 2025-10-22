import pandas as pd
# Import Matplotlib untuk menampilkan gambar
import matplotlib.pyplot as plt 
import seaborn as sns

# Bagian Pemrosesan Data Awal
df = pd.read_csv("kelulusan_mahasiswa.csv")
print(df.info())
print(df.head())
print(df.isnull().sum())
df = df.drop_duplicates()

# Bagian Visualisasi 1: boxplot
sns.boxplot(x=df['IPK'])
plt.show() # <-- Tambahkan ini untuk menampilkan boxplot

print(df.describe())

# Bagian Visualisasi 2: histplot
sns.histplot(df['IPK'], bins=10, kde=True)
plt.show() # <-- Tambahkan ini untuk menampilkan histogram

# Bagian Visualisasi 3: scatterplot
sns.scatterplot(x='IPK', y='Waktu_Belajar_Jam', data=df, hue='Lulus')
plt.show() # <-- Tambahkan ini untuk menampilkan scatterplot

# Bagian Visualisasi 4: heatmap
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show() # <-- Tambahkan ini untuk menampilkan heatmap

# Bagian Pembuatan Fitur
df['Rasio_Absensi'] = df['Jumlah_Absensi'] / 14
df['IPK_x_Study'] = df['IPK'] * df['Waktu_Belajar_Jam']
df.to_csv("processed_kelulusan.csv", index=False)

# Bagian Pemisahan Data
from sklearn.model_selection import train_test_split

X = df.drop('Lulus', axis=1)
y = df['Lulus']

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)