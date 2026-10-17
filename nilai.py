import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Baca data nilai siswa.
data = pd.read_csv("datasets/dataset_nilai_siswa.csv")

# 2. Pilih lima nilai mata pelajaran sebagai data untuk klasifikasi.
mapel = ["math", "indonesian", "english", "science", "social_science"]

# 3. Sisihkan 20% data untuk menguji model; sisanya dipakai untuk latihan.
X_train, X_test, y_train, y_test = train_test_split(
    data[mapel], data["status"], test_size=0.2, random_state=42, stratify=data["status"]
)

# 4. Latih model dan prediksi status pada data uji.
model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
prediksi_uji = model.predict(X_test)
print(f"Akurasi data uji: {model.score(X_test, y_test) * 100:.2f}%")

# 5. Tampilkan jumlah prediksi data uji yang benar dan salah.
benar = (prediksi_uji == y_test).sum()
plt.bar(["Benar", "Salah"], [benar, len(y_test) - benar], color=["seagreen", "tomato"])
plt.ylabel("Jumlah siswa pada data uji")
plt.title("Hasil pengujian klasifikasi kelulusan")
plt.savefig("nilai_testing.png", bbox_inches="tight")
plt.close()

# 6. Minta nilai siswa baru satu per satu.
nama_mapel = ["Matematika", "Bahasa Indonesia", "Bahasa Inggris", "Sains", "IPS"]
nilai = [float(input(f"Nilai {nama}: ")) for nama in nama_mapel]

# 7. Prediksi dan tampilkan status kelulusan siswa.
print(model.predict(pd.DataFrame([nilai], columns=mapel))[0])
