import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Baca data contoh bunga Iris.
data = pd.read_csv("datasets/Iris.csv")

# 2. Pilih empat ukuran bunga sebagai data untuk klasifikasi.
fitur = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]

# 3. Sisihkan 20% data untuk menguji model; sisanya dipakai untuk latihan.
X_train, X_test, y_train, y_test = train_test_split(
    data[fitur], data["Species"], test_size=0.2, random_state=42, stratify=data["Species"]
)

# 4. Latih model dan prediksi jenis bunga pada data uji.
model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
prediksi_uji = model.predict(X_test)
print(f"Akurasi data uji: {model.score(X_test, y_test) * 100:.2f}%")

# 5. Tampilkan jumlah prediksi data uji yang benar dan salah.
benar = (prediksi_uji == y_test).sum()
plt.bar(["Benar", "Salah"], [benar, len(y_test) - benar], color=["seagreen", "tomato"])
plt.ylabel("Jumlah bunga pada data uji")
plt.title("Hasil pengujian klasifikasi Iris")
plt.savefig("iris_testing.png", bbox_inches="tight")
plt.close()

# 6. Minta ukuran bunga baru satu per satu.
nama_fitur = ["Panjang sepal (cm)", "Lebar sepal (cm)", "Panjang petal (cm)", "Lebar petal (cm)"]
ukuran = [float(input(f"Masukkan {nama}: ")) for nama in nama_fitur]

# 7. Prediksi jenis Iris berdasarkan ukuran yang dimasukkan.
print(model.predict(pd.DataFrame([ukuran], columns=fitur))[0])
