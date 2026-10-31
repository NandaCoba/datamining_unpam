from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

# ==========================
# Dataset
# ==========================
data = {
    "luas": [36, 45, 50, 60, 70, 80, 90, 100, 120, 150],
    "kamar": [2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    "harga": [300, 380, 420, 500, 600, 700, 800, 900, 1100, 1400]
}

df = pd.DataFrame(data)

# ==========================
# Training Model
# ==========================
X = df[["luas", "kamar"]]
y = df["harga"]

model = LinearRegression()
model.fit(X, y)

# ==========================
# FastAPI
# ==========================
app = FastAPI(title="Prediksi Harga Rumah")

class House(BaseModel):
    luas: int
    kamar: int

@app.get("/")
def home():
    return {
        "message": "API Prediksi Harga Rumah"
    }

@app.post("/predict")
def predict(data: House):
    hasil = model.predict([[data.luas, data.kamar]])

    # Mencegah hasil prediksi negatif
    harga = max(0, round(float(hasil[0]), 2))

    return {
        "luas": data.luas,
        "kamar": data.kamar,
        "prediksi_harga_juta": harga
    }