import pandas as pd

# Membaca dataset
df = pd.read_csv("datasets/potential_customer.csv")


df.info()

# 1. Memisahkan data kategorikal dan numerikal
kategorikal = df.select_dtypes(include=["object"])
numerikal = df.select_dtypes(include=["number"])

print("DATA KATEGORIKAL")
print(kategorikal.columns.tolist())

print("\nDATA NUMERIKAL")
print(numerikal.columns.tolist())

# 2. Menampilkan tipe data setiap kolom
print("\nJENIS DATA")
print(df.dtypes)

# 3. Menghitung data duplikat
print("\nJUMLAH DUPLIKAT")
print(df.duplicated().sum())

# Menghitung nilai null per kolom
print("\nJUMLAH NULL")
print(df.isnull().sum().sum())

# Menghitung angka nol per kolom numerikal
print("\nJUMLAH ZERO")
print(numerikal.eq(0).sum())


## kategorikal = 8
## numerikal = 5