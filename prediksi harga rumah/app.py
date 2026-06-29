import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Prediksi Harga Rumah", layout="wide")

st.title("🏠 Aplikasi Prediksi Harga Rumah Menggunakan Regresi Linear")

st.write("""
Aplikasi ini digunakan untuk memprediksi harga rumah berdasarkan dataset
House Sales in King County menggunakan algoritma Linear Regression.
""")

# ==========================
# Dataset otomatis
# ==========================
import os

path = os.path.join(os.path.dirname(__file__), "data.csv")
df = pd.read_csv(path)
st.subheader("Dataset")
st.dataframe(df.head())

st.subheader("Informasi Dataset")
st.write("Jumlah Baris :", df.shape[0])
st.write("Jumlah Kolom :", df.shape[1])

st.subheader("Missing Value")
st.write(df.isnull().sum())

# =========================
# VISUALISASI 1
# =========================
st.subheader("Grafik Distribusi Harga Rumah")

fig, ax = plt.subplots(figsize=(8,5))
sns.histplot(df['price'], bins=30, kde=True, ax=ax)
st.pyplot(fig)

# =========================
# VISUALISASI 2
# =========================
st.subheader("Hubungan Luas Rumah dengan Harga")

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.scatterplot(
    data=df,
    x='sqft_living',
    y='price',
    ax=ax2
)

st.pyplot(fig2)

# ======================
# PREPROCESSING
# ======================
st.subheader("Preprocessing")

df = df.select_dtypes(include=np.number)

X = df.drop('price', axis=1)
y = df['price']

X = X.fillna(X.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ======================
    # MODEL REGRESI LINEAR
    # ======================
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.subheader("Hasil Evaluasi Model")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.write("MAE :", round(mae,2))
st.write("MSE :", round(mse,2))
st.write("RMSE :", round(rmse,2))
st.write("R2 Score :", round(r2,2))

    # ======================
    # GRAFIK HASIL PREDIKSI
    # ======================
st.subheader("Grafik Prediksi vs Aktual")

fig3, ax3 = plt.subplots(figsize=(8,5))

ax3.scatter(y_test, y_pred)
ax3.set_xlabel("Harga Aktual")
ax3.set_ylabel("Harga Prediksi")
ax3.set_title("Prediksi vs Aktual")

st.pyplot(fig3)

    # ======================
    # PREDIKSI MANUAL
    # ======================
st.subheader("Prediksi Harga Rumah")

bedrooms = st.number_input("Jumlah Kamar", 1, 10, 3)
bathrooms = st.number_input("Jumlah Kamar Mandi", 1.0, 10.0, 2.0)
sqft = st.number_input("Luas Rumah (sqft)", 500, 10000, 2000)

if st.button("Prediksi"):

        input_data = pd.DataFrame(
            np.zeros((1, X.shape[1])),
            columns=X.columns
        )

        if 'bedrooms' in X.columns:
            input_data['bedrooms'] = bedrooms

        if 'bathrooms' in X.columns:
            input_data['bathrooms'] = bathrooms

        if 'sqft_living' in X.columns:
            input_data['sqft_living'] = sqft

        prediksi = model.predict(input_data)

        st.success(
            f"Perkiraan Harga Rumah : ${prediksi[0]:,.0f}"
        )
