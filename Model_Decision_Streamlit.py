import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
import os

# ================== Membaca Data ==================
# Membaca file CSV
file_path = "D:\OneDrive\Dokumen\Kampus Merdeka Stupen\maribelajar\Capstone Project\Project\datamodel.csv"
df = pd.read_csv(file_path, delimiter=';')

# Kolom yang akan diabaikan
ignore_columns = ['ID_Provinsi', 'Provinsi', 'Tahun']

# Fitur dan label
features = [col for col in df.columns if col not in ignore_columns + ['Pengangguran']]
label_column = 'Pengangguran'

# Pisahkan X (fitur) dan y (label)
X = df[features]
y = df[label_column]

# ================== Model Decision Tree ==================
# Membuat dan melatih model Decision Tree
model = DecisionTreeRegressor(random_state=42).fit(X, y)

# Evaluasi Model
y_pred = model.predict(X)  # Prediksi y menggunakan X
mse = mean_squared_error(y, y_pred)  # Menghitung Mean Squared Error
r2 = r2_score(y, y_pred)  # Menghitung R-squared

# ================== Aplikasi Streamlit ==================
st.set_page_config(
    page_title="Prediksi Decision Tree",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Styling Streamlit
st.markdown(
    """
    <style>
    .stApp {
        background-color: ##181818; /* Warna latar belakang */
        color: #FFFFFF; /* Warna teks */
    }
    .header {
        font-size: 28px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== Sidebar ==================
st.sidebar.title("Tentang Aplikasi")

# Menampilkan kolom X dinamis
input_columns = ', '.join(features)
st.sidebar.markdown(f"""
- **Model yang digunakan**: Decision Tree  
- **Legenda**:
  - Nama provinsi yang ingin diprediksi
  - Jumlah Angkatan Kerja (AK) provinsi
  - Pengeluaran Perkapita provinsi
  - Produk Domestik Regional Bruto (PDRB) provinsi
  - Upah rata-rata per jam pekerja 
  - Pertumbuhan Produksi Industri Kecil dan Menengah (PP IKM) (Year to Year)
  - PDRB (Rupiah) perkapita provinsi atas harga berlaku
  - Jumlah Penduduk (dalam ribu jiwa) provinsi  
  - Gini ratio provinsi
  - Jumlah perusahaan konstruksi skala kecil (PK-K)
  - Jumlah perusahaan konstruksi skala menengah (PK-M)
  - Jumlah perusahaan konstruksi skala besar (PK-B)
  - Jumlah Industri Mikro
  - Jumlah Industri Kecil
- **Interaktif**: Dibangun dengan Streamlit  
""")

st.sidebar.title("Evaluasi Model")
st.sidebar.markdown(f"""
- **Mean Squared Error (MSE)**: {mse:.2f}  
- **R-squared (R²)**: {r2:.2f}  
""")

# ================== Judul Aplikasi ==================
st.markdown('<div class="header">Prediksi Tingkat Pengangguran di Indonesia Berdasarkan Faktor Ekonomi</div>', unsafe_allow_html=True)
st.write("Aplikasi ini memprediksi tingkat pengangguran berdasarkan input variabel yang relevan.")

# ================== Input Data Interaktif ==================
st.header("Masukkan Data Input")

input_values = []

# Membuat form input data
with st.form(key="input_form"):
    # Input data untuk "Nama Provinsi"
    nama_provinsi = st.text_input("Nama Provinsi")

    # Memasukan fitur
    for col in features:
        value = st.number_input(
            f"{col}", min_value=-1.797e+308, value=0.0, step=0.01, format="%.2f"
        )
        input_values.append(value)
    submit_button = st.form_submit_button(label="🎯 Prediksi")


# ================== Fungsi Simpan CSV ==================
def save_or_append_prediction_to_csv(data, file_name="predictions.csv", folder_path=r"C:\Users\ASUS\Komunitas Maribelajar Indonesia\CP7 - 02 - Bhinneka - Documents\General\Predictions"):
    # Membuat folder jika belum ada
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    file_path = os.path.join(folder_path, file_name)
    
    try:
        # Jika file sudah ada, tambahkan data baru
        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path)
            combined_data = pd.concat([existing_data, data], ignore_index=True)
        else:
            combined_data = data

        # Menulis ulang file CSV dengan mode aman
        combined_data.to_csv(file_path, index=False)
        
        return file_path
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menyimpan file: {e}")
        raise e

# ================== Hasil Prediksi ==================
if submit_button:
    try:
        # Melakukan prediksi
        prediksi = model.predict([input_values])
        st.success(f"🎉 Prediksi Tingkat Pengangguran: **{prediksi[0]:,.2f}**")
        st.balloons()

        # Menyimpan hasil input dan prediksi ke dalam file CSV
        hasil_prediksi = pd.DataFrame([input_values], columns=features)
        hasil_prediksi['Prediksi Pengangguran'] = prediksi[0]
        hasil_prediksi['Nama Provinsi'] = nama_provinsi

        file_path = save_or_append_prediction_to_csv(hasil_prediksi)
        st.info(f"📁 Hasil prediksi disimpan di file: `{file_path}`")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# ================== Tampilkan Data & Ringkasan ==================
st.markdown("### Ringkasan Data Input:")
st.write(df.head())

st.markdown("### Statistik Data:")
st.write(df.describe())

# ================== Footer ==================
st.markdown(
    """
    <div style="text-align: center; margin-top: 30px;">
        <b style="color: #FFFFFF;">Dibuat dengan ❤️ oleh Tim Bhinneka</b>
    </div>
    """,
    unsafe_allow_html=True,
)
