import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import datetime

# 1. Konfigurasi Halaman Dashboard
st.set_page_config(page_title="QC Inspector Dashboard", layout="wide")
st.title("🏭 Sistem Informasi Quality Control & WIP Tracker")

# 2. Inisialisasi Database Sementara (Session State)
# Ini berfungsi menggantikan MySQL untuk menyimpan riwayat selama aplikasi berjalan
if 'riwayat_qc' not in st.session_state:
    st.session_state.riwayat_qc = pd.DataFrame(columns=[
        'Waktu', 'Kategori Cacat', 'Akurasi (%)', 'Status'
    ])

# 3. Memuat Model AI (Gunakan st.cache_resource agar model tidak di-load berulang kali)
@st.cache_resource
def muat_model():
    return tf.keras.models.load_model('model_master_inspeksi_rgb.keras')

try:
    model = muat_model()
    model_siap = True
except:
    st.error("Model 'model_master_inspeksi_rgb.keras' tidak ditemukan di folder yang sama.")
    model_siap = False

kategori_master = [
    'Cacat Pengecoran (Casting Defect)', 'Korosi / Karat (Corrosion)', 
    'Retakan Halus (NEU Crazing)', 'Kotoran Material (NEU Inclusion)', 
    'Bercak/Tambalan (NEU Patches)', 'Permukaan Keropos (NEU Pitted)', 
    'Kerak Rol (NEU Rolled-in Scale)', 'Goresan (NEU Scratches)',
    '✅ OK / Normal (Mulus)'
]

# 4. Membagi Layar menjadi 2 Kolom (Kiri: Kamera/Upload, Kanan: Dashboard)
kolom_kiri, kolom_kanan = st.columns([1, 2])

with kolom_kiri:
    st.header("Kamera Inspeksi")
    file_unggah = st.file_uploader("Pindai Komponen Logam...", type=["jpg", "png", "jpeg"])
    
    if file_unggah is not None and model_siap:
        img = Image.open(file_unggah)
        st.image(img, caption="Gambar diproses...", use_column_width=True)
        
        if st.button("Jalankan Inspeksi AI"):
            # Proses Gambar
            img_resized = img.convert('RGB').resize((128, 128))
            img_array = np.expand_dims(np.array(img_resized), axis=0)
            
            # Prediksi
            prediksi = model.predict(img_array)[0]
            index_tertinggi = np.argmax(prediksi)
            nama_kelas = kategori_master[index_tertinggi]
            probabilitas = float(prediksi[index_tertinggi] * 100)
            
            status_kelayakan = "Lolos" if index_tertinggi == 8 else "Reject"
            
            # Tampilkan Hasil di Kiri
            if status_kelayakan == "Lolos":
                st.success(f"**STATUS: LOLOS**\n\nKategori: {nama_kelas} ({probabilitas:.2f}%)")
            else:
                st.error(f"**STATUS: REJECT**\n\nKategori: {nama_kelas} ({probabilitas:.2f}%)")
            
            # Simpan ke Session State (Pengganti Database)
            waktu_sekarang = datetime.datetime.now().strftime("%H:%M:%S")
            data_baru = pd.DataFrame({
                'Waktu': [waktu_sekarang],
                'Kategori Cacat': [nama_kelas],
                'Akurasi (%)': [round(probabilitas, 2)],
                'Status': [status_kelayakan]
            })
            st.session_state.riwayat_qc = pd.concat([data_baru, st.session_state.riwayat_qc], ignore_index=True)

with kolom_kanan:
    st.header("📊 Live WIP Dashboard")
    
    # Menghitung Statistik Menggunakan Pandas
    df = st.session_state.riwayat_qc
    total_inspeksi = len(df)
    total_lolos = len(df[df['Status'] == 'Lolos'])
    total_reject = len(df[df['Status'] == 'Reject'])
    
    # Menampilkan Angka Metrik (Indikator Kinerja)
    metrik1, metrik2, metrik3 = st.columns(3)
    metrik1.metric("Total Diinspeksi", total_inspeksi)
    metrik2.metric("Total Lolos", total_lolos)
    metrik3.metric("Total Reject", total_reject)
    
    st.subheader("Riwayat Inspeksi Terakhir")
    if total_inspeksi > 0:
        # Menampilkan tabel interaktif
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Belum ada komponen yang diinspeksi.")