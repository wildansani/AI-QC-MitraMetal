import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta, timezone
import base64
from io import BytesIO

st.set_page_config(page_title="QC Inspector Dashboard", layout="wide")
st.title("🏭 Sistem Informasi Quality Control & WIP Tracker")

# Konfigurasi Waktu WIB (UTC+7)
wib = timezone(timedelta(hours=7))

# Fungsi konversi gambar ke format Base64 untuk DataFrame
def get_image_base64(img_pil):
    buffered = BytesIO()
    img_pil.convert("RGB").save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{img_str}"

if 'riwayat_qc' not in st.session_state:
    st.session_state.riwayat_qc = pd.DataFrame(columns=[
        'No', 'Waktu (WIB)', 'Gambar', 'Kategori Cacat', 'Akurasi (%)', 'Status'
    ])

@st.cache_resource
def muat_model():
    return tf.keras.models.load_model('model_master_inspeksi_rgb.keras')

try:
    model = muat_model()
    model_siap = True
except:
    st.error("Model tidak ditemukan.")
    model_siap = False

kategori_master = [
    'Cacat Pengecoran (Casting Defect)', 'Korosi / Karat (Corrosion)', 
    'Retakan Halus (NEU Crazing)', 'Kotoran Material (NEU Inclusion)', 
    'Bercak/Tambalan (NEU Patches)', 'Permukaan Keropos (NEU Pitted)', 
    'Kerak Rol (NEU Rolled-in Scale)', 'Goresan (NEU Scratches)',
    '✅ OK / Normal (Mulus)'
]

kolom_kiri, kolom_kanan = st.columns([1, 2])

with kolom_kiri:
    st.header("Kamera Inspeksi")
    file_unggah = st.file_uploader("Pindai Komponen Logam...", type=["jpg", "png", "jpeg"])
    
    if file_unggah is not None and model_siap:
        img = Image.open(file_unggah)
        st.image(img, caption="Gambar diproses...", use_container_width=True)
        
        if st.button("Jalankan Inspeksi AI"):
            img_resized = img.convert('RGB').resize((128, 128))
            img_array = np.expand_dims(np.array(img_resized), axis=0)
            
            prediksi = model.predict(img_array)[0]
            index_tertinggi = np.argmax(prediksi)
            nama_kelas = kategori_master[index_tertinggi]
            probabilitas = float(prediksi[index_tertinggi] * 100)
            status_kelayakan = "Lolos" if index_tertinggi == 8 else "Reject"
            
            if status_kelayakan == "Lolos":
                st.success(f"**STATUS: LOLOS**\n\nKategori: {nama_kelas} ({probabilitas:.2f}%)")
            else:
                st.error(f"**STATUS: REJECT**\n\nKategori: {nama_kelas} ({probabilitas:.2f}%)")
            
            # Menyusun data untuk dimasukkan ke tabel
            waktu_sekarang = datetime.now(wib).strftime("%H:%M:%S")
            foto_b64 = get_image_base64(img)
            nomor_baru = len(st.session_state.riwayat_qc) + 1
            
            data_baru = pd.DataFrame({
                'No': [nomor_baru],
                'Waktu (WIB)': [waktu_sekarang],
                'Gambar': [foto_b64],
                'Kategori Cacat': [nama_kelas],
                'Akurasi (%)': [round(probabilitas, 2)],
                'Status': [status_kelayakan]
            })
            st.session_state.riwayat_qc = pd.concat([data_baru, st.session_state.riwayat_qc], ignore_index=True)

with kolom_kanan:
    st.header("📊 Live WIP Dashboard")
    
    df = st.session_state.riwayat_qc
    total_inspeksi = len(df)
    total_lolos = len(df[df['Status'] == 'Lolos'])
    total_reject = len(df[df['Status'] == 'Reject'])
    
    metrik1, metrik2, metrik3 = st.columns(3)
    metrik1.metric("Total Diinspeksi", total_inspeksi)
    metrik2.metric("Total Lolos", total_lolos)
    metrik3.metric("Total Reject", total_reject)
    
    st.subheader("Riwayat Inspeksi Terakhir")
    if total_inspeksi > 0:
        # Konfigurasi kolom khusus untuk merender gambar dan menyembunyikan indeks asli
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "No": st.column_config.NumberColumn("No.", format="%d"),
                "Gambar": st.column_config.ImageColumn("Foto Komponen"),
                "Akurasi (%)": st.column_config.NumberColumn("Akurasi (%)", format="%.2f%%")
            }
        )
    else:
        st.info("Belum ada komponen yang diinspeksi.")
