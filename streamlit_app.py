import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- PATCH BYPASS ERROR KERAS 3 ---
# Memaksa Keras untuk mengabaikan parameter 'quantization_config' yang bikin error
def apply_keras_patch():
    layer_classes = [
        tf.keras.layers.Dense, 
        tf.keras.layers.Conv2D, 
        tf.keras.layers.MaxPooling2D, 
        tf.keras.layers.Flatten, 
        tf.keras.layers.Dropout
    ]
    for layer_cls in layer_classes:
        try:
            orig_from_config = layer_cls.from_config
            def make_patched_from_config(orig_func):
                def patched(cls, config):
                    config.pop('quantization_config', None)
                    return orig_func(config)
                return classmethod(patched)
            layer_cls.from_config = make_patched_from_config(orig_from_config)
        except Exception:
            pass

apply_keras_patch()
# ----------------------------------

# 1. Konfigurasi Halaman (Diubah menjadi Wide agar lebih elegan)
st.set_page_config(page_title="Prediksi Retak Beton", layout="wide", page_icon="🏗️")

# --- CSS Kustom untuk mempercantik Tampilan ---
st.markdown("""
<style>
    .main-title { font-size: 38px; font-weight: bold; color: #1E3A8A; text-align: center; margin-bottom: 5px;}
    .sub-title { font-size: 18px; color: #6B7280; text-align: center; margin-bottom: 30px; }
    div.stButton > button:first-child {
        width: 100%;
        background-color: #2563EB;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
    }
    div.stButton > button:hover {
        background-color: #1D4ED8;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 2. Definisikan Kelas (Label)
class_names = ['Retak', 'Tidak_Retak']

# 3. Fungsi untuk Memuat Model
@st.cache_resource
def load_model():
    model_path = 'model_crack_beton.h5'
    
    if not os.path.exists(model_path):
        file_id = '1yaUHZ5p6aSxFuRYduiQKMwWpIJwf-if3' 
        try:
            gdown.download(id=file_id, output=model_path, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh model dari GDrive: {e}")
            return None
        
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal memuat model. Detail: {e}")
        return None

# 4. Fungsi untuk Memprediksi Gambar
def prediksi_gambar(image_pil, model):
    img_height = 150
    img_width = 150

    img_resized = image_pil.resize((img_width, img_height))
    img_array = np.array(img_resized, dtype=np.float32)
    
    # Penambahan normalisasi agar akurasi AI terjaga
    img_array = img_array / 255.0 
    
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    score = predictions[0]

    if len(class_names) == 2 and predictions.shape[-1] == 1:
        predicted_class_idx = 1 if score[0] >= 0.5 else 0
        konfidensi = score[0] if predicted_class_idx == 1 else 1 - score[0]
        konfidensi = konfidensi * 100
    else:
        predicted_class_idx = np.argmax(score)
        konfidensi = np.max(score) * 100

    hasil_prediksi = class_names[predicted_class_idx]
    return hasil_prediksi, konfidensi

# 5. UI Header
st.markdown('<div class="main-title">🏗️ Sistem Deteksi Cerdas Retak Beton</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Unggah foto permukaan beton untuk mendeteksi integritas strukturnya menggunakan AI.</div>', unsafe_allow_html=True)

with st.spinner("Sedang menyiapkan model AI... (Memakan waktu sesaat untuk unduh awal)"):
    model_beton = load_model()

if model_beton is None:
    st.stop()

st.divider() # Garis pembatas

# 6. Membagi Layar menjadi 2 Kolom (Kiri Upload, Kanan Hasil)
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### 📂 1. Input Gambar")
    uploaded_file = st.file_uploader("Pilih atau tarik gambar beton Anda ke sini...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Pratinjau Gambar", use_container_width=True)

with col2:
    st.markdown("### 📊 2. Hasil Analisis AI")
    if uploaded_file is not None:
        if st.button("Mulai Deteksi Keretakan 🔍"):
            with st.spinner("AI sedang memindai piksel gambar..."):
                hasil, konfidensi = prediksi_gambar(image, model_beton)
            
            # Box Hasil yang lebih rapi
            st.markdown("#### Kesimpulan:")
            if hasil == 'Retak':
                st.error(f"⚠️ **PERHATIAN: Permukaan Beton Terdeteksi RETAK**")
                st.metric(label="Tingkat Keyakinan Model", value=f"{konfidensi:.2f}%", delta="- Butuh Perhatian", delta_color="inverse")
            else:
                st.success(f"✅ **AMAN: Permukaan Beton Terdeteksi {hasil.replace('_', ' ').upper()}**")
                st.metric(label="Tingkat Keyakinan Model", value=f"{konfidensi:.2f}%", delta="Kondisi Baik")
                
    else:
        st.info("👈 Silakan unggah gambar di panel sebelah kiri terlebih dahulu.")
