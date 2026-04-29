import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
import os

# --- SETUP TAMPILAN WEB ---
st.set_page_config(layout="wide", page_title="Diabetes prediction using supervised machine learning")
st.title("**Diabetes prediction using supervised machine learning**")
st.write("Dianalisa menggunakan 3 Fitur Utama: **Glucose, BMI, dan Age Link Jurnal > https://www.sciencedirect.com/science/article/pii/S1877050922021858?__cf_chl_tk=6uQLqR6bQzDhcj_CexbWDRZzFcI1xGtvKuqqU3nnKQU-1777098497-1.0.1.1-P2PIN_SMC42EsxRwYA8sI5nlhRmMD_otOtqeYcL.Crw**")
st.caption("Datashet > https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
# --- PERBESAR TULISAN DI SELURUH HALAMAN ---
st.markdown("""
<style>
/* Perbesar formula LaTeX */
.katex { font-size: 1.5em !important; }

/* Perbesar teks markdown biasa (bold label langkah) */
.stMarkdown p, .stMarkdown strong, .stMarkdown em {
    font-size: 1.15rem !important;
}

/* Perbesar caption */
.stCaptionContainer p {
    font-size: 1.05rem !important;
}

/* Perbesar judul expander */
.streamlit-expanderHeader p {
    font-size: 1.2rem !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

st.divider()

# --- LOAD DATASET ---
@st.cache_data
def load_data():
    # Mengambil lokasi folder tempat script ini berada
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'diabetes.csv')
    return pd.read_csv(file_path)

try:
    df = load_data()
except FileNotFoundError:
    st.error("🚨 File 'diabetes.csv' tidak ditemukan! Pastikan file CSV sudah ditaruh di folder yang sama dengan kodingan ini.")
    st.stop()

# --- PEMBERSIHAN DATA ---
# Pastikan data diubah ke bentuk angka untuk menghindari error
for col in ['Glucose', 'BMI', 'Age', 'Outcome']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[['Glucose', 'BMI', 'Age', 'Outcome']].dropna()

# Pisahkan variabel Input (X) dan Target (y)
X = df[['Glucose', 'BMI', 'Age']]
y = df['Outcome']

# --- TRAINING MODEL AI ---
model = GaussianNB()
model.fit(X, y)

# --- AREA INPUT PASIEN ---
st.subheader("Masukkan Data Uji Pasien:")
col1, col2, col3 = st.columns(3)
with col1:
    in_glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0.0, value=120.0, step=1.0)
with col2:
    in_bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=25.0, step=0.1)
with col3:
    in_age = st.number_input("Umur (Age)", min_value=1.0, value=30.0, step=1.0)

# --- TOMBOL EKSEKUSI ---
if st.button("Analisa Probabilitas Gaussian", type="primary"):
    
    # Proses data input ke model AI
    input_data = [[in_glucose, in_bmi, in_age]]
    prediksi = model.predict(input_data)[0]
    probabilitas = model.predict_proba(input_data)[0]

    # --- TAMPILKAN HASIL KEPUTUSAN ---
    st.divider()
    st.subheader("Hasil Analisis:")
    if prediksi == 1:
        st.error(f"⚠️ **1 (POSITIF DIABETES)** - Mesin yakin sebesar **{probabilitas[1]*100:.2f}%**")
    else:
        st.success(f"✅ **0 (NEGATIF / SEHAT)** - Mesin yakin sebesar **{probabilitas[0]*100:.2f}%**")

    # --- VISUALISASI KURVA LONCENG ---
    st.divider()
    st.subheader("📊 Visualisasi Distribusi Gaussian")
    st.write("Garis hitam putus-putus adalah posisi data pasien. Kurva yang posisinya lebih tinggi pada titik tersebut menunjukkan nilai *Likelihood* (Peluang) yang lebih besar.")

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fitur = ['Glucose', 'BMI', 'Age']
    inputs = [in_glucose, in_bmi, in_age]

    for i, col in enumerate(fitur):
        mu_0 = df[df['Outcome']==0][col].mean()
        std_0 = df[df['Outcome']==0][col].std()
        
        mu_1 = df[df['Outcome']==1][col].mean()
        std_1 = df[df['Outcome']==1][col].std()

        x = np.linspace(df[col].min() - 10, df[col].max() + 10, 100)

        # Gambar Lonceng Sehat (Biru) dan Sakit (Merah)
        axs[i].plot(x, norm.pdf(x, mu_0, std_0), color='#1f77b4', linewidth=2, label='0 (Sehat)')
        axs[i].plot(x, norm.pdf(x, mu_1, std_1), color='#d62728', linewidth=2, label='1 (Sakit)')
        axs[i].axvline(inputs[i], color='black', linestyle='--', linewidth=2, label='Input Pasien')

        axs[i].set_title(f"Kurva {col}")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    st.pyplot(fig)

    # --- CHEAT SHEET PAPAN TULIS (LATEX MATH STYLE) ---
    st.divider()
    with st.expander("📐 Buka Derivasi Matematis — Gaussian Naive Bayes"):

        # ====================================================
        # HITUNG SEMUA NILAI YANG DIBUTUHKAN
        # ====================================================
        jml_0      = len(df[df['Outcome'] == 0])
        jml_1      = len(df[df['Outcome'] == 1])
        total_data = len(df)
        prior_0    = jml_0 / total_data
        prior_1    = jml_1 / total_data

        mean_0 = df[df['Outcome'] == 0][fitur].mean()
        var_0  = df[df['Outcome'] == 0][fitur].var()
        mean_1 = df[df['Outcome'] == 1][fitur].mean()
        var_1  = df[df['Outcome'] == 1][fitur].var()

        def gauss(x, m, v):
            return (1 / np.sqrt(2 * np.pi * v)) * np.exp(-((x - m)**2) / (2 * v))

        g0_gluc = gauss(in_glucose, mean_0['Glucose'], var_0['Glucose'])
        g0_bmi  = gauss(in_bmi,     mean_0['BMI'],     var_0['BMI'])
        g0_age  = gauss(in_age,     mean_0['Age'],     var_0['Age'])
        g1_gluc = gauss(in_glucose, mean_1['Glucose'], var_1['Glucose'])
        g1_bmi  = gauss(in_bmi,     mean_1['BMI'],     var_1['BMI'])
        g1_age  = gauss(in_age,     mean_1['Age'],     var_1['Age'])

        total_0 = prior_0 * g0_gluc * g0_bmi * g0_age
        total_1 = prior_1 * g1_gluc * g1_bmi * g1_age

        def fmt(val, decimals=2):
            if val < 0.01 and val > 0:
                s = f"{val:.6f}".rstrip('0')
                if s.endswith('.'): s += '0'
                return s.replace('.', ',')
            return f"{val:.{decimals}f}".replace('.', ',')
            
        def fmt_long(val):
            s = f"{val:.8f}".rstrip('0')
            if s.endswith('.'): s += '0'
            return s.replace('.', ',')

        st.markdown("**INPUT PASIEN**")
        st.write(f"Glucose = {in_glucose:.0f}")
        st.write(f"BMI = {fmt(in_bmi, 1)}")
        st.write(f"Age = {in_age:.0f}")

        st.markdown("**PRIOR PROBABILITY**")
        st.latex(rf"P(H=0) = \frac{{{jml_0}}}{{{total_data}}} = {fmt(prior_0)}")
        st.latex(rf"P(H=1) = \frac{{{jml_1}}}{{{total_data}}} = {fmt(prior_1)}")

        st.divider()

        # ── KELAS H = 0 (SEHAT) ─────────────────────────────────
        st.markdown("### KELAS H = 0 (SEHAT)")
        
        # Hitungan rinci Glucose H=0
        akar_bawah0 = 2 * 3.14 * var_0['Glucose']
        pangkat_atas0 = (in_glucose - mean_0['Glucose'])**2
        pangkat_bawah0 = 2 * var_0['Glucose']
        kiri_akhir0 = 1 / np.sqrt(akar_bawah0)
        kanan_akhir0 = np.exp(-(pangkat_atas0 / pangkat_bawah0))

        st.markdown("**1) Glucose**")
        st.latex(rf"\mu = {fmt(mean_0['Glucose'])}")
        st.latex(rf"\sigma^2 = {fmt(var_0['Glucose'])}")
        st.latex(rf"P(Glucose={in_glucose:.0f}|H=0)")
        st.latex(rf"= \frac{{1}}{{\sqrt{{2 \times 3,14 \times {fmt(var_0['Glucose'])}}}}} \times 2,72^{{-\frac{{({in_glucose:.0f}-{fmt(mean_0['Glucose'])})^2}}{{2 \times {fmt(var_0['Glucose'])}}}}}")
        st.latex(rf"= \frac{{1}}{{\sqrt{{{fmt(akar_bawah0)}}}}} \times 2,72^{{-\frac{{{fmt(pangkat_atas0)}}}{{{fmt(pangkat_bawah0)}}}}}")
        st.latex(rf"= \frac{{1}}{{{fmt(np.sqrt(akar_bawah0))}}} \times 2,72^{{-{fmt(pangkat_atas0 / pangkat_bawah0)}}}")
        st.latex(rf"= {fmt(kiri_akhir0)} \times {fmt(kanan_akhir0)}")
        st.latex(rf"= {fmt(g0_gluc)} \text{{ *(Boleh lebih dari 2 digit karena 0,00...)*}}")

        st.markdown("**2) BMI**")
        st.latex(rf"\mu = {fmt(mean_0['BMI'])}")
        st.latex(rf"\sigma^2 = {fmt(var_0['BMI'])}")
        st.latex(rf"P(BMI={fmt(in_bmi,1)}|H=0) = {fmt(g0_bmi)}")

        st.markdown("**3) Age**")
        st.latex(rf"\mu = {fmt(mean_0['Age'])}")
        st.latex(rf"\sigma^2 = {fmt(var_0['Age'])}")
        st.latex(rf"P(Age={in_age:.0f}|H=0) = {fmt(g0_age)}")

        st.markdown("**Gabung Kelas H = 0**")
        st.latex(r"P(X|H=0) = P(H=0) \times P(Glucose) \times P(BMI) \times P(Age)")
        st.latex(rf"P(X|H=0) = {fmt(prior_0)} \times {fmt(g0_gluc)} \times {fmt(g0_bmi)} \times {fmt(g0_age)}")
        st.latex(rf"= {fmt_long(total_0)}")

        st.divider()

        # ── KELAS H = 1 (SAKIT) ─────────────────────────────────
        st.markdown("### KELAS H = 1 (SAKIT)")
        
        # Hitungan rinci Glucose H=1
        akar_bawah1 = 2 * 3.14 * var_1['Glucose']
        pangkat_atas1 = (in_glucose - mean_1['Glucose'])**2
        pangkat_bawah1 = 2 * var_1['Glucose']
        kiri_akhir1 = 1 / np.sqrt(akar_bawah1)
        kanan_akhir1 = np.exp(-(pangkat_atas1 / pangkat_bawah1))

        st.markdown("**1) Glucose**")
        st.latex(rf"\mu = {fmt(mean_1['Glucose'])}")
        st.latex(rf"\sigma^2 = {fmt(var_1['Glucose'])}")
        st.latex(rf"P(Glucose={in_glucose:.0f}|H=1)")
        st.latex(rf"= \frac{{1}}{{\sqrt{{2 \times 3,14 \times {fmt(var_1['Glucose'])}}}}} \times 2,72^{{-\frac{{({in_glucose:.0f}-{fmt(mean_1['Glucose'])})^2}}{{2 \times {fmt(var_1['Glucose'])}}}}}")
        st.latex(rf"= \frac{{1}}{{\sqrt{{{fmt(akar_bawah1)}}}}} \times 2,72^{{-\frac{{{fmt(pangkat_atas1)}}}{{{fmt(pangkat_bawah1)}}}}}")
        st.latex(rf"= \frac{{1}}{{{fmt(np.sqrt(akar_bawah1))}}} \times 2,72^{{-{fmt(pangkat_atas1 / pangkat_bawah1)}}}")
        st.latex(rf"= {fmt(kiri_akhir1)} \times {fmt(kanan_akhir1)}")
        st.latex(rf"= {fmt(g1_gluc)} \text{{ *(Boleh lebih dari 2 digit karena 0,00...)*}}")

        st.markdown("**2) BMI**")
        st.latex(rf"\mu = {fmt(mean_1['BMI'])}")
        st.latex(rf"\sigma^2 = {fmt(var_1['BMI'])}")
        st.latex(rf"P(BMI={fmt(in_bmi,1)}|H=1) = {fmt(g1_bmi)}")

        st.markdown("**3) Age**")
        st.latex(rf"\mu = {fmt(mean_1['Age'])}")
        st.latex(rf"\sigma^2 = {fmt(var_1['Age'])}")
        st.latex(rf"P(Age={in_age:.0f}|H=1) = {fmt(g1_age)}")

        st.markdown("**Gabung Kelas H = 1**")
        st.latex(r"P(X|H=1) = P(H=1) \times P(Glucose) \times P(BMI) \times P(Age)")
        st.latex(rf"P(X|H=1) = {fmt(prior_1)} \times {fmt(g1_gluc)} \times {fmt(g1_bmi)} \times {fmt(g1_age)}")
        st.latex(rf"= {fmt_long(total_1)}")

        st.divider()

        # ── KEPUTUSAN ──────────────────────────────────────────
        st.markdown("### PERBANDINGAN & KEPUTUSAN")
        st.markdown("**KEPUTUSAN:**")
        
        if total_0 > total_1:
            st.latex(r"P(H=0) > P(H=1)")
            st.latex(r"\Rightarrow \textbf{Masuk Kelas H = 0 (SEHAT)}")
        else:
            st.latex(r"P(H=1) > P(H=0)")
            st.latex(r"\Rightarrow \textbf{Masuk Kelas H = 1 (SAKIT)}")
