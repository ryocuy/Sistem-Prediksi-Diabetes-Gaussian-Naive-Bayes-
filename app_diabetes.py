import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
import os
import math

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

# ====================================================================
# LOGIKA "EXCEL-STYLE": FULL PRECISION BACKEND, 2-DIGIT DISPLAY FRONTEND
# ====================================================================
# Konstanta manual (sesuai papan tulis dosen)
PI = 3.14
E  = 2.72

def d(val, decimals=2):
    """DISPLAY ONLY — format angka ke string dengan koma.
    Jika round ke 2 digit hasilnya 0,00 padahal val > 0, 
    tampilkan 3-4 digit agar tidak nol mutlak."""
    if isinstance(val, (int, np.integer)):
        return str(val)
    # Cek apakah 2 digit menghasilkan "0.00" padahal val bukan nol
    if abs(val) > 0 and abs(val) < 0.005:
        for digits in range(3, 8):
            test = f"{val:.{digits}f}"
            if float(test) != 0.0:
                test = test.rstrip('0')
                if test.endswith('.'):
                    test += '0'
                return test.replace('.', ',')
        return f"{val:.6f}".rstrip('0').replace('.', ',')
    return f"{val:.{decimals}f}".replace('.', ',')

def d_input(val):
    """Format input value — integer jika bulat, 1 desimal jika pecahan."""
    if val == int(val):
        return f"{int(val)}"
    return f"{val:.1f}".replace('.', ',')

def d_sci(val):
    """Format angka sangat kecil sebagai notasi ilmiah LaTeX yang rapi.
    Contoh: 1,24 × 10^{-5}"""
    if val == 0:
        return "0"
    if abs(val) >= 0.01:
        return d(val)
    # Ambil eksponen
    exp = math.floor(math.log10(abs(val)))
    mantissa = val / (10 ** exp)
    return rf"{d(mantissa)} \times 10^{{{exp}}}"


def calculate_and_render_step_by_step(x_val, mean_val, var_val, fitur_name, kelas, x_display):
    """
    Menghitung Gaussian PDF FULL PRECISION di backend (seperti Excel),
    tapi MENAMPILKAN setiap langkah dengan format 2 desimal (seperti layar Excel).
    
    Menggunakan pi = 3.14 dan e = 2.72 agar akar rumusnya sama dengan papan tulis.
    TIDAK ada round() pada variabel — semua presisi penuh.
    
    Returns: hasil akhir (presisi penuh, TIDAK dibulatkan)
    """
    
    # ── BACKEND: Hitung presisi penuh (TIDAK ada round) ──
    
    # Langkah 1: 2 × pi × variance
    akar_bawah = 2 * PI * var_val
    
    # Langkah 2: sqrt(akar_bawah)
    hasil_akar = akar_bawah ** 0.5
    
    # Langkah 3: 1 / sqrt(...)
    kiri = 1 / hasil_akar if hasil_akar != 0 else 0
    
    # Langkah 4: (x - mean)^2
    selisih = x_val - mean_val
    pangkat_atas = selisih ** 2
    
    # Langkah 5: 2 × variance
    pangkat_bawah = 2 * var_val
    
    # Langkah 6: eksponen = -(pangkat_atas / pangkat_bawah)
    if pangkat_bawah != 0:
        eksponen_val = pangkat_atas / pangkat_bawah
    else:
        eksponen_val = 0
    neg_eksponen = -eksponen_val
    
    # Langkah 7: e^(neg_eksponen) — pakai E = 2.72
    kanan = E ** neg_eksponen
    
    # Langkah 8: hasil akhir = kiri × kanan (PRESISI PENUH)
    hasil_akhir = kiri * kanan
    
    # ── FRONTEND: Render step-by-step dengan FORMAT 2 desimal ──
    
    st.latex(rf"mean = {d(mean_val)}")
    st.latex(rf"var = {d(var_val)}")
    
    st.latex(rf"P({fitur_name}={x_display}|H={kelas})")
    
    # Baris 1: Rumus lengkap dengan angka
    st.latex(
        rf"= \frac{{1}}{{\sqrt{{2 \times 3,14 \times {d(var_val)}}}}}"
        rf" \times 2,72^{{-\frac{{({x_display}-{d(mean_val)})^2}}{{2 \times {d(var_val)}}}}}"
    )
    
    # Baris 2: Hasil perkalian dalam akar & hasil pangkat
    st.latex(
        rf"= \frac{{1}}{{\sqrt{{{d(akar_bawah)}}}}}"
        rf" \times 2,72^{{-\frac{{{d(pangkat_atas)}}}{{{d(pangkat_bawah)}}}}}"
    )
    
    # Baris 3: Setelah akar & setelah bagi eksponen
    st.latex(
        rf"= \frac{{1}}{{{d(hasil_akar)}}}"
        rf" \times 2,72^{{{d(neg_eksponen)}}}"
    )
    
    # Baris 4: Kiri × Kanan (display only)
    st.latex(rf"= {d(kiri)} \times {d(kanan)}")
    
    # Baris 5: Hasil akhir
    nota = ""
    if abs(hasil_akhir) < 0.005 and hasil_akhir != 0:
        nota = r" \text{ *(Boleh lebih dari 2 digit karena 0,00...)*}"
    st.latex(rf"= {d(hasil_akhir)}{nota}")
    
    return hasil_akhir  # RETURN PRESISI PENUH — bukan yang dibulatkan


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
        # HITUNG PARAMETER DARI DATASET (PRESISI PENUH)
        # ====================================================
        jml_0      = len(df[df['Outcome'] == 0])
        jml_1      = len(df[df['Outcome'] == 1])
        total_data = len(df)
        prior_0    = jml_0 / total_data   # presisi penuh
        prior_1    = jml_1 / total_data   # presisi penuh

        mean_0 = df[df['Outcome'] == 0][fitur].mean()
        var_0  = df[df['Outcome'] == 0][fitur].var()
        mean_1 = df[df['Outcome'] == 1][fitur].mean()
        var_1  = df[df['Outcome'] == 1][fitur].var()

        # ── INPUT PASIEN ──
        st.markdown("**INPUT PASIEN**")
        st.write(f"Glucose = {d_input(in_glucose)}")
        st.write(f"BMI = {d_input(in_bmi)}")
        st.write(f"Age = {d_input(in_age)}")

        st.markdown("---")
        st.markdown("**KONSTANTA YANG DIGUNAKAN (PERHITUNGAN MANUAL)**")
        st.latex(r"\pi = 3,14 \quad;\quad e = 2,72")

        st.markdown("---")

        # ── PRIOR PROBABILITY ──
        st.markdown("**PRIOR PROBABILITY**")
        st.latex(rf"P(H=0) = \frac{{{jml_0}}}{{{total_data}}} = {d(prior_0)}")
        st.latex(rf"P(H=1) = \frac{{{jml_1}}}{{{total_data}}} = {d(prior_1)}")

        st.divider()

        # ══════════════════════════════════════════════════════
        # KELAS H = 0 (SEHAT)
        # ══════════════════════════════════════════════════════
        st.markdown("### KELAS H = 0 (SEHAT)")
        
        # --- Glucose H=0 ---
        st.markdown("**1) Glucose**")
        g0_gluc = calculate_and_render_step_by_step(
            in_glucose, mean_0['Glucose'], var_0['Glucose'],
            "Glucose", 0, d_input(in_glucose)
        )

        st.markdown("---")

        # --- BMI H=0 ---
        st.markdown("**2) BMI**")
        g0_bmi = calculate_and_render_step_by_step(
            in_bmi, mean_0['BMI'], var_0['BMI'],
            "BMI", 0, d_input(in_bmi)
        )

        st.markdown("---")

        # --- Age H=0 ---
        st.markdown("**3) Age**")
        g0_age = calculate_and_render_step_by_step(
            in_age, mean_0['Age'], var_0['Age'],
            "Age", 0, d_input(in_age)
        )

        st.markdown("---")

        # ── GABUNG KELAS H=0 ──
        # Backend: kalikan presisi penuh (seperti Excel kalikan sel)
        total_0 = prior_0 * g0_gluc * g0_bmi * g0_age

        st.markdown("**Gabung Kelas H = 0**")
        st.latex(r"P(X|H=0) = P(H=0) \times P(Glucose) \times P(BMI) \times P(Age)")
        st.latex(
            rf"P(X|H=0) = {d(prior_0)} \times {d(g0_gluc)} \times {d(g0_bmi)} \times {d(g0_age)}"
        )
        st.latex(rf"= {d_sci(total_0)}")

        st.divider()

        # ══════════════════════════════════════════════════════
        # KELAS H = 1 (SAKIT)
        # ══════════════════════════════════════════════════════
        st.markdown("### KELAS H = 1 (SAKIT)")

        # --- Glucose H=1 ---
        st.markdown("**1) Glucose**")
        g1_gluc = calculate_and_render_step_by_step(
            in_glucose, mean_1['Glucose'], var_1['Glucose'],
            "Glucose", 1, d_input(in_glucose)
        )

        st.markdown("---")

        # --- BMI H=1 ---
        st.markdown("**2) BMI**")
        g1_bmi = calculate_and_render_step_by_step(
            in_bmi, mean_1['BMI'], var_1['BMI'],
            "BMI", 1, d_input(in_bmi)
        )

        st.markdown("---")

        # --- Age H=1 ---
        st.markdown("**3) Age**")
        g1_age = calculate_and_render_step_by_step(
            in_age, mean_1['Age'], var_1['Age'],
            "Age", 1, d_input(in_age)
        )

        st.markdown("---")

        # ── GABUNG KELAS H=1 ──
        # Backend: kalikan presisi penuh (seperti Excel kalikan sel)
        total_1 = prior_1 * g1_gluc * g1_bmi * g1_age

        st.markdown("**Gabung Kelas H = 1**")
        st.latex(r"P(X|H=1) = P(H=1) \times P(Glucose) \times P(BMI) \times P(Age)")
        st.latex(
            rf"P(X|H=1) = {d(prior_1)} \times {d(g1_gluc)} \times {d(g1_bmi)} \times {d(g1_age)}"
        )
        st.latex(rf"= {d_sci(total_1)}")

        st.divider()

        # ── PERBANDINGAN & KEPUTUSAN ──
        st.markdown("### PERBANDINGAN & KEPUTUSAN")
        st.latex(rf"P(X|H=0) = {d_sci(total_0)}")
        st.latex(rf"P(X|H=1) = {d_sci(total_1)}")
        st.markdown("**KEPUTUSAN:**")
        
        if total_0 > total_1:
            st.latex(r"P(H=0) > P(H=1)")
            st.latex(r"\Rightarrow \textbf{Masuk Kelas H = 0 (SEHAT)}")
        else:
            st.latex(r"P(H=1) > P(H=0)")
            st.latex(r"\Rightarrow \textbf{Masuk Kelas H = 1 (SAKIT)}")
