import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
import os
import math


st.set_page_config(layout="wide", page_title="Diabetes prediction using supervised machine learning")
st.title("**Diabetes prediction using supervised machine learning**")
st.info("**Kelompok 10:** Rio | Imam | Reza")
st.write("**Kelompok 10** Membuat Program Naive Bayes Gaussian menggunakan Sklearn dan bahasa Python beserta hitungan Excel data_4")

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


def load_data():

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
for col in ['Age', 'BMI', 'Glucose', 'Insulin', 'Classification']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df[['Age', 'BMI', 'Glucose', 'Insulin', 'Classification']].dropna()


X = df[['Age', 'BMI', 'Glucose', 'Insulin']]
y = df['Classification']


# =====================================================================
# INI ADALAH PERHITUNGAN SISTEM (SCIKIT-LEARN) UTAMA
# Di sinilah AI Naive Bayes dilatih secara otomatis oleh library Python
# (Bukan manual matematis)
# =====================================================================
model = GaussianNB()
model.fit(X, y)

# --- TAMPILKAN DATASET ---
st.subheader("📚 Dataset Training (100 Data: 50 Kelas 1 (Sehat), 50 Kelas 2 (Sakit))")
st.dataframe(df, use_container_width=True)

st.divider()


st.subheader("Masukkan Data Uji Pasien:")
col1, col2, col3, col4 = st.columns(4)
with col1:
    in_age = st.number_input("Umur (Age)", min_value=1.0, value=30.0, step=1.0)
with col2:
    in_bmi = st.number_input("Indeks Massa Tubuh (BMI)", min_value=0.0, value=25.0, step=0.1)
with col3:
    in_glucose = st.number_input("Kadar Glukosa (Glucose)", min_value=0.0, value=120.0, step=1.0)
with col4:
    in_insulin = st.number_input("Insulin", min_value=0.0, value=5.0, step=0.1)


PI = 3.14
E  = 2.72

def d(val, decimals=4):
    """DISPLAY ONLY — format angka ke string dengan koma.
    Menggunakan 4 digit desimal agar hitungan manual di kertas tidak melenceng jauh,
    lalu membuang nol berlebih di belakang agar tetap rapi."""
    if isinstance(val, (int, np.integer)):
        return str(val)

    # Format dengan batas desimal
    s = f"{val:.{decimals}f}"
    
    # Hapus trailing zero dan titik/koma jika tidak diperlukan
    if '.' in s:
        s = s.rstrip('0').rstrip('.')
        # Jika ternyata angkanya 0 (misal 0.0000), kembalikan "0"
        if s == "":
            s = "0"
            
    return s.replace('.', ',')

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

    exp = math.floor(math.log10(abs(val)))
    mantissa = val / (10 ** exp)
    return rf"{d(mantissa)} \times 10^{{{exp}}}"

def d_auto(val):
    """Format presisi desimal dinamis (maksimal 4 angka penting) tanpa nol belakang panjang."""
    if val == 0:
        return "0"
    exp = math.floor(math.log10(abs(val)))
    if exp >= 0:
        s = f"{val:.4f}".rstrip('0')
    else:
        prec = abs(exp) + 4
        s = f"{val:.{prec}f}".rstrip('0')
    if s.endswith('.'): s += '0'
    return s.replace('.', ',')


def calculate_and_render_step_by_step(x_val, mean_val, var_val, fitur_name, kelas, x_display):
    """
    Menghitung Gaussian PDF FULL PRECISION di backend (seperti Excel),
    tapi MENAMPILKAN setiap langkah dengan format 2 desimal (seperti layar Excel).
    
    Menggunakan pi = 3.14 dan e = 2.72 agar akar rumusnya sama dengan papan tulis.
    TIDAK ada round() pada variabel — semua presisi penuh.
    
    Returns: hasil akhir (presisi penuh, TIDAK dibulatkan)
    """
    

    akar_bawah = 2 * PI * var_val
    

    hasil_akar = akar_bawah ** 0.5
    

    kiri = 1 / hasil_akar if hasil_akar != 0 else 0
    

    selisih = x_val - mean_val
    pangkat_atas = selisih ** 2
    

    pangkat_bawah = 2 * var_val
    

    if pangkat_bawah != 0:
        eksponen_val = pangkat_atas / pangkat_bawah
    else:
        eksponen_val = 0
    neg_eksponen = -eksponen_val

    kanan = E ** neg_eksponen
    
 
    hasil_akhir = kiri * kanan
    

    
    st.latex(rf"mean = {d(mean_val)}")
    st.latex(rf"var = {d(var_val)}")
    
    st.latex(rf"P({fitur_name}={x_display}|H={kelas})")
    
  
    st.latex(
        rf"= \frac{{1}}{{\sqrt{{2 \times 3,14 \times {d(var_val)}}}}}"
        rf" \times 2,72^{{-\frac{{({x_display}-{d(mean_val)})^2}}{{2 \times {d(var_val)}}}}}"
    )
    

    st.latex(
        rf"= \frac{{1}}{{\sqrt{{{d(akar_bawah)}}}}}"
        rf" \times 2,72^{{-\frac{{{d(pangkat_atas)}}}{{{d(pangkat_bawah)}}}}}"
    )
    

    st.latex(
        rf"= \frac{{1}}{{\sqrt{{{d(akar_bawah)}}}}}"
        rf" \times 2,72^{{{d(neg_eksponen)}}}"
    )


    st.latex(
        rf"= \frac{{1}}{{\sqrt{{{d(akar_bawah)}}}}}"
        rf" \times {d(kanan)}"
    )
    

    st.latex(
        rf"= \frac{{1}}{{{d(hasil_akar)}}}"
        rf" \times {d(kanan)}"
    )


    nota = ""
    if abs(hasil_akhir) < 0.005 and hasil_akhir != 0:
        nota = r" \text{ *(Boleh lebih dari 2 digit karena 0,00...)*}"
    st.latex(rf"= {d(kiri)} \times {d(kanan)} = {d(hasil_akhir)}{nota}")
    
    return hasil_akhir  



if st.button("Analisa Probabilitas Gaussian", type="primary"):
    
    # =====================================================================
    # INI ADALAH PROSES PREDIKSI OLEH SISTEM AI (SCIKIT-LEARN)
    # Hasil akhir (1 atau 2) ditentukan 100% oleh fungsi bawaan library ini,
    # BUKAN dari teks penjabaran Derivasi Matematis di layar.
    # =====================================================================
    input_data = [[in_age, in_bmi, in_glucose, in_insulin]]
    prediksi = model.predict(input_data)[0]
    probabilitas = model.predict_proba(input_data)[0]


    st.divider()
    st.subheader("Hasil Analisis:")
    if prediksi == 2:
        st.error(f"⚠️ **2 (SAKIT)**")
    else:
        st.success(f"✅ **1 (SEHAT)**")


    st.divider()
    st.subheader("📊 Visualisasi Distribusi Gaussian")
    st.write("Garis hitam putus-putus adalah posisi data pasien. Kurva yang posisinya lebih tinggi pada titik tersebut menunjukkan nilai *Likelihood* (Peluang) yang lebih besar.")

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    fitur = ['Age', 'BMI', 'Glucose', 'Insulin']
    inputs = [in_age, in_bmi, in_glucose, in_insulin]

    for i, col in enumerate(fitur):
        mu_0 = df[df['Classification']==1][col].mean()
        std_0 = df[df['Classification']==1][col].std()
        
        mu_1 = df[df['Classification']==2][col].mean()
        std_1 = df[df['Classification']==2][col].std()

        x = np.linspace(df[col].min() - 10, df[col].max() + 10, 100)

        # Gambar Lonceng Sehat (Biru) dan Sakit (Merah)
        axs[i].plot(x, norm.pdf(x, mu_0, std_0), color='#1f77b4', linewidth=2, label='Kelas 1 (Sehat)')
        axs[i].plot(x, norm.pdf(x, mu_1, std_1), color='#d62728', linewidth=2, label='Kelas 2 (Sakit)')
        axs[i].axvline(inputs[i], color='black', linestyle='--', linewidth=2, label='Input Pasien')

        axs[i].set_title(f"Kurva {col}")
        axs[i].legend()
        axs[i].grid(True, alpha=0.3)

    st.pyplot(fig)


    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download Grafik Distribusi Gaussian",
        data=buf,
        file_name="grafik_gaussian_naive_bayes.png",
        mime="image/png",
    )


    st.divider()
    with st.expander("📐 Buka Derivasi Matematis — Gaussian Naive Bayes"):

        # ====================================================
        # HITUNG PARAMETER DARI DATASET (PRESISI PENUH)
        # ====================================================
        jml_1      = len(df[df['Classification'] == 1])
        jml_2      = len(df[df['Classification'] == 2])
        total_data = len(df)
        prior_1    = jml_1 / total_data   # presisi penuh
        prior_2    = jml_2 / total_data   # presisi penuh

        mean_1 = df[df['Classification'] == 1][fitur].mean()
        var_1  = df[df['Classification'] == 1][fitur].var()
        mean_2 = df[df['Classification'] == 2][fitur].mean()
        var_2  = df[df['Classification'] == 2][fitur].var()


        st.markdown("**INPUT PASIEN**")
        st.write(f"Glucose = {d_input(in_glucose)}")
        st.write(f"BMI = {d_input(in_bmi)}")
        st.write(f"Age = {d_input(in_age)}")

        st.markdown("---")
        st.markdown("**KONSTANTA YANG DIGUNAKAN (PERHITUNGAN MANUAL)**")
        st.latex(r"\pi = 3,14 \quad;\quad e = 2,72")

        st.markdown("---")

 
        st.markdown("**PRIOR PROBABILITY**")
        st.latex(rf"P(Kelas=1) = \frac{{{jml_2}}}{{{total_data}}} = {d(prior_2)}")
        st.latex(rf"P(Kelas=2) = \frac{{{jml_2}}}{{{total_data}}} = {d(prior_2)}")

        st.divider()


        st.markdown("### KELAS 1 (SEHAT)")
        
        # --- Age Kelas=1 ---
        st.markdown("**1) Age**")
        g1_age = calculate_and_render_step_by_step(
            in_age, mean_1['Age'], var_1['Age'],
            "Age", 1, d_input(in_age)
        )

        st.markdown("---")

        # --- BMI Kelas=1 ---
        st.markdown("**2) BMI**")
        g1_bmi = calculate_and_render_step_by_step(
            in_bmi, mean_1['BMI'], var_1['BMI'],
            "BMI", 1, d_input(in_bmi)
        )

        st.markdown("---")

        # --- Glucose Kelas=1 ---
        st.markdown("**3) Glucose**")
        g1_gluc = calculate_and_render_step_by_step(
            in_glucose, mean_1['Glucose'], var_1['Glucose'],
            "Glucose", 1, d_input(in_glucose)
        )

        st.markdown("---")

        # --- Insulin Kelas=1 ---
        st.markdown("**4) Insulin**")
        g1_insulin = calculate_and_render_step_by_step(
            in_insulin, mean_1['Insulin'], var_1['Insulin'],
            "Insulin", 1, d_input(in_insulin)
        )

        st.markdown("---")

        # ── GABUNG KELAS Kelas=1 ──
        total_1 = prior_1 * g1_age * g1_bmi * g1_gluc * g1_insulin

        st.markdown("**Gabung Kelas 1**")
        st.latex(r"P(X|Kelas=1) = P(Kelas=1) \times P(Age) \times P(BMI) \times P(Glucose) \times P(Insulin)")
        st.latex(
            rf"P(X|Kelas=1) = {d(prior_1)} \times {d(g1_age)} \times {d(g1_bmi)} \times {d(g1_gluc)} \times {d(g1_insulin)}"
        )
        st.latex(rf"= {d_sci(total_1)} = {d_auto(total_1)}")

        st.divider()


        st.markdown("### KELAS 2 (SAKIT)")

        # --- Age Kelas=2 ---
        st.markdown("**1) Age**")
        g2_age = calculate_and_render_step_by_step(
            in_age, mean_2['Age'], var_2['Age'],
            "Age", 2, d_input(in_age)
        )

        st.markdown("---")

        # --- BMI Kelas=2 ---
        st.markdown("**2) BMI**")
        g2_bmi = calculate_and_render_step_by_step(
            in_bmi, mean_2['BMI'], var_2['BMI'],
            "BMI", 2, d_input(in_bmi)
        )

        st.markdown("---")

        # --- Glucose Kelas=2 ---
        st.markdown("**3) Glucose**")
        g2_gluc = calculate_and_render_step_by_step(
            in_glucose, mean_2['Glucose'], var_2['Glucose'],
            "Glucose", 2, d_input(in_glucose)
        )

        st.markdown("---")

        # --- Insulin Kelas=2 ---
        st.markdown("**4) Insulin**")
        g2_insulin = calculate_and_render_step_by_step(
            in_insulin, mean_2['Insulin'], var_2['Insulin'],
            "Insulin", 2, d_input(in_insulin)
        )

        st.markdown("---")

        # ── GABUNG KELAS Kelas=2 ──
        total_2 = prior_2 * g2_age * g2_bmi * g2_gluc * g2_insulin

        st.markdown("**Gabung Kelas 2**")
        st.latex(r"P(X|Kelas=2) = P(Kelas=2) \times P(Age) \times P(BMI) \times P(Glucose) \times P(Insulin)")
        st.latex(
            rf"P(X|Kelas=2) = {d(prior_2)} \times {d(g2_age)} \times {d(g2_bmi)} \times {d(g2_gluc)} \times {d(g2_insulin)}"
        )
        st.latex(rf"= {d_sci(total_2)} = {d_auto(total_2)}")

        st.divider()


        st.markdown("### PERBANDINGAN & KEPUTUSAN")
        st.latex(rf"P(X|Kelas=1) = {d_sci(total_1)} = {d_auto(total_1)}")
        st.latex(rf"P(X|Kelas=2) = {d_sci(total_2)} = {d_auto(total_2)}")
        st.markdown("**KEPUTUSAN:**")
        
        if total_1 > total_2:
            st.latex(r"P(Kelas=1) > P(Kelas=2)")
            st.latex(r"\Rightarrow \textbf{Masuk Kelas 1 (SEHAT)}")
        else:
            st.latex(r"P(Kelas=2) > P(Kelas=1)")
            st.latex(r"\Rightarrow \textbf{Masuk Kelas 2 (SAKIT)}")
