import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB
import os
import math
import io

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


def calculate_and_render_step_by_step(x_val, mean_val, var_val, fitur_name, kelas, x_display, lines):
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
    
    latex_mean = rf"mean = {d(mean_val)}"
    latex_var = rf"var = {d(var_val)}"
    st.latex(latex_mean)
    st.latex(latex_var)
    lines.append(('latex', latex_mean))
    lines.append(('latex', latex_var))
    
    l1 = rf"P({fitur_name}={x_display}|H={kelas})"
    st.latex(l1)
    lines.append(('latex', l1))
    
    # Baris 1: Rumus lengkap dengan angka
    l2 = (
        rf"= \frac{{1}}{{\sqrt{{2 \times 3,14 \times {d(var_val)}}}}}"
        rf" \times 2,72^{{-\frac{{({x_display}-{d(mean_val)})^2}}{{2 \times {d(var_val)}}}}}"
    )
    st.latex(l2)
    lines.append(('latex', l2))
    
    # Baris 2: Hasil perkalian dalam akar & hasil pangkat
    l3 = (
        rf"= \frac{{1}}{{\sqrt{{{d(akar_bawah)}}}}}"
        rf" \times 2,72^{{-\frac{{{d(pangkat_atas)}}}{{{d(pangkat_bawah)}}}}}"
    )
    st.latex(l3)
    lines.append(('latex', l3))
    
    # Baris 3: Setelah akar & setelah bagi eksponen
    l4 = (
        rf"= \frac{{1}}{{{d(hasil_akar)}}}"
        rf" \times 2,72^{{{d(neg_eksponen)}}}"
    )
    st.latex(l4)
    lines.append(('latex', l4))
    
    # Baris 4: Kiri × Kanan (display only)
    l5 = rf"= {d(kiri)} \times {d(kanan)}"
    st.latex(l5)
    lines.append(('latex', l5))
    
    # Baris 5: Hasil akhir
    nota = ""
    if abs(hasil_akhir) < 0.005 and hasil_akhir != 0:
        nota = r" \text{ *(Boleh lebih dari 2 digit karena 0,00...)*}"
    l6 = rf"= {d(hasil_akhir)}{nota}"
    st.latex(l6)
    lines.append(('latex', l6))
    
    return hasil_akhir  # RETURN PRESISI PENUH — bukan yang dibulatkan


def build_derivation_image(lines):
    """Render semua baris derivasi (teks & LaTeX) ke satu gambar matplotlib."""
    # Hitung tinggi gambar berdasarkan jumlah baris
    line_h = 0.45  # tinggi per baris dalam inci
    total_lines = len(lines)
    fig_h = max(8, total_lines * line_h + 2)
    
    fig_img, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    y = 0.99
    dy = 1.0 / (total_lines + 2)  # spacing per line
    
    for line_type, content in lines:
        if line_type == 'header':
            ax.text(0.02, y, content, fontsize=14, fontweight='bold',
                    verticalalignment='top', transform=ax.transAxes,
                    fontfamily='sans-serif')
            y -= dy * 1.3
        elif line_type == 'text':
            ax.text(0.03, y, content, fontsize=11,
                    verticalalignment='top', transform=ax.transAxes,
                    fontfamily='sans-serif')
            y -= dy
        elif line_type == 'latex':
            ax.text(0.05, y, f'${content}$', fontsize=13,
                    verticalalignment='top', transform=ax.transAxes,
                    fontfamily='sans-serif')
            y -= dy * 1.1
        elif line_type == 'divider':
            y -= dy * 0.3
            ax.axhline(y=y, xmin=0.02, xmax=0.98, color='gray',
                       linewidth=0.5, transform=ax.transAxes)
            y -= dy * 0.3
    
    fig_img.tight_layout()
    return fig_img


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

    # --- TOMBOL DOWNLOAD GRAFIK ---
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download Grafik Distribusi Gaussian",
        data=buf,
        file_name="grafik_gaussian_naive_bayes.png",
        mime="image/png",
    )

    # --- CHEAT SHEET PAPAN TULIS (LATEX MATH STYLE) ---
    st.divider()
    with st.expander("📐 Buka Derivasi Matematis — Gaussian Naive Bayes"):

        # Kumpulkan semua baris untuk gambar download
        deriv_lines = []

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
        deriv_lines.append(('header', 'INPUT PASIEN'))
        
        t1 = f"Glucose = {d_input(in_glucose)}"
        t2 = f"BMI = {d_input(in_bmi)}"
        t3 = f"Age = {d_input(in_age)}"
        st.write(t1); st.write(t2); st.write(t3)
        deriv_lines.append(('text', t1))
        deriv_lines.append(('text', t2))
        deriv_lines.append(('text', t3))

        st.markdown("---")
        deriv_lines.append(('divider', ''))
        
        st.markdown("**KONSTANTA YANG DIGUNAKAN (PERHITUNGAN MANUAL)**")
        deriv_lines.append(('header', 'KONSTANTA YANG DIGUNAKAN'))
        
        lk = r"\pi = 3,14 \quad;\quad e = 2,72"
        st.latex(lk)
        deriv_lines.append(('latex', lk))

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # ── PRIOR PROBABILITY ──
        st.markdown("**PRIOR PROBABILITY**")
        deriv_lines.append(('header', 'PRIOR PROBABILITY'))
        
        lp0 = rf"P(H=0) = \frac{{{jml_0}}}{{{total_data}}} = {d(prior_0)}"
        lp1 = rf"P(H=1) = \frac{{{jml_1}}}{{{total_data}}} = {d(prior_1)}"
        st.latex(lp0); st.latex(lp1)
        deriv_lines.append(('latex', lp0))
        deriv_lines.append(('latex', lp1))

        st.divider()
        deriv_lines.append(('divider', ''))

        # ══════════════════════════════════════════════════════
        # KELAS H = 0 (SEHAT)
        # ══════════════════════════════════════════════════════
        st.markdown("### KELAS H = 0 (SEHAT)")
        deriv_lines.append(('header', 'KELAS H = 0 (SEHAT)'))
        
        # --- Glucose H=0 ---
        st.markdown("**1) Glucose**")
        deriv_lines.append(('header', '1) Glucose'))
        g0_gluc = calculate_and_render_step_by_step(
            in_glucose, mean_0['Glucose'], var_0['Glucose'],
            "Glucose", 0, d_input(in_glucose), deriv_lines
        )

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # --- BMI H=0 ---
        st.markdown("**2) BMI**")
        deriv_lines.append(('header', '2) BMI'))
        g0_bmi = calculate_and_render_step_by_step(
            in_bmi, mean_0['BMI'], var_0['BMI'],
            "BMI", 0, d_input(in_bmi), deriv_lines
        )

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # --- Age H=0 ---
        st.markdown("**3) Age**")
        deriv_lines.append(('header', '3) Age'))
        g0_age = calculate_and_render_step_by_step(
            in_age, mean_0['Age'], var_0['Age'],
            "Age", 0, d_input(in_age), deriv_lines
        )

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # ── GABUNG KELAS H=0 ──
        # Backend: kalikan presisi penuh (seperti Excel kalikan sel)
        total_0 = prior_0 * g0_gluc * g0_bmi * g0_age

        st.markdown("**Gabung Kelas H = 0**")
        deriv_lines.append(('header', 'Gabung Kelas H = 0'))
        
        lg0a = r"P(X|H=0) = P(H=0) \times P(Glucose) \times P(BMI) \times P(Age)"
        lg0b = rf"P(X|H=0) = {d(prior_0)} \times {d(g0_gluc)} \times {d(g0_bmi)} \times {d(g0_age)}"
        lg0c = rf"= {d_sci(total_0)}"
        st.latex(lg0a); st.latex(lg0b); st.latex(lg0c)
        deriv_lines.append(('latex', lg0a))
        deriv_lines.append(('latex', lg0b))
        deriv_lines.append(('latex', lg0c))

        st.divider()
        deriv_lines.append(('divider', ''))

        # ══════════════════════════════════════════════════════
        # KELAS H = 1 (SAKIT)
        # ══════════════════════════════════════════════════════
        st.markdown("### KELAS H = 1 (SAKIT)")
        deriv_lines.append(('header', 'KELAS H = 1 (SAKIT)'))

        # --- Glucose H=1 ---
        st.markdown("**1) Glucose**")
        deriv_lines.append(('header', '1) Glucose'))
        g1_gluc = calculate_and_render_step_by_step(
            in_glucose, mean_1['Glucose'], var_1['Glucose'],
            "Glucose", 1, d_input(in_glucose), deriv_lines
        )

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # --- BMI H=1 ---
        st.markdown("**2) BMI**")
        deriv_lines.append(('header', '2) BMI'))
        g1_bmi = calculate_and_render_step_by_step(
            in_bmi, mean_1['BMI'], var_1['BMI'],
            "BMI", 1, d_input(in_bmi), deriv_lines
        )

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # --- Age H=1 ---
        st.markdown("**3) Age**")
        deriv_lines.append(('header', '3) Age'))
        g1_age = calculate_and_render_step_by_step(
            in_age, mean_1['Age'], var_1['Age'],
            "Age", 1, d_input(in_age), deriv_lines
        )

        st.markdown("---")
        deriv_lines.append(('divider', ''))

        # ── GABUNG KELAS H=1 ──
        # Backend: kalikan presisi penuh (seperti Excel kalikan sel)
        total_1 = prior_1 * g1_gluc * g1_bmi * g1_age

        st.markdown("**Gabung Kelas H = 1**")
        deriv_lines.append(('header', 'Gabung Kelas H = 1'))
        
        lg1a = r"P(X|H=1) = P(H=1) \times P(Glucose) \times P(BMI) \times P(Age)"
        lg1b = rf"P(X|H=1) = {d(prior_1)} \times {d(g1_gluc)} \times {d(g1_bmi)} \times {d(g1_age)}"
        lg1c = rf"= {d_sci(total_1)}"
        st.latex(lg1a); st.latex(lg1b); st.latex(lg1c)
        deriv_lines.append(('latex', lg1a))
        deriv_lines.append(('latex', lg1b))
        deriv_lines.append(('latex', lg1c))

        st.divider()
        deriv_lines.append(('divider', ''))

        # ── PERBANDINGAN & KEPUTUSAN ──
        st.markdown("### PERBANDINGAN & KEPUTUSAN")
        deriv_lines.append(('header', 'PERBANDINGAN & KEPUTUSAN'))
        
        ld0 = rf"P(X|H=0) = {d_sci(total_0)}"
        ld1 = rf"P(X|H=1) = {d_sci(total_1)}"
        st.latex(ld0); st.latex(ld1)
        deriv_lines.append(('latex', ld0))
        deriv_lines.append(('latex', ld1))
        
        st.markdown("**KEPUTUSAN:**")
        deriv_lines.append(('header', 'KEPUTUSAN:'))
        
        if total_0 > total_1:
            lk1 = r"P(H=0) > P(H=1)"
            lk2 = r"\Rightarrow \textbf{Masuk Kelas H = 0 (SEHAT)}"
            st.latex(lk1); st.latex(lk2)
            deriv_lines.append(('latex', lk1))
            deriv_lines.append(('text', '=> Masuk Kelas H = 0 (SEHAT)'))
        else:
            lk1 = r"P(H=1) > P(H=0)"
            lk2 = r"\Rightarrow \textbf{Masuk Kelas H = 1 (SAKIT)}"
            st.latex(lk1); st.latex(lk2)
            deriv_lines.append(('latex', lk1))
            deriv_lines.append(('text', '=> Masuk Kelas H = 1 (SAKIT)'))

        # ── TOMBOL DOWNLOAD DERIVASI ──
        st.markdown("---")
        fig_deriv = build_derivation_image(deriv_lines)
        buf_deriv = io.BytesIO()
        fig_deriv.savefig(buf_deriv, format="png", dpi=150, bbox_inches="tight",
                          facecolor='white', edgecolor='none')
        buf_deriv.seek(0)
        plt.close(fig_deriv)
        
        st.download_button(
            label="📥 Download Gambar Derivasi Matematis (PNG)",
            data=buf_deriv,
            file_name="derivasi_gaussian_naive_bayes.png",
            mime="image/png",
        )
