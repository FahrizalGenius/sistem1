import streamlit as st
import pandas as pd
import numpy as np
from utils.state import initialize_state, STAGES, is_complete, can_open, mark_complete
from utils.ui import apply_custom_css, render_sidebar, PAGE_FILES, PAGE_ICONS
from utils.data import generate_sample_paddy_data

st.set_page_config(
    page_title="System 1 | Peramalan Panen Padi AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_state()
apply_custom_css()
render_sidebar()

# --- HERO SECTION ---
st.markdown("""
<div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%); border-radius: 20px; padding: 36px 40px; color: white; margin-bottom: 28px; box-shadow: 0 12px 32px rgba(6, 78, 59, 0.22); position: relative; overflow: hidden;">
    <div style="display: inline-block; background: rgba(52, 211, 153, 0.2); border: 1px solid rgba(167, 243, 208, 0.4); padding: 4px 14px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #a7f3d0; margin-bottom: 12px;">
        🌾 MBKM Riset • Artificial Intelligence & Agriculture 4.0
    </div>
    <h1 style="font-size: 2.3rem; font-weight: 800; margin: 0 0 12px 0; color: #ffffff; letter-spacing: -0.5px;">
        Sistem Cerdas Peramalan Hasil Panen Padi
    </h1>
    <p style="font-size: 1.05rem; margin: 0 0 24px 0; color: #d1fae5; line-height: 1.6; max-width: 900px;">
        Platform komparasi peramalan deret waktu (<em>Time Series Forecasting</em>) menggunakan arsitektur 
        <strong>Deep Learning LSTM Baseline</strong> dan optimasi metaheuristik <strong>LSTM-Particle Swarm Optimization (LSTM-PSO)</strong> 
        untuk estimasi produksi panen padi yang akurat dan objektif.
    </p>
</div>
""", unsafe_allow_html=True)

# Quick Action Bar
c_cta1, c_cta2, c_spacer = st.columns([1.5, 1.8, 2.7])
with c_cta1:
    st.page_link("pages/1_Upload_Dataset.py", label="Mulai Pipeline Sekarang ➔", icon="🚀", use_container_width=True)

with c_cta2:
    if st.button("⚡ Muat Contoh Dataset Demo (10 Tahun)", use_container_width=True, help="Otomatis memuat data simulasi panen padi Jawa Barat/Jawa Timur (120 bulan)"):
        df_sample = generate_sample_paddy_data()
        st.session_state.raw_df = df_sample
        st.session_state.source_name = "Dataset_Demo_Panen_Padi_120Bulan.csv"
        mark_complete(1)
        st.success("✅ Dataset simulasi panen padi berhasil dimuat! Anda dapat langsung lanjut ke Tahap 2.")
        st.rerun()

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# --- STATUS RINGKAS PENELITIAN (KPI ROW) ---
df = st.session_state.get("raw_df")
preprocessed = st.session_state.get("preprocessed_df")
model = st.session_state.get("trained_model")
metrics = st.session_state.get("eval_metrics")

k1, k2, k3, k4 = st.columns(4)

with k1:
    if df is not None:
        st.metric("📊 Dataset", f"{len(df):,} Baris", f"{len(df.columns)} Kolom")
    else:
        st.metric("📊 Dataset", "Belum Ada", "Menunggu upload")

with k2:
    if preprocessed is not None:
        target = st.session_state.get("target_col", "Target")
        feats_count = len(st.session_state.get("feature_cols", []))
        st.metric("⚙️ Fitur & Target", f"{feats_count} Fitur", f"Target: {target[:12]}..")
    else:
        st.metric("⚙️ Fitur & Target", "Belum Diatur", "Tahap 3")

with k3:
    if model is not None:
        m_name = st.session_state.get("model_choice", "LSTM")
        st.metric("🧠 Model Aktif", m_name, "Model dilatih")
    else:
        st.metric("🧠 Model Aktif", "Belum Dipilih", "Tahap 5 & 6")

with k4:
    if metrics and "R²" in metrics and np.isfinite(metrics["R²"]):
        r2_pct = metrics["R²"]
        st.metric("📈 Akurasi (R²)", f"{r2_pct:.4f}", f"MAPE: {metrics.get('MAPE (%)', 0):.2f}%")
    else:
        st.metric("📈 Akurasi (R²)", "—", "Menunggu evaluasi")

st.markdown("<div style='margin-bottom: 30px;'></div>", unsafe_allow_html=True)

# --- WORKFLOW TAHAPAN PENELITIAN ---
st.markdown("""
<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px;">
    <div>
        <h3 style="margin: 0; font-size: 1.3rem; font-weight: 800; color: #0f172a;">🗺️ Alur Kerja & Tahapan Pipeline</h3>
        <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #334155; font-weight: 500;">Setiap tahap dirancang bertahap untuk menjaga integritas metodologi penelitian deret waktu.</p>
    </div>
</div>
""", unsafe_allow_html=True)

steps_info = [
    ("1. Upload Dataset", "Unggah data historis panen padi dalam format CSV atau Excel (.xlsx/.xls).", "📂"),
    ("2. Pengecekan Data", "Audit missing value, identifikasi baris duplikat, dan tentukan penanganan.", "🔍"),
    ("3. Preprocessing", "Seleksi fitur masukan, tentukan variabel target, dan normalisasi (Min-Max/Standard).", "⚙️"),
    ("4. Pengaturan Data", "Atur urutan waktu kronologis, time step sequence, dan rasio split latih/uji.", "⏱️"),
    ("5. Pemilihan Model", "Pilih arsitektur LSTM Baseline atau optimasi metaheuristik LSTM-PSO.", "🧠"),
    ("6. Training & Prediksi", "Latih model tanpa data leakage, pantau kurva loss, dan hasilkan prediksi.", "🚀"),
    ("7. Evaluasi", "Analisis metrik evaluasi (MAE, MSE, RMSE, MAPE, R²), grafik aktual vs prediksi, & ekspor CSV.", "📊"),
]

# Tampilkan Grid Tahapan 2 Baris (4 kolom & 3 kolom)
grid_row1 = st.columns(4)
for i in range(4):
    stage_idx = i + 1
    title, desc, icon = steps_info[i]
    done = is_complete(stage_idx)
    openable = can_open(stage_idx)
    page_file = PAGE_FILES[i]

    with grid_row1[i]:
        status_badge = "✅ Selesai" if done else ("🔓 Terbuka" if openable else "🔒 Terkunci")
        border_color = "#10b981" if done else ("#94a3b8" if openable else "#cbd5e1")
        bg_card = "#ffffff" if openable else "#f1f5f9"
        
        st.markdown(f"""
        <div style="background: {bg_card}; border: 1.5px solid {border_color}; border-radius: 14px; padding: 18px; min-height: 180px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <span style="font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: {'#d1fae5' if done else ('#e2e8f0' if openable else '#f8fafc')}; color: {'#065f46' if done else '#334155'};">{status_badge}</span>
                </div>
                <h4 style="margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 700; color: #0f172a;">{title}</h4>
                <p style="margin: 0; font-size: 0.82rem; color: #334155; line-height: 1.45; font-weight: 500;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if openable:
            st.page_link(page_file, label=f"Buka Tahap {stage_idx} ➔", use_container_width=True)
        else:
            st.button(f"Terkunci 🔒", key=f"btn_locked_{stage_idx}", disabled=True, use_container_width=True)

st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

grid_row2 = st.columns(3)
for i in range(4, 7):
    stage_idx = i + 1
    title, desc, icon = steps_info[i]
    done = is_complete(stage_idx)
    openable = can_open(stage_idx)
    page_file = PAGE_FILES[i]

    with grid_row2[i - 4]:
        status_badge = "✅ Selesai" if done else ("🔓 Terbuka" if openable else "🔒 Terkunci")
        border_color = "#10b981" if done else ("#94a3b8" if openable else "#cbd5e1")
        bg_card = "#ffffff" if openable else "#f1f5f9"
        
        st.markdown(f"""
        <div style="background: {bg_card}; border: 1.5px solid {border_color}; border-radius: 14px; padding: 18px; min-height: 180px; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 12px rgba(0,0,0,0.04);">
            <div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem;">{icon}</span>
                    <span style="font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: {'#d1fae5' if done else ('#e2e8f0' if openable else '#f8fafc')}; color: {'#065f46' if done else '#334155'};">{status_badge}</span>
                </div>
                <h4 style="margin: 0 0 6px 0; font-size: 0.98rem; font-weight: 700; color: #0f172a;">{title}</h4>
                <p style="margin: 0; font-size: 0.82rem; color: #334155; line-height: 1.45; font-weight: 500;">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if openable:
            st.page_link(page_file, label=f"Buka Tahap {stage_idx} ➔", use_container_width=True)
        else:
            st.button(f"Terkunci 🔒", key=f"btn_locked_{stage_idx}", disabled=True, use_container_width=True)

st.markdown("<div style='margin-bottom: 36px;'></div>", unsafe_allow_html=True)

# --- METODOLOGI & KOMPARASI MODEL ---
st.markdown("### 🔬 Panduan Metodologi & Arsitektur Model")

tab1, tab2, tab3 = st.tabs(["🧠 Arsitektur LSTM vs LSTM-PSO", "🛡️ Protokol Anti-Data Leakage", "📐 Pedoman Interpretasi Metrik"])

with tab1:
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        st.markdown("""
        <div class="agri-card">
            <h4 style="color: #065f46; margin-top: 0;">1. LSTM Baseline (Standard)</h4>
            <p style="font-size: 0.88rem; color: #475569;">
                Long Short-Term Memory adalah varian Recurrent Neural Network (RNN) yang dirancang mengatasi 
                <em>vanishing gradient</em> melalui mekanisme gating:
            </p>
            <ul style="font-size: 0.85rem; color: #334155;">
                <li><strong>Forget Gate:</strong> Membuang informasi historis yang sudah tidak relevan.</li>
                <li><strong>Input Gate:</strong> Menyimpan informasi deret waktu baru ke dalam sel status.</li>
                <li><strong>Output Gate:</strong> Memutuskan informasi yang dikeluarkan sebagai prediksi.</li>
            </ul>
            <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 0;">
                <em>Karakteristik:</em> Hyperparameter (units, learning rate, batch size) ditentukan secara manual (manual tuning).
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c_m2:
        st.markdown("""
        <div class="agri-card">
            <h4 style="color: #047857; margin-top: 0;">2. LSTM-PSO (Metaheuristic Optimization)</h4>
            <p style="font-size: 0.88rem; color: #475569;">
                Mengintegrasikan algoritma kecerdasan kawanan (<em>Particle Swarm Optimization</em>) untuk menemukan 
                kombinasi hyperparameter paling optimal secara otomatis:
            </p>
            <ul style="font-size: 0.85rem; color: #334155;">
                <li><strong>Ruang Pencarian:</strong> Unit sel LSTM, laju pembelajaran (Learning Rate), dan ukuran Batch.</li>
                <li><strong>Fungsi Fitness:</strong> Meminimalkan Mean Squared Error (MSE) pada subset validasi internal.</li>
                <li><strong>Personal & Global Best:</strong> Partikel bergerak menuju posisi terbaik individu dan kelompok.</li>
            </ul>
            <p style="font-size: 0.82rem; color: #64748b; margin-bottom: 0;">
                <em>Karakteristik:</em> Menghilangkan subjektivitas trial-and-error dalam pencarian arsitektur optimal.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab2:
    st.markdown("""
    <div class="agri-card">
        <h4 style="color: #065f46; margin-top: 0;">Penerapan Prinsip Deret Waktu yang Benar</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <h5 style="color: #0f172a; margin-bottom: 4px;">1. Pembagian Kronologis (Bukan Acak)</h5>
                <p style="font-size: 0.85rem; color: #475569;">
                    Data deret waktu tidak boleh diacak (<code>shuffle=False</code>). Bagian awal dijadikan data latih 
                    (misal 80%), dan bagian akhir masa depan dijadikan data uji (20%) untuk mensimulasikan kondisi peramalan nyata.
                </p>
            </div>
            <div>
                <h5 style="color: #0f172a; margin-bottom: 4px;">2. Normalisasi Tanpa Kebocoran (Data Leakage)</h5>
                <p style="font-size: 0.85rem; color: #475569;">
                    Objek scaler (MinMaxScaler / StandardScaler) hanya di-<em>fit</em> pada subset data latih. 
                    Data uji hanya di-<em>transform</em> menggunakan parameter statistik latih agar tidak terjadi kebocoran informasi masa depan.
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.markdown("""
    <div class="agri-card">
        <h4 style="color: #065f46; margin-top: 0;">Panduan Interpretasi Metrik Evaluasi</h4>
        <div style="overflow-x: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <thead>
                    <tr style="background: #f1f5f9; text-align: left; border-bottom: 2px solid #cbd5e1;">
                        <th style="padding: 8px 12px;">Metrik</th>
                        <th style="padding: 8px 12px;">Definisi</th>
                        <th style="padding: 8px 12px;">Interpretasi Nilai</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 12px; font-weight: 700;">MAE</td>
                        <td style="padding: 8px 12px;">Mean Absolute Error (rata-rata selisih absolut)</td>
                        <td style="padding: 8px 12px;">Semakin kecil semakin baik. Dalam satuan ton hasil panen.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 12px; font-weight: 700;">RMSE</td>
                        <td style="padding: 8px 12px;">Root Mean Squared Error (akar kuadrat selisih kuadrat)</td>
                        <td style="padding: 8px 12px;">Sensitif terhadap outlier atau lonjakan error besar.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #e2e8f0;">
                        <td style="padding: 8px 12px; font-weight: 700;">MAPE (%)</td>
                        <td style="padding: 8px 12px;">Mean Absolute Percentage Error (persentase kesalahan)</td>
                        <td style="padding: 8px 12px;">&lt; 10%: Sangat Akurat, 10–20%: Akurat, 20–50%: Layak.</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px 12px; font-weight: 700;">R² Score</td>
                        <td style="padding: 8px 12px;">Koefisien Determinasi (variansi target terjelaskan)</td>
                        <td style="padding: 8px 12px;">Mendekati 1.0 = Sangat Baik. Nilai negatif = Prediksi lebih buruk dari rata-rata.</td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>
    """, unsafe_allow_html=True)
