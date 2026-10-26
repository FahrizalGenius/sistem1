import streamlit as st
from utils.state import can_open, invalidate_from, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(5)
stage_header("5. Pemilihan Model", "Tentukan model peramalan: arsitektur LSTM Baseline manual atau optimasi cerdas LSTM-Particle Swarm Optimization (LSTM-PSO).")

# --- PEMILIHAN METODE MODEL ---
curr_choice = st.session_state.get("model_choice", "LSTM Baseline")
model_choice = st.radio(
    "Pilih Arsitektur Model:",
    ["LSTM Baseline", "LSTM-PSO"],
    horizontal=True,
    index=0 if curr_choice == "LSTM Baseline" else 1,
    help="LSTM Baseline menggunakan nilai tetap yang Anda atur manual. LSTM-PSO mengoptimasi units, learning rate, dan batch size otomatis."
)

if model_choice == "LSTM Baseline":
    st.markdown("""
    <div style="background: #f0fdf4; border: 1.5px solid #a7f3d0; border-radius: 14px; padding: 18px 24px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-size: 1.4rem;">🧠</span>
            <h4 style="margin: 0; color: #065f46; font-size: 1.05rem;">Model: LSTM Baseline (Standard)</h4>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.5;">
            Seluruh hyperparameter (jumlah neuron LSTM, learning rate, ukuran batch) diatur secara manual oleh peneliti untuk dijadikan tolok ukur (baseline) perbandingan kinerja.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="background: #eff6ff; border: 1.5px solid #bfdbfe; border-radius: 14px; padding: 18px 24px; margin-bottom: 24px;">
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px;">
            <span style="font-size: 1.4rem;">🐝</span>
            <h4 style="margin: 0; color: #1e40af; font-size: 1.05rem;">Model: LSTM-PSO (Optimasi Metaheuristik Swarm)</h4>
        </div>
        <p style="margin: 0; font-size: 0.88rem; color: #334155; line-height: 1.5;">
            Algoritma Particle Swarm Optimization (PSO) akan mensimulasikan sekumpulan partikel untuk mengeksplorasi kombinasi hyperparameter terbaik secara adaptif guna meminimalkan MSE validasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- HYPERPARAMETER CORE ---
st.markdown("### 🎛️ Konfigurasi Arsitektur Jaringan")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <h5 style="margin: 0 0 4px 0; color: #0f172a;">Arsitektur Sel</h5>
        <span style="font-size: 0.75rem; color: #64748b;">Neuron & Kedalaman Lapisan</span>
    </div>
    """, unsafe_allow_html=True)
    units = st.select_slider("Jumlah Unit LSTM (Hidden Neurons):", options=[16, 24, 32, 48, 64, 96, 128], value=64)
    layers = st.selectbox("Jumlah Lapisan (Layers) LSTM:", [1, 2, 3], index=0)

with c2:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <h5 style="margin: 0 0 4px 0; color: #0f172a;">Optimasi & Regularisasi</h5>
        <span style="font-size: 0.75rem; color: #64748b;">Laju Belajar & Pencegah Overfit</span>
    </div>
    """, unsafe_allow_html=True)
    learning_rate = st.selectbox("Learning Rate (Adam Optimizer):", [0.0001, 0.0005, 0.001, 0.005, 0.01], index=2)
    dropout = st.slider("Tingkat Dropout (0 s.d 0.5):", 0.0, 0.5, 0.1, 0.05)

with c3:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 12px;">
        <h5 style="margin: 0 0 4px 0; color: #0f172a;">Pelatihan Batch</h5>
        <span style="font-size: 0.75rem; color: #64748b;">Ukuran Iterasi & Epochs</span>
    </div>
    """, unsafe_allow_html=True)
    batch_size = st.selectbox("Batch Size:", [8, 16, 32, 64], index=1)
    epochs = st.slider("Maksimum Epoch Pelatihan:", 10, 200, 50, 5)

validation_split = st.slider("Proporsi Validasi Internal dari Data Latih (Chronological Tail):", 0.0, 0.3, 0.15, 0.05)
st.caption("ℹ️ Validasi menggunakan potongan akhir data latih secara kronologis (tanpa shuffle) untuk early stopping & evaluasi fitness.")

pso_config = {}
if model_choice == "LSTM-PSO":
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🐝 Konfigurasi Particle Swarm Optimization (PSO)")

    p1, p2, p3 = st.columns(3)
    with p1:
        particles = st.slider("Jumlah Partikel (Swarm Size):", 3, 20, 5, help="Jumlah kombinasi kandidat per iterasi pencarian.")
    with p2:
        iterations = st.slider("Jumlah Iterasi PSO (Generasi):", 1, 20, 5, help="Siklus pergerakan swarm menuju gbest.")
    with p3:
        epochs_search = st.slider("Epoch Evaluasi Tiap Partikel:", 3, 30, 8, help="Epoch ringkas untuk mengevaluasi loss kandidat.")

    col_u, col_lr = st.columns(2)
    with col_u:
        units_min, units_max = st.select_slider("Rentang Pencarian Unit LSTM:", options=[16, 24, 32, 48, 64, 96, 128], value=(16, 96))
    with col_lr:
        lr_min, lr_max = st.select_slider("Rentang Pencarian Learning Rate:", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=(0.0001, 0.005))

    pso_config = {
        "particles": particles,
        "iterations": iterations,
        "epochs_search": epochs_search,
        "units_min": units_min,
        "units_max": units_max,
        "lr_min": lr_min,
        "lr_max": lr_max,
        "batch_options": [8, 16, 32, 64],
    }

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

if st.button("🚀 Simpan Konfigurasi Model", type="primary", use_container_width=True):
    invalidate_from(5)
    st.session_state.model_choice = model_choice
    st.session_state.model_params = {
        "units": units,
        "layers": layers,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "batch_size": batch_size,
        "epochs": epochs,
        "validation_split": validation_split,
        "pso": pso_config,
    }
    mark_complete(5)
    st.success(f"✅ Konfigurasi **{model_choice}** berhasil disimpan!")
    st.rerun()

render_step_navigation(5)
