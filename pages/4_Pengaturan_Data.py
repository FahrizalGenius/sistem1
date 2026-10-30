import streamlit as st
from utils.state import can_open, invalidate_from, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(4)
stage_header("4. Pengaturan Data", "Konfigurasi deret waktu: atur time step (sequence windowing) dan pembagian kronologis data latih/uji tanpa kebocoran data.")

df = st.session_state.preprocessed_df
if df is None:
    st.warning("Hasil preprocessing belum tersedia.")
    st.stop()

date_col = st.session_state.get("date_col")
if date_col:
    st.markdown(f"""
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #166534;">📅 Kolom Waktu Terverifikasi:</span>
        <span style="color: #15803d; margin-left: 6px;"><strong>{date_col}</strong> (Data telah terurut secara kronologis dari waktu paling awal ke terbaru).</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("ℹ️ Kolom tanggal tidak dipilih spesifik; sistem menggunakan urutan baris asli berkas sebagai urutan waktu kronologis.")

col_config1, col_config2 = st.columns(2)

with col_config1:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">⏱️ 1. Windowing Sequence (Time Step)</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Jumlah periode masa lampau yang dijadikan input untuk memprediksi 1 periode berikutnya.</p>
    </div>
    """, unsafe_allow_html=True)

    curr_ts = int(st.session_state.get("time_step", 6))
    time_step_options = [1, 2, 3, 4, 6, 8, 12, 18, 24]
    time_step = st.selectbox(
        "Pilih Time Step (Lag Window):",
        options=time_step_options,
        index=time_step_options.index(curr_ts) if curr_ts in time_step_options else 4,
        help="Contoh: Time Step 6 pada data bulanan berarti model melihat 6 bulan terakhir untuk memprediksi panen bulan ke-7."
    )

    frequency = st.selectbox(
        "Frekuensi Observasi Data (Label Dokumentasi):",
        ["Ikuti urutan data", "Bulanan (Monthly)", "Musim Tanam / Triwulan", "Mingguan", "Harian", "Tahunan"],
        index=1 if "Bulan" in str(st.session_state.get("source_name", "")) else 0
    )

with col_config2:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">✂️ 2. Rasio Pembagian Data Latih & Uji</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Pembagian dilakukan berurutan secara kronologis (tanpa shuffle) untuk mencegah data leakage.</p>
    </div>
    """, unsafe_allow_html=True)

    split_label = st.selectbox("Pilih proporsi split (Latih : Uji):", ["80 : 20 (Rekomendasi)", "70 : 30", "90 : 10", "75 : 25", "Kustom..."])
    if split_label.startswith("Kustom"):
        train_percent = st.slider("Persentase data latih (%):", 50, 95, 80)
        split_ratio = train_percent / 100.0
    else:
        train_percent = int(split_label.split(":")[0].strip())
        split_ratio = train_percent / 100.0

n_total = len(df)
split_idx = int(n_total * split_ratio)
n_train = split_idx
n_test = n_total - split_idx

# Estimasi sampel sequence yang dihasilkan
est_train_seq = max(0, n_train - time_step)
est_test_seq = max(0, n_test)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# --- VISUALISASI SPLIT DATA SECARA KRONOLOGIS ---
st.markdown("### 📊 Visualisasi Pembagian Linier Waktu")

st.markdown(f"""
<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 24px;">
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 700; margin-bottom: 8px;">
        <span style="color: #059669;">🟢 Data Latih: {train_percent}% ({n_train:,} baris)</span>
        <span style="color: #2563eb;">🔵 Data Uji: {100-train_percent}% ({n_test:,} baris)</span>
    </div>
    <div style="display: flex; height: 26px; border-radius: 8px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);">
        <div style="width: {train_percent}%; background: linear-gradient(90deg, #059669, #10b981); display: flex; align-items: center; justify-content: center; color: white; font-size: 0.76rem; font-weight: 700;">
            Data Latih (Train)
        </div>
        <div style="width: {100-train_percent}%; background: linear-gradient(90deg, #3b82f6, #60a5fa); display: flex; align-items: center; justify-content: center; color: white; font-size: 0.76rem; font-weight: 700;">
            Data Uji (Test)
        </div>
    </div>
    <div style="display: flex; justify-content: space-between; font-size: 0.76rem; color: #64748b; margin-top: 6px;">
        <span>Observasi Awal [Index 0]</span>
        <span>Titik Potong [Index {split_idx}]</span>
        <span>Observasi Terakhir [Index {n_total-1}]</span>
    </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Observasi", f"{n_total:,} Baris")
with m2:
    st.metric("Panjang Input (Time Step)", f"{time_step} Langkah")
with m3:
    st.metric("Sequence Latih", f"{est_train_seq:,} Sampel")
with m4:
    st.metric("Sequence Uji", f"{est_test_seq:,} Sampel")

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

if st.button("🚀 Simpan Pengaturan Deret Waktu", type="primary", use_container_width=True):
    if n_total <= time_step + 2:
        st.error("Dataset terlalu kecil untuk nilai time step ini.")
    elif split_idx <= time_step or split_idx >= n_total:
        st.error("Rasio pembagian data tidak mencukupi untuk membentuk sequence latih dan uji.")
    else:
        invalidate_from(4)
        st.session_state.time_step = int(time_step)
        st.session_state.split_ratio = float(split_ratio)
        st.session_state.frequency = frequency
        mark_complete(4)
        st.success(f"✅ Pengaturan deret waktu disimpan: Time Step = {time_step}, Split = {train_percent}:{100-train_percent}.")
        st.rerun()

render_step_navigation(4)
