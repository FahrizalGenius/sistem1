import streamlit as st
from utils.data import find_kabupaten_column, get_kabupaten_list, infer_data_frequency
from utils.state import can_open, invalidate_from, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(4)
stage_header("4. Pengaturan Data", "Konfigurasi deret waktu: atur wilayah kabupaten, time step sequence, dan penentuan jumlah data latih/uji.")

df_raw_prep = st.session_state.preprocessed_df
if df_raw_prep is None:
    st.warning("Hasil preprocessing belum tersedia.")
    st.stop()

# --- VERIFIKASI / SELEKSI KABUPATEN PADA TAHAP 4 ---
clean_df_source = st.session_state.get("clean_df")
kab_col = st.session_state.get("kabupaten_col") or find_kabupaten_column(clean_df_source) or find_kabupaten_column(df_raw_prep)
kab_list = get_kabupaten_list(clean_df_source if clean_df_source is not None else df_raw_prep, kab_col) if kab_col else []

curr_selected_kab = st.session_state.get("selected_kabupaten", "Seluruh Data (Semua Kabupaten)")

if kab_col and kab_list:
    st.markdown("### 🌾 Wilayah / Kabupaten yang Diproses")
    kab_options = ["Seluruh Data (Semua Kabupaten)"] + kab_list
    kab_idx = kab_options.index(curr_selected_kab) if curr_selected_kab in kab_options else 0
    
    selected_kab = st.selectbox(
        f"Pilih Kabupaten untuk Pelatihan & Prediksi (Kolom: '{kab_col}'):",
        options=kab_options,
        index=kab_idx,
        help="Anda dapat mengganti kabupaten di sini jika ingin melatih model pada kabupaten tertentu."
    )
    
    if selected_kab != curr_selected_kab:
        st.session_state.selected_kabupaten = selected_kab
        if clean_df_source is not None and kab_col in clean_df_source.columns:
            if selected_kab != "Seluruh Data (Semua Kabupaten)":
                df = clean_df_source[clean_df_source[kab_col].astype(str) == selected_kab].copy().reset_index(drop=True)
            else:
                df = clean_df_source.copy()
            st.session_state.preprocessed_df = df
        curr_selected_kab = selected_kab
    
    # Filter data lokal untuk tampilan
    if curr_selected_kab != "Seluruh Data (Semua Kabupaten)" and kab_col in df_raw_prep.columns:
        df = df_raw_prep[df_raw_prep[kab_col].astype(str) == curr_selected_kab].copy().reset_index(drop=True)
    else:
        df = df_raw_prep.copy()

    st.info(f"🌾 Data Aktif: **{curr_selected_kab}** (Tersedia **{len(df):,} baris** observasi).")
else:
    df = df_raw_prep.copy()
    curr_selected_kab = "Seluruh Data (Semua Kabupaten)"

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# --- OTOMATISASI DETEKSI FREKUENSI DERET WAKTU ---
det_freq, det_rsn, rec_ts = infer_data_frequency(df, date_col=st.session_state.get("date_col"))
st.session_state.detected_frequency = det_freq
st.session_state.frequency_reason = det_rsn

st.markdown(f"""
<div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
    <span style="font-weight: 700; color: #166534;">🌾 Deteksi Otomatis Frekuensi Data ({curr_selected_kab}):</span>
    <span style="color: #15803d; margin-left: 6px;"><strong>{det_freq}</strong> — {det_rsn}.</span>
</div>
""", unsafe_allow_html=True)

col_config1, col_config2 = st.columns(2)

with col_config1:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">1. Frekuensi Observasi & Windowing Sequence (Time Step)</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Sistem otomatis mengenali frekuensi data. Anda dapat menyesuaikan time step sesuai kebutuhan.</p>
    </div>
    """, unsafe_allow_html=True)

    freq_options = ["Bulanan (Monthly)", "Tahunan (Yearly)", "Musim Tanam / Triwulan", "Mingguan", "Harian", "Ikuti urutan data"]
    curr_freq = st.session_state.get("frequency", det_freq)
    freq_idx = freq_options.index(curr_freq) if curr_freq in freq_options else (freq_options.index(det_freq) if det_freq in freq_options else 0)

    frequency = st.selectbox(
        "Frekuensi Observasi Data (Otomatis Terbaca):",
        options=freq_options,
        index=freq_idx,
        help="Sistem otomatis membaca data bulanan atau tahunan berdasarkan struktur data."
    )

    # Sesuaikan pilihan time step berdasarkan frekuensi observasi
    if "Tahunan" in frequency:
        time_step_options = [1, 2, 3, 4, 5]
        default_ts = min(rec_ts, 2)
    elif "Musim" in frequency:
        time_step_options = [1, 2, 3, 4, 6]
        default_ts = 3
    else:
        time_step_options = [1, 2, 3, 4, 6, 8, 12, 18, 24]
        default_ts = 6 if 6 in time_step_options else 4

    curr_ts = int(st.session_state.get("time_step", default_ts))
    time_step = st.selectbox(
        "Pilih Time Step (Lag Window):",
        options=time_step_options,
        index=time_step_options.index(curr_ts) if curr_ts in time_step_options else (time_step_options.index(default_ts) if default_ts in time_step_options else 0),
        help="Jumlah langkah/periode historis yang digunakan untuk memproyeksikan 1 langkah masa depan."
    )

with col_config2:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">2. Penentuan Data yang Dilatih</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Tentukan berapa porsi data yang dijadikan bahan pelatihan (train) vs pengujian (test).</p>
    </div>
    """, unsafe_allow_html=True)

    mode_options = ["Berdasarkan Rasio Split (Persentase)", "Berdasarkan Jumlah Baris Latih Kustom"]
    curr_mode = st.session_state.get("train_count_mode", "Rasio Split (Persentase)")
    train_mode = st.radio("Metode Penentuan Data Latih:", mode_options, index=0 if curr_mode == mode_options[0] else 1)

    n_total = len(df)

    if train_mode == "Berdasarkan Rasio Split (Persentase)":
        split_label = st.selectbox(
            "Pilih proporsi split (Latih : Uji):",
            ["80 : 20 (Rekomendasi)", "70 : 30", "90 : 10", "75 : 25", "Kustom Persentase..."]
        )
        if split_label.startswith("Kustom"):
            train_percent = st.selectbox("Persentase data latih (%):", list(range(50, 96, 5)), index=6)
            split_ratio = train_percent / 100.0
        else:
            train_percent = int(split_label.split(":")[0].strip())
            split_ratio = train_percent / 100.0
        split_idx = int(n_total * split_ratio)
        n_train = split_idx
        n_test = n_total - split_idx
    else:
        # Mode jumlah baris latih kustom
        min_train = max(time_step + 2, 5)
        max_train = max(min_train + 1, n_total - 2)
        default_custom = min(int(n_total * 0.8), max_train)
        
        n_train = st.number_input(
            f"Jumlah baris data yang ingin dilatih (antara {min_train} s.d {max_train}):",
            min_value=min_train,
            max_value=max_train,
            value=default_custom,
            step=1,
            help=f"Dari total {n_total} baris, Anda melatih {n_train} baris awal, sisanya untuk data uji."
        )
        split_idx = int(n_train)
        n_test = n_total - split_idx
        split_ratio = float(split_idx / n_total) if n_total > 0 else 0.8
        train_percent = int(round(split_ratio * 100))

# Estimasi sampel sequence yang dihasilkan
est_train_seq = max(0, n_train - time_step)
est_test_seq = max(0, n_test)

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# --- VISUALISASI SPLIT DATA SECARA KRONOLOGIS ---
st.markdown("### Visualisasi Pembagian Linier Waktu")

st.markdown(f"""
<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; margin-bottom: 24px;">
    <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 700; margin-bottom: 8px;">
        <span style="color: #059669;">Data Latih: {train_percent}% ({n_train:,} baris)</span>
        <span style="color: #2563eb;">Data Uji: {100-train_percent}% ({n_test:,} baris)</span>
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
    st.metric("Total Observasi", f"{n_total:,} Baris", curr_selected_kab if curr_selected_kab != "Seluruh Data (Semua Kabupaten)" else "Semua Data")
with m2:
    st.metric("Panjang Input (Time Step)", f"{time_step} Langkah")
with m3:
    st.metric("Data Latih", f"{n_train:,} Baris", f"{est_train_seq:,} sequence")
with m4:
    st.metric("Data Uji", f"{n_test:,} Baris", f"{est_test_seq:,} sequence")

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

if st.button("Simpan Pengaturan Deret Waktu", type="primary", use_container_width=True):
    if n_total <= time_step + 2:
        st.error("Dataset terlalu kecil untuk nilai time step ini.")
    elif split_idx <= time_step or split_idx >= n_total:
        st.error("Jumlah data latih tidak mencukupi untuk membentuk sequence latih dan uji.")
    else:
        invalidate_from(4)
        st.session_state.time_step = int(time_step)
        st.session_state.split_ratio = float(split_ratio)
        st.session_state.frequency = frequency
        st.session_state.train_count_mode = train_mode
        st.session_state.custom_train_rows = int(n_train)
        st.session_state.preprocessed_df = df
        mark_complete(4)
        kab_label = f" untuk {curr_selected_kab}" if curr_selected_kab != "Seluruh Data (Semua Kabupaten)" else ""
        st.success(f"Pengaturan deret waktu disimpan{kab_label}: Time Step = {time_step}, Latih = {n_train} baris ({train_percent}%), Uji = {n_test} baris.")
        st.rerun()

render_step_navigation(4)
