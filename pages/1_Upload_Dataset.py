import streamlit as st
import pandas as pd
from utils.data import (
    read_uploaded_file,
    validate_dataset,
    generate_sample_paddy_data,
    find_kabupaten_column,
    get_kabupaten_list,
    infer_data_frequency
)
from utils.state import invalidate_from, mark_complete
from utils.ui import stage_header, render_step_navigation

stage_header("1. Upload Dataset", "Unggah data historis panen padi dalam format CSV atau Excel, atau gunakan dataset simulasi bawaan.")

tab_upload, tab_sample = st.tabs(["Unggah File Mandiri (CSV / Excel)", "🌾 Gunakan Dataset Demo Panen Padi"])

with tab_upload:
    st.markdown("""
    <div style="background: #ffffff; border: 1.5px dashed #10b981; border-radius: 14px; padding: 20px; margin-bottom: 20px; text-align: center;">
        <p style="margin: 0; color: #475569; font-size: 0.9rem;">
            Format didukung: <strong>.csv, .xlsx, .xls</strong>. Pastikan file memiliki baris berurutan secara kronologis menurut waktu.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded = st.file_uploader("Pilih file CSV atau Excel dari komputer Anda", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    
    sheet_name = None
    if uploaded:
        if uploaded.name.lower().endswith((".xlsx", ".xls")):
            try:
                sheets = pd.ExcelFile(uploaded).sheet_names
                if len(sheets) > 1:
                    sheet_name = st.selectbox("Pilih lembar kerja (sheet) Excel:", sheets)
            except Exception:
                pass
                
        if st.button("Baca & Validasi Dataset", type="primary", use_container_width=True):
            try:
                df = read_uploaded_file(uploaded, sheet_name)
                issues = validate_dataset(df)
                if issues:
                    for issue in issues:
                        st.error(issue)
                else:
                    invalidate_from(1)
                    st.session_state.raw_df = df.copy()
                    st.session_state.source_name = uploaded.name
                    st.session_state.quality_checked = False
                    st.session_state.clean_df = None
                    st.session_state.selected_kabupaten = "Seluruh Data (Semua Kabupaten)"
                    
                    kab_col = find_kabupaten_column(df)
                    st.session_state.kabupaten_col = kab_col
                    
                    freq_lbl, freq_rsn, rec_ts = infer_data_frequency(df)
                    st.session_state.detected_frequency = freq_lbl
                    st.session_state.frequency = freq_lbl
                    st.session_state.frequency_reason = freq_rsn
                    st.session_state.time_step = rec_ts
                    
                    mark_complete(1)
                    st.success(f"Dataset berhasil diverifikasi: **{df.shape[0]:,} baris** × **{df.shape[1]} kolom**.")
                    st.rerun()
            except Exception as exc:
                st.error(f"Gagal membaca dataset: {exc}")

with tab_sample:
    st.markdown("""
    <div class="agri-card">
        <h4 style="color: #065f46; margin-top: 0;">🌾 Dataset Simulasi Panen Padi (10 Tahun / 120 Bulan)</h4>
        <p style="font-size: 0.88rem; color: #475569;">
            Dataset sintetis berbasis tren agroklimat Indonesia yang mencakup siklus tanam musiman panen raya (Maret–Mei & September–Oktober).
        </p>
        <div style="font-size: 0.82rem; color: #334155; margin-bottom: 16px;">
            <strong>Variabel yang disertakan:</strong>
            <ul>
                <li><code>Tanggal</code>: Periode bulanan (2014-01 s.d. 2023-12)</li>
                <li><code>Luas_Panen_ha</code>: Luas lahan panen bulanan (hektar)</li>
                <li><code>Curah_Hujan_mm</code>: Akumulasi curah hujan bulanan (mm)</li>
                <li><code>Suhu_Rata2_C</code>: Suhu udara rata-rata (°C)</li>
                <li><code>Kelembaban_Persen</code>: Kelembaban relatif rata-rata (%)</li>
                <li><code>Produksi_Padi_Ton</code>: <em>Variabel Target</em> — Total hasil panen gabah/padi (Ton)</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Muat Dataset Simulasi Demo", type="primary", use_container_width=True):
        df_demo = generate_sample_paddy_data()
        invalidate_from(1)
        st.session_state.raw_df = df_demo
        st.session_state.source_name = "Dataset_Demo_Panen_Padi_120Bulan.csv"
        st.session_state.quality_checked = False
        st.session_state.clean_df = None
        st.session_state.selected_kabupaten = "Seluruh Data (Semua Kabupaten)"
        st.session_state.kabupaten_col = None
        
        freq_lbl, freq_rsn, rec_ts = infer_data_frequency(df_demo)
        st.session_state.detected_frequency = freq_lbl
        st.session_state.frequency = freq_lbl
        st.session_state.frequency_reason = freq_rsn
        st.session_state.time_step = rec_ts
        
        mark_complete(1)
        st.success("Dataset simulasi panen padi berhasil dimuat ke sistem!")
        st.rerun()

# --- TAMPILAN DATASET AKTIF ---
df = st.session_state.get("raw_df")
if df is not None:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    st.markdown("### Ringkasan & Pratinjau Data Aktif")
    
    kab_col = find_kabupaten_column(df)
    if kab_col:
        st.session_state.kabupaten_col = kab_col
        kabs = get_kabupaten_list(df, kab_col)
        st.info(f"🌾 Kolom Wilayah Terdeteksi: **{kab_col}** ({len(kabs)} kabupaten: {', '.join(kabs[:8])}{'...' if len(kabs) > 8 else ''}). Anda dapat melatih model untuk seluruh data atau memilih kabupaten tertentu pada tahap Preprocessing.")
    
    det_freq = st.session_state.get("detected_frequency")
    det_rsn = st.session_state.get("frequency_reason")
    if not det_freq:
        det_freq, det_rsn, rec_ts = infer_data_frequency(df)
        st.session_state.detected_frequency = det_freq
        st.session_state.frequency = det_freq
        st.session_state.frequency_reason = det_rsn
        st.session_state.time_step = rec_ts
    
    st.markdown(f"""
    <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;">
        <span style="font-weight: 700; color: #166534;">Frekuensi Deret Waktu Terdeteksi Otomatis:</span>
        <span style="color: #15803d; margin-left: 6px;"><strong>{det_freq}</strong> ({det_rsn}). Rekomendasi time step: {st.session_state.get('time_step', 6)} langkah.</span>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Baris Observasi", f"{len(df):,}")
    with c2:
        st.metric("Total Kolom Fitur", len(df.columns))
    with c3:
        st.metric("Kolom Numerik", len(df.select_dtypes(include="number").columns))
    with c4:
        st.metric("Nama Sumber File", (st.session_state.source_name or "Dataset")[:16])

    with st.expander("Lihat Pratinjau Data (50 Baris Pertama)", expanded=True):
        st.dataframe(df.head(50), use_container_width=True)

    render_step_navigation(1)
