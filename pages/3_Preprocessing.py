import streamlit as st
import pandas as pd
import numpy as np
from utils.data import infer_datetime_columns
from utils.state import can_open, invalidate_from, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(3)
stage_header("3. Preprocessing", "Konfigurasi variabel masukan, tetapkan target peramalan hasil panen, urutkan dimensi waktu, dan terapkan normalisasi data.")

df = st.session_state.clean_df
if df is None:
    st.warning("Data hasil pembersihan belum tersedia.")
    st.stop()

st.markdown(f"""
<div style="background: #ffffff; border: 1px solid #d1fae5; border-radius: 12px; padding: 14px 20px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between;">
    <div>
        <span style="font-weight: 700; color: #065f46;">📊 Dataset Siap Diproses:</span>
        <span style="color: #475569; margin-left: 8px;"><strong>{df.shape[0]:,}</strong> baris observasi × <strong>{df.shape[1]}</strong> kolom</span>
    </div>
    <span class="badge-pill badge-success">Status: Bersih</span>
</div>
""", unsafe_allow_html=True)

# --- KONFIGURASI PREPROCESSING ---
st.markdown("### ⚙️ Pengaturan Fitur & Target")

c_left, c_right = st.columns(2)

with c_left:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">📅 1. Dimensi Waktu & Tanggal</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Kolom tanggal digunakan untuk memastikan baris terurut secara kronologis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    date_candidates = infer_datetime_columns(df)
    date_options = ["Tidak ada (Ikuti urutan baris)"] + list(df.columns)
    default_date_idx = (date_options.index(date_candidates[0])) if date_candidates and date_candidates[0] in date_options else 0
    date_col_choice = st.selectbox("Pilih kolom tanggal/waktu:", date_options, index=default_date_idx)
    date_col = None if date_col_choice.startswith("Tidak ada") else date_col_choice

    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 16px; margin-top: 20px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">🎯 2. Variabel Target (Output)</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Nilai yang akan diproyeksikan/diramalkan oleh model LSTM (misal: Produksi Panen).</p>
    </div>
    """, unsafe_allow_html=True)
    
    numeric_cols = list(df.select_dtypes(include=["number"]).columns)
    default_target_idx = 0
    # Auto-detect target jika ada kata produksi/panen/yield
    for idx, col in enumerate(numeric_cols):
        if any(term in col.lower() for term in ["produksi", "panen", "yield", "target", "hasil"]):
            default_target_idx = idx
            break
            
    target_col = st.selectbox("Pilih variabel target:", numeric_cols if numeric_cols else list(df.columns), index=default_target_idx)

with c_right:
    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">📊 3. Fitur Masukan (Input Variables)</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Pilih variabel masukan. <em>Catatan: Target harus disertakan agar sequence historisnya terbaca.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Target harus termasuk di multiselect
    initial_features = numeric_cols if numeric_cols else list(df.columns)
    feature_cols = st.multiselect("Pilih fitur masukan:", list(df.columns), default=initial_features)

    st.markdown("""
    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 16px; margin-bottom: 16px; margin-top: 20px;">
        <h5 style="margin: 0 0 8px 0; color: #0f172a;">⚖️ 4. Metode Normalisasi Data</h5>
        <p style="font-size: 0.8rem; color: #64748b; margin-bottom: 12px;">Penskalaan nilai fitur agar konvergen optimal pada fungsi aktivasi LSTM.</p>
    </div>
    """, unsafe_allow_html=True)
    
    scaler_name = st.radio(
        "Pilih algoritma normalisasi:",
        ["Min-Max (Rentang [0, 1] — Disarankan untuk LSTM)", "StandardScaler (Mean=0, Std=1)", "Tidak ada (Gunakan data asli)"],
        index=0,
    )
    scaler_key = "Min-Max" if "Min-Max" in scaler_name else ("StandardScaler" if "StandardScaler" in scaler_name else "Tidak ada")

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

if st.button("🚀 Terapkan Preprocessing & Simpan Konfigurasi", type="primary", use_container_width=True):
    if not feature_cols:
        st.error("Pilih minimal satu fitur masukan.")
    elif target_col not in feature_cols:
        st.error(f"Variabel target ('{target_col}') harus termasuk dalam daftar fitur masukan untuk membentuk deret waktu historis!")
    else:
        out = df.copy()
        if date_col:
            out[date_col] = pd.to_datetime(out[date_col], errors="coerce", dayfirst=True)
            if out[date_col].isna().any():
                st.error("Kolom tanggal memiliki nilai yang tidak dapat dibaca. Silakan periksa format tanggal.")
                st.stop()
            out = out.sort_values(date_col).reset_index(drop=True)

        for col in feature_cols:
            if col != date_col:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.replace([np.inf, -np.inf], np.nan)
        if out[feature_cols].isna().any().any():
            st.error("Masih terdapat nilai kosong (missing) setelah konversi numerik. Kembali ke Tahap 2 untuk pembersihan.")
        else:
            invalidate_from(3)
            st.session_state.preprocessed_df = out
            st.session_state.feature_cols = feature_cols
            st.session_state.target_col = target_col
            st.session_state.date_col = date_col
            st.session_state.scaler_name = scaler_key
            mark_complete(3)
            st.success("✅ Preprocessing berhasil diterapkan dan data telah diurutkan menurut waktu!")
            st.rerun()

# --- STATISTIK DESKRIPTIF SETELAH PREPROCESSING ---
preprocessed = st.session_state.get("preprocessed_df")
if preprocessed is not None and st.session_state.get("feature_cols"):
    sel_feats = st.session_state.feature_cols
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📈 Ringkasan Statistik Deskriptif Fitur Terpilih")
    
    desc_df = preprocessed[sel_feats].describe().T[["count", "mean", "std", "min", "50%", "max"]]
    desc_df.columns = ["Jumlah", "Rata-rata (Mean)", "Standar Deviasi", "Nilai Min", "Median (50%)", "Nilai Max"]
    st.dataframe(desc_df.style.format("{:,.2f}"), use_container_width=True)

    with st.expander("👁️ Pratinjau Data Hasil Preprocessing (30 Baris Pertama)"):
        st.dataframe(preprocessed[sel_feats].head(30), use_container_width=True)

render_step_navigation(3)
