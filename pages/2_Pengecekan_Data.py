import streamlit as st
import pandas as pd
from utils.data import missing_summary
from utils.state import can_open, invalidate_from, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(2)
stage_header("2. Pengecekan Data", "Audit kualitas dataset: deteksi missing value, identifikasi duplikasi, dan tetapkan strategi pembersihan.")

df = st.session_state.raw_df
if df is None:
    st.warning("Belum ada dataset yang dimuat.")
    st.stop()

# --- HEALTH METRICS ---
total_cells = df.shape[0] * df.shape[1]
total_missing = int(df.isna().sum().sum())
dup_count = int(df.duplicated().sum())
completeness_pct = round(((total_cells - total_missing) / total_cells) * 100, 2) if total_cells > 0 else 100

st.markdown("### 🩺 Skor Kualitas & Integritas Data")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Tingkat Kelengkapan", f"{completeness_pct}%", "Data utuh" if total_missing == 0 else f"{total_missing} nilai kosong")
with c2:
    st.metric("Total Missing Value", f"{total_missing:,}", "0 diharapkan" if total_missing == 0 else "Perlu ditangani")
with c3:
    st.metric("Baris Duplikat", f"{dup_count:,}", "Bersih" if dup_count == 0 else "Ada duplikasi")
with c4:
    st.metric("Dimensi Data Mentah", f"{df.shape[0]:,} Baris", f"{df.shape[1]} Kolom")

st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

# --- TABEL RINGKASAN PER KOLOM ---
st.markdown("### 📋 Audit Kualitas Per Kolom")
summary = missing_summary(df)

# Highlight styling jika ada missing value
def highlight_missing(row):
    if row["Missing value"] > 0:
        return ["background-color: #fef2f2; color: #991b1b; font-weight: 600;"] * len(row)
    return [""] * len(row)

st.dataframe(summary.style.apply(highlight_missing, axis=1), use_container_width=True)

if dup_count > 0:
    with st.expander(f"⚠️ Ditemukan {dup_count} Baris Duplikat — Klik untuk Meninjau", expanded=False):
        st.dataframe(df[df.duplicated(keep=False)], use_container_width=True)

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# --- STRATEGI PENANGANAN DATA ---
st.markdown("### 🛠️ Pengaturan Pembersihan Data")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 10px 0; color: #0f172a; font-size: 0.95rem;">1. Penanganan Missing Value</h5>
    </div>
    """, unsafe_allow_html=True)
    missing_action = st.radio(
        "Pilih tindakan untuk missing value:",
        [
            "Biarkan untuk ditangani pada preprocessing",
            "Hapus baris yang memiliki missing value",
            "Isi dengan mean (kolom numerik)",
            "Isi dengan median (kolom numerik)",
            "Forward fill berdasarkan urutan baris",
        ],
        index=0,
        label_visibility="collapsed"
    )

with col_right:
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 18px; margin-bottom: 16px;">
        <h5 style="margin: 0 0 10px 0; color: #0f172a; font-size: 0.95rem;">2. Penanganan Baris Duplikat</h5>
    </div>
    """, unsafe_allow_html=True)
    duplicate_action = st.radio(
        "Pilih tindakan untuk duplikat:",
        ["Pertahankan", "Hapus baris duplikat"],
        index=1 if dup_count > 0 else 0,
        label_visibility="collapsed"
    )

st.caption("ℹ️ Data mentah asli tetap tersimpan secara aman di session state; pembersihan diterapkan pada dataset kerja.")

if st.button("🚀 Terapkan Pembersihan & Simpan Hasil", type="primary", use_container_width=True):
    cleaned = df.copy()
    if duplicate_action == "Hapus baris duplikat":
        cleaned = cleaned.drop_duplicates()
    if missing_action == "Hapus baris yang memiliki missing value":
        cleaned = cleaned.dropna()
    elif missing_action == "Isi dengan mean (kolom numerik)":
        for col in cleaned.select_dtypes(include="number").columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
    elif missing_action == "Isi dengan median (kolom numerik)":
        for col in cleaned.select_dtypes(include="number").columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
    elif missing_action == "Forward fill berdasarkan urutan baris":
        cleaned = cleaned.ffill()

    invalidate_from(2)
    st.session_state.clean_df = cleaned
    st.session_state.quality_checked = True
    st.session_state.missing_action = missing_action
    st.session_state.duplicate_action = duplicate_action
    mark_complete(2)
    st.success(f"✅ Pembersihan berhasil diterapkan! Dataset kerja sekarang memiliki **{len(cleaned):,} baris**.")
    st.rerun()

clean_df = st.session_state.get("clean_df")
if clean_df is not None:
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.info(f"✨ **Dataset Bersih Tersimpan:** {len(clean_df):,} baris × {clean_df.shape[1]} kolom siap diproses.")
    with st.expander("👁️ Pratinjau Dataset Hasil Pembersihan (30 Baris Pertama)"):
        st.dataframe(clean_df.head(30), use_container_width=True)

render_step_navigation(2)
