import streamlit as st
import pandas as pd
import numpy as np
from utils.modeling import calculate_metrics
from utils.state import can_open, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(7)
stage_header("7. Evaluasi", "Evaluasi komprehensif performa model peramalan panen padi pada data uji yang tidak pernah dilihat saat pelatihan.")

y_true = st.session_state.get("y_test")
y_pred = st.session_state.get("y_pred")
df = st.session_state.get("preprocessed_df")

if y_true is None or y_pred is None:
    st.warning("Hasil peramalan belum tersedia. Silakan jalankan Tahap 6 terlebih dahulu.")
    st.stop()

# Hitung metrik evaluasi
metrics = calculate_metrics(y_true, y_pred)
st.session_state.eval_metrics = metrics
mark_complete(7)

# Analisis Kualitatif Model
r2_val = metrics.get("R²", 0.0)
mape_val = metrics.get("MAPE (%)", 0.0)

if r2_val >= 0.85 and mape_val <= 10.0:
    eval_status = "Sangat Akurat & Handal 🌟"
    status_bg = "#d1fae5"
    status_color = "#065f46"
elif r2_val >= 0.70 and mape_val <= 20.0:
    eval_status = "Kinerja Baik & Layak Pakai 👍"
    status_bg = "#eff6ff"
    status_color = "#1e40af"
elif r2_val > 0.40:
    eval_status = "Kinerja Cukup (Dapat Dioptimasi Lagi) ⚠️"
    status_bg = "#fef3c7"
    status_color = "#92400e"
else:
    eval_status = "Performa Perlu Peningkatan ❌"
    status_bg = "#fee2e2"
    status_color = "#991b1b"

# --- HERO METRICS HEADER ---
st.markdown(f"""
<div style="background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 16px; padding: 20px 24px; margin-bottom: 24px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; box-shadow: 0 4px 14px rgba(0,0,0,0.03);">
    <div>
        <span style="font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px;">Model yang Dievaluasi:</span>
        <h3 style="margin: 2px 0 0 0; color: #064e3b; font-size: 1.3rem;">🌾 {st.session_state.get('model_choice', 'LSTM')}</h3>
    </div>
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 0.82rem; color: #475569;">Penilaian Akurasi:</span>
        <span style="background: {status_bg}; color: {status_color}; font-weight: 700; padding: 6px 14px; border-radius: 20px; font-size: 0.82rem;">
            {eval_status}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# 5 Kartu Metrik Utama
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    mae_val = metrics["MAE"]
    st.metric("📏 MAE", f"{mae_val:,.2f}" if np.isfinite(mae_val) else "—", "Error rata-rata absolut")

with col2:
    rmse_val = metrics["RMSE"]
    st.metric("📐 RMSE", f"{rmse_val:,.2f}" if np.isfinite(rmse_val) else "—", "Akar error kuadrat")

with col3:
    mse_val = metrics["MSE"]
    st.metric("📊 MSE", f"{mse_val:,.1f}" if np.isfinite(mse_val) else "—", "Mean squared error")

with col4:
    mape_str = f"{mape_val:.2f}%" if np.isfinite(mape_val) else "—"
    st.metric("🎯 MAPE", mape_str, "Persentase error")

with col5:
    r2_str = f"{r2_val:.4f}" if np.isfinite(r2_val) else "—"
    st.metric("🌟 R² Score", r2_str, "Koefisien determinasi")

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# --- PERSIAPAN DATAFRAME HASIL ---
positions = st.session_state.get("test_positions", range(len(y_true)))
date_col = st.session_state.get("date_col")

if date_col and df is not None and date_col in df.columns:
    x_axis = df.iloc[positions][date_col].dt.strftime("%Y-%m-%d") if hasattr(df.iloc[positions][date_col], "dt") else df.iloc[positions][date_col].astype(str)
    x_axis = x_axis.reset_index(drop=True)
else:
    x_axis = [f"Periode {i+1}" for i in range(len(y_true))]

result = pd.DataFrame({
    "Periode": x_axis,
    "Nilai Aktual": np.asarray(y_true, dtype=float),
    "Nilai Prediksi": np.asarray(y_pred, dtype=float),
    "Error (Selisih)": np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float),
    "Error Absolut (%)": np.abs((np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)) / np.where(y_true != 0, y_true, 1)) * 100
})

# --- GRAFIK INTERAKTIF ---
st.markdown("### 📈 Visualisasi Aktual vs Prediksi Model")

tab_plot1, tab_plot2 = st.tabs(["📉 Grafik Deret Waktu (Aktual vs Prediksi)", "📊 Analisis Residual Error"])

with tab_plot1:
    chart_df = result.set_index("Periode")[["Nilai Aktual", "Nilai Prediksi"]]
    st.line_chart(chart_df, color=["#059669", "#2563eb"])
    st.caption("🟢 Hijau: Nilai Aktual Panen | 🔵 Biru: Nilai Prediksi Model Deep Learning.")

with tab_plot2:
    error_df = result.set_index("Periode")[["Error (Selisih)"]]
    st.bar_chart(error_df, color=["#f59e0b"])
    st.caption("Grafik residual menunjukkan sebaran penyimpangan nilai prediksi terhadap aktual (Residual = Aktual - Prediksi).")

st.markdown("<div style='margin-bottom: 28px;'></div>", unsafe_allow_html=True)

# --- TABEL RINCIAN & PUSAT EKSPOR ---
st.markdown("### 📋 Tabel Rincian Prediksi Data Uji")
st.dataframe(result.style.format({
    "Nilai Aktual": "{:,.2f}",
    "Nilai Prediksi": "{:,.2f}",
    "Error (Selisih)": "{:,.2f}",
    "Error Absolut (%)": "{:.2f}%"
}), use_container_width=True)

st.markdown("<div style='margin-bottom: 16px;'></div>", unsafe_allow_html=True)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    st.download_button(
        "📥 Unduh Hasil Lengkap Aktual & Prediksi (CSV)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="hasil_lengkap_evaluasi_prediksi.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_dl2:
    st.download_button(
        "📥 Unduh Ringkasan Metrik Evaluasi (CSV)",
        data=pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8"),
        file_name="ringkasan_metrik_evaluasi.csv",
        mime="text/csv",
        use_container_width=True,
    )

render_step_navigation(7)
