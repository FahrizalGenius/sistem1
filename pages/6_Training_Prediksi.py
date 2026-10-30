import streamlit as st
import pandas as pd
import numpy as np
from utils.modeling import prepare_sequences, train_baseline, train_pso
from utils.state import can_open, invalidate_from, mark_complete
from utils.ui import stage_header, require_previous, render_step_navigation

require_previous(6)
stage_header("6. Training & Prediksi", "Eksekusi pelatihan Deep Learning, pantau kurva konvergensi loss & swarm fitness, serta hasilkan estimasi data uji.")

df = st.session_state.preprocessed_df
if df is None:
    st.warning("Data hasil preprocessing belum tersedia.")
    st.stop()

params = st.session_state.model_params
if not params:
    st.warning("Konfigurasi model belum disimpan. Silakan kembali ke Tahap 5.")
    st.stop()

# --- PRE-FLIGHT CHECKLIST ---
st.markdown("### 📋 Ringkasan Parameter Pra-Pelatihan")

p_model = st.session_state.model_choice
p_target = st.session_state.target_col
p_feats = st.session_state.feature_cols
p_ts = st.session_state.time_step
p_split = st.session_state.split_ratio

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Model Terpilih", p_model)
with c2:
    st.metric("Variabel Target", str(p_target)[:16])
with c3:
    st.metric("Panjang Sequence (Time Step)", f"{p_ts} Periode")
with c4:
    st.metric("Rasio Split (Latih / Uji)", f"{p_split:.0%} / {1-p_split:.0%}")

with st.expander("🔍 Rincian Hyperparameter & Fitur Masukan", expanded=False):
    st.write("**Fitur Masukan:**", ", ".join(p_feats))
    st.json({k: v for k, v in params.items() if k != "pso"})
    if p_model == "LSTM-PSO":
        st.write("**Konfigurasi PSO:**")
        st.json(params.get("pso", {}))

st.markdown("<div style='margin-bottom: 24px;'></div>", unsafe_allow_html=True)

# --- TOMBOL RUN TRAINING ---
if st.button("🚀 Mulai Pelatihan & Hasilkan Prediksi", type="primary", use_container_width=True):
    try:
        status_box = st.status("Sedang mempersiapkan data sequence & melatih arsitektur neural network...", expanded=True)
        with status_box:
            st.write("1. Membangun sequence chronologically dan fitting scaler pada data latih...")
            X_train, y_train, X_test, y_test, test_positions, scaler_x, scaler_y, work = prepare_sequences(
                df=df,
                feature_cols=p_feats,
                target_col=p_target,
                time_step=p_ts,
                split_ratio=p_split,
                scaler_name=st.session_state.get("scaler_name", "Min-Max"),
            )

            validation_split = float(params.get("validation_split", 0.15))

            if p_model == "LSTM-PSO":
                st.write("2. Menjalankan pencarian optimasi Particle Swarm Optimization (PSO)...")
                val_count = max(1, int(len(X_train) * max(validation_split, 0.15)))
                if len(X_train) <= val_count + 2:
                    raise ValueError("Jumlah data latih terlalu sedikit untuk validasi PSO. Kurangi time step atau ubah rasio split.")
                
                X_fit, y_fit = X_train[:-val_count], y_train[:-val_count]
                X_val, y_val = X_train[-val_count:], y_train[-val_count:]
                pso = params["pso"]
                
                model, history, best = train_pso(
                    X_fit, y_fit, X_val, y_val,
                    pso_particles=pso["particles"],
                    pso_iterations=pso["iterations"],
                    units_min=pso["units_min"],
                    units_max=pso["units_max"],
                    lr_min=pso["lr_min"],
                    lr_max=pso["lr_max"],
                    batch_options=pso["batch_options"],
                    epochs=pso["epochs_search"],
                    layers=params["layers"],
                    dropout=params["dropout"],
                    validation_split=0.0,
                )
                
                st.write(f"3. Melatih ulang arsitektur gbest terbaik: Units={best['units']}, LR={best['learning_rate']:.5f}, Batch={best['batch_size']}...")
                model, history = train_baseline(
                    X_train, y_train, best["units"], params["layers"], params["dropout"],
                    best["learning_rate"], best["batch_size"], params["epochs"],
                    params["validation_split"],
                )
                st.session_state.pso_history = best["pso_history"]
                st.session_state.best_pso = best
            else:
                st.write("2. Melatih model LSTM Baseline dengan parameter konstan...")
                model, history = train_baseline(
                    X_train, y_train, params["units"], params["layers"], params["dropout"],
                    params["learning_rate"], params["batch_size"], params["epochs"],
                    params["validation_split"],
                )
                st.session_state.pso_history = None
                st.session_state.best_pso = None

            st.write("4. Menjalankan inferensi prediksi pada data uji...")
            pred_scaled = model.predict(X_test, verbose=0).ravel()
            pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

            invalidate_from(6)
            st.session_state.trained_model = model
            st.session_state.scaler_x = scaler_x
            st.session_state.scaler_y = scaler_y
            st.session_state.y_test = y_test
            st.session_state.y_pred = pred
            st.session_state.test_positions = test_positions
            st.session_state.training_info = {
                "history": history,
                "X_train_shape": X_train.shape,
                "X_test_shape": X_test.shape,
                "target_col": p_target,
            }
            mark_complete(6)
            status_box.update(label="✅ Pelatihan & Proyeksi Selesai dengan Sukses!", state="complete", expanded=False)
        st.rerun()

    except Exception as exc:
        st.error(f"Pelatihan gagal: {exc}")
        st.info("Pastikan parameter sesuai dengan volume dataset dan TensorFlow berjalan dengan normal.")

# --- HASIL TRAINING AKTIF ---
if st.session_state.get("trained_model") is not None:
    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Analisis Proses Pelatihan")

    info = st.session_state.training_info
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Sequence Latih Dibuat", f"{info['X_train_shape'][0]:,} Sampel")
    with s2:
        st.metric("Sequence Uji Dibuat", f"{info['X_test_shape'][0]:,} Sampel")
    with s3:
        st.metric("Dimensi Input Tensor", f"({info['X_train_shape'][1]}, {info['X_train_shape'][2]})")

    # Tampilkan parameter terbaik jika PSO
    if st.session_state.model_choice == "LSTM-PSO" and st.session_state.get("best_pso"):
        best = st.session_state.best_pso
        st.markdown(f"""
        <div style="background: #eff6ff; border: 1.5px solid #93c5fd; border-radius: 12px; padding: 16px 20px; margin: 18px 0;">
            <div style="color: #1e40af; font-weight: 700; margin-bottom: 6px;">🏆 Hasil Optimasi Swarm PSO (Global Best):</div>
            <div style="display: flex; gap: 24px; font-size: 0.88rem; color: #1e3a8a;">
                <span><strong>Unit LSTM Terpilih:</strong> {best['units']}</span>
                <span><strong>Learning Rate Terpilih:</strong> {best['learning_rate']:.5f}</span>
                <span><strong>Batch Size Terpilih:</strong> {best['batch_size']}</span>
                <span><strong>Validation MSE:</strong> {best['best_validation_mse_scaled']:.6f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_loss, tab_pso_chart = st.tabs(["📉 Kurva Konvergensi Loss (MSE)", "🐝 Konvergensi Fitness PSO"])

    with tab_loss:
        history = info.get("history", {})
        if history and "loss" in history:
            hist_df = pd.DataFrame({
                "Epoch": range(1, len(history["loss"]) + 1),
                "Training Loss (MSE)": history["loss"]
            })
            if "val_loss" in history:
                hist_df["Validation Loss (MSE)"] = history["val_loss"]
            
            st.line_chart(hist_df.set_index("Epoch"), color=["#059669", "#ef4444"] if "val_loss" in history else ["#059669"])
            st.caption("Penurunan kurva loss menunjukkan proses konvergensi bobot model selama epoch pelatihan.")

    with tab_pso_chart:
        if st.session_state.model_choice == "LSTM-PSO" and st.session_state.get("pso_history"):
            pso_df = pd.DataFrame(st.session_state.pso_history).set_index("Iterasi")
            st.line_chart(pso_df, color=["#2563eb"])
            st.caption("Penurunan skor Best MSE per iterasi menandakan kawanan partikel berhasil menemukan solusi parameter yang lebih presisi.")
        else:
            st.info("Kurva fitness hanya aktif ketika model LSTM-PSO dipilih.")

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    st.download_button(
        "📥 Unduh Hasil Prediksi Mentah (CSV)",
        data=pd.DataFrame({"Aktual": st.session_state.y_test, "Prediksi": st.session_state.y_pred}).to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_padi.csv",
        mime="text/csv",
    )

    # ====== PREDIKSI TAHUN 2026 ======
    st.markdown("---")
    st.markdown("### 🔮 Prediksi Hasil Panen Padi Tahun 2026")
    st.caption(f"Prediksi dilakukan secara rekursif berdasarkan time step = **{p_ts} periode**. Setiap langkah prediksi menggunakan output prediksi sebelumnya sebagai input berikutnya.")

    model_2026 = st.session_state.trained_model
    scaler_x_2026 = st.session_state.scaler_x
    scaler_y_2026 = st.session_state.scaler_y
    df_work = st.session_state.preprocessed_df
    p_feats_2026 = st.session_state.feature_cols
    p_target_2026 = st.session_state.target_col

    try:
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        scaler_name_2026 = st.session_state.get("scaler_name", "Min-Max")
        scaler_cls_2026 = MinMaxScaler if scaler_name_2026 == "Min-Max" else StandardScaler

        # Ambil data terakhir sebagai seed
        # Pastikan hanya kolom numerik yang digunakan (filter Timestamp/object)
        work_numeric_2026 = df_work[p_feats_2026].copy()
        for col in work_numeric_2026.columns:
            if hasattr(work_numeric_2026[col], "dt") or pd.api.types.is_datetime64_any_dtype(work_numeric_2026[col]):
                # Konversi datetime ke ordinal int agar bisa di-scale
                work_numeric_2026[col] = work_numeric_2026[col].apply(
                    lambda x: x.toordinal() if pd.notnull(x) else 0
                )
            elif work_numeric_2026[col].dtype == object:
                # Coba konversi ke numeric, isi NaN dengan 0
                work_numeric_2026[col] = pd.to_numeric(work_numeric_2026[col], errors="coerce").fillna(0)
        work_arr_2026 = work_numeric_2026.values.astype(float)
        x_all_2026 = scaler_x_2026.transform(work_arr_2026)

        # Tentukan jumlah prediksi berdasarkan time_step: minimal 4 (triwulan), atau sesuai time step
        n_pred_2026 = max(4, p_ts)  # setidaknya 4 periode untuk Q1-Q4

        # Recursive forecast
        last_window = x_all_2026[-p_ts:].copy()  # shape (time_step, n_features)
        target_idx = p_feats_2026.index(p_target_2026)

        preds_scaled_2026 = []
        window = last_window.copy()
        for step in range(n_pred_2026):
            inp = window.reshape(1, p_ts, len(p_feats_2026))
            pred_s = model_2026.predict(inp, verbose=0).ravel()[0]
            preds_scaled_2026.append(pred_s)
            # Geser window dan update nilai target
            new_row = window[-1].copy()
            new_row[target_idx] = pred_s
            window = np.vstack([window[1:], new_row])

        # Inverse transform ke satuan asli
        preds_2026 = scaler_y_2026.inverse_transform(np.array(preds_scaled_2026).reshape(-1, 1)).ravel()

        # Buat label periode
        period_labels_2026 = [f"Periode {i+1} / 2026" for i in range(n_pred_2026)]

        # Tampilkan metrik ringkas
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); border-radius: 14px; padding: 20px 24px; color: white; margin-bottom: 20px;">
            <h4 style="margin: 0 0 8px 0; color: #a7f3d0;">📊 Ringkasan Prediksi 2026</h4>
            <div style="display: flex; gap: 32px; flex-wrap: wrap;">
                <div><span style="font-size: 0.8rem; color: #d1fae5;">Jumlah Periode Diprediksi</span><br><strong style="font-size: 1.4rem;">{n_pred_2026} Periode</strong></div>
                <div><span style="font-size: 0.8rem; color: #d1fae5;">Total Prediksi (Quintal)</span><br><strong style="font-size: 1.4rem;">{sum(preds_2026):,.2f} Kw</strong></div>
                <div><span style="font-size: 0.8rem; color: #d1fae5;">Total Prediksi (Ton)</span><br><strong style="font-size: 1.4rem;">{sum(preds_2026)/10:,.2f} Ton</strong></div>
                <div><span style="font-size: 0.8rem; color: #d1fae5;">Rata-rata per Periode</span><br><strong style="font-size: 1.4rem;">{sum(preds_2026)/n_pred_2026:,.2f} Kw</strong></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Tabel prediksi
        df_pred_2026 = pd.DataFrame({
            "Periode": period_labels_2026,
            "Prediksi (Quintal/Kw)": preds_2026,
            "Prediksi (Ton)": preds_2026 / 10,
        })

        st.dataframe(df_pred_2026.style.format({
            "Prediksi (Quintal/Kw)": "{:,.2f}",
            "Prediksi (Ton)": "{:,.3f}",
        }), use_container_width=True)

        # Grafik line prediksi 2026
        st.markdown("#### 📈 Grafik Prediksi Deret Waktu 2026")
        chart_df_2026 = pd.DataFrame({
            "Periode": period_labels_2026,
            "Prediksi (Quintal)": preds_2026,
        }).set_index("Periode")
        st.line_chart(chart_df_2026, color=["#10b981"])

        # === ANALISIS KUARTAL (Q1–Q4) ===
        st.markdown("#### 📊 Analisis Per Kuartal Tahun 2026")
        st.caption("Pembagian kuartal dilakukan secara merata dari total periode yang diprediksi.")

        # Bagi ke 4 kuartal secara proporsional
        n = len(preds_2026)
        q_size = n // 4
        remainder = n % 4

        q_vals = []
        q_labels = []
        idx_start = 0
        for q in range(4):
            extra = 1 if q < remainder else 0
            idx_end = idx_start + q_size + extra
            q_sum_kw = float(np.sum(preds_2026[idx_start:idx_end])) if idx_end > idx_start else 0.0
            q_vals.append(q_sum_kw)
            q_labels.append(f"Q{q+1} 2026")
            idx_start = idx_end

        q_df = pd.DataFrame({
            "Kuartal": q_labels,
            "Total (Quintal/Kw)": q_vals,
            "Total (Ton)": [v/10 for v in q_vals],
        })

        # Kolom: bar chart kuartal + tabel total
        col_bar, col_tot = st.columns([2, 1])
        with col_bar:
            st.bar_chart(q_df.set_index("Kuartal")[["Total (Quintal/Kw)"]], color=["#059669"])
            st.caption("Diagram batang total produksi per kuartal (Quintal/Kw)")
        with col_tot:
            st.markdown("**📋 Rincian Per Kuartal**")
            for _, row in q_df.iterrows():
                st.metric(
                    label=row["Kuartal"],
                    value=f"{row['Total (Quintal/Kw)']:,.2f} Kw",
                    delta=f"{row['Total (Ton)']:,.3f} Ton"
                )
            st.markdown(f"""
            <div style="background: #f0fdf4; border: 1.5px solid #a7f3d0; border-radius: 10px; padding: 12px 16px; margin-top: 12px;">
                <div style="font-weight: 700; color: #065f46; margin-bottom: 4px;">🧮 Total Keseluruhan 2026</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #059669;">{sum(q_vals):,.2f} Kw</div>
                <div style="font-size: 0.9rem; color: #047857;">{sum(q_vals)/10:,.3f} Ton</div>
            </div>
            """, unsafe_allow_html=True)

        # Download prediksi 2026
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        st.download_button(
            "📥 Unduh Prediksi 2026 (CSV)",
            data=df_pred_2026.to_csv(index=False).encode("utf-8"),
            file_name="prediksi_panen_2026.csv",
            mime="text/csv",
            use_container_width=True,
        )
    except Exception as e_2026:
        st.error(f"Gagal membuat prediksi 2026: {e_2026}")

render_step_navigation(6)
