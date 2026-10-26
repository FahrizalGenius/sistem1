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

render_step_navigation(6)
