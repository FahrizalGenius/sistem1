import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def prepare_sequences(df, feature_cols, target_col, time_step, split_ratio, scaler_name="Min-Max"):
    """Chronological split, fit scalers on training portion only, then build sequences."""
    if not feature_cols:
        raise ValueError("Pilih setidaknya satu fitur.")
    if target_col not in df.columns:
        raise ValueError("Kolom target tidak ditemukan.")
    if target_col not in feature_cols:
        raise ValueError("Untuk implementasi ini, kolom target juga harus termasuk fitur agar target historis tersedia.")

    work = df[feature_cols].copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan)
    if work.isna().any().any():
        raise ValueError("Masih ada missing/non-numeric value pada fitur yang dipilih. Periksa preprocessing.")
    if len(work) <= time_step + 2:
        raise ValueError("Data terlalu sedikit untuk time step yang dipilih.")

    split_idx = int(len(work) * split_ratio)
    if split_idx <= time_step or split_idx >= len(work):
        raise ValueError("Rasio split tidak menghasilkan data latih/uji yang cukup.")

    scaler_cls = MinMaxScaler if scaler_name == "Min-Max" else StandardScaler
    scaler_x = scaler_cls()
    scaler_y = scaler_cls()

    # Fit only on training rows to avoid leakage.
    x_scaler_cols = feature_cols
    train_x_raw = work.iloc[:split_idx][x_scaler_cols]
    scaler_x.fit(train_x_raw)
    scaler_y.fit(work.iloc[:split_idx][[target_col]])

    x_all = scaler_x.transform(work[x_scaler_cols])
    y_all = scaler_y.transform(work[[target_col]]).ravel()

    X, y, positions = [], [], []
    for target_pos in range(time_step, len(work)):
        X.append(x_all[target_pos-time_step:target_pos, :])
        y.append(y_all[target_pos])
        positions.append(target_pos)

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    positions = np.asarray(positions)

    train_mask = positions < split_idx
    test_mask = positions >= split_idx
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test_scaled = X[test_mask], y[test_mask]
    test_positions = positions[test_mask]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Data latih atau data uji kosong. Ubah rasio split atau time step.")

    y_test_actual = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()
    return X_train, y_train, X_test, y_test_actual, test_positions, scaler_x, scaler_y, work

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    nonzero = np.abs(y_true) > 1e-12
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else float("nan")
    r2 = r2_score(y_true, y_pred) if len(y_true) >= 2 else float("nan")
    return {"MAE": float(mae), "MSE": float(mse), "RMSE": rmse, "MAPE (%)": mape, "R²": float(r2)}

def build_lstm(units=64, layers=1, dropout=0.0, learning_rate=0.001, input_shape=None):
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow belum terpasang. Jalankan: pip install tensorflow") from exc

    model = tf.keras.Sequential()
    for i in range(layers):
        return_sequences = i < layers - 1
        if i == 0:
            model.add(tf.keras.layers.LSTM(units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(tf.keras.layers.LSTM(units, return_sequences=return_sequences))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model

def train_baseline(X_train, y_train, units, layers, dropout, learning_rate, batch_size, epochs, validation_split):
    model = build_lstm(
        units=units, layers=layers, dropout=dropout,
        learning_rate=learning_rate, input_shape=(X_train.shape[1], X_train.shape[2])
    )
    callbacks = []
    try:
        import tensorflow as tf
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if validation_split > 0 else "loss",
            patience=10, restore_best_weights=True
        ))
    except ImportError:
        pass
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split if validation_split > 0 else 0.0,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )
    return model, history.history

def train_pso(X_train, y_train, X_val, y_val, pso_particles, pso_iterations,
              units_min, units_max, lr_min, lr_max, batch_options,
              epochs, layers, dropout, validation_split):
    """Compact PSO over units, learning rate, and batch size; minimizes validation MSE."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError("TensorFlow belum terpasang. Jalankan: pip install tensorflow") from exc

    rng = np.random.default_rng(42)
    # Particle dimensions: units, learning rate, batch-size option index.
    lower = np.array([units_min, np.log10(lr_min), 0.0], dtype=float)
    upper = np.array([units_max, np.log10(lr_max), max(0, len(batch_options)-1)], dtype=float)
    positions = rng.uniform(lower, upper, size=(pso_particles, 3))
    velocities = rng.uniform(-0.1, 0.1, size=(pso_particles, 3))
    pbest_positions = positions.copy()
    pbest_scores = np.full(pso_particles, np.inf)
    gbest_position = None
    gbest_score = np.inf
    history = []

    def decode(pos):
        units = int(np.clip(round(pos[0] / 8) * 8, units_min, units_max))
        lr = float(10 ** np.clip(pos[1], np.log10(lr_min), np.log10(lr_max)))
        batch_idx = int(np.clip(round(pos[2]), 0, len(batch_options)-1))
        return units, lr, int(batch_options[batch_idx])

    for iteration in range(pso_iterations):
        for i in range(pso_particles):
            units, lr, batch_size = decode(positions[i])
            tf.keras.backend.clear_session()
            model = build_lstm(
                units=units, layers=layers, dropout=dropout,
                learning_rate=lr, input_shape=(X_train.shape[1], X_train.shape[2])
            )
            model.fit(
                X_train, y_train,
                epochs=max(3, min(epochs, 30)),
                batch_size=batch_size,
                validation_split=validation_split if validation_split > 0 else 0.0,
                shuffle=False, verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss" if validation_split > 0 else "loss",
                    patience=4, restore_best_weights=True
                )],
            )
            pred = model.predict(X_val, verbose=0).ravel()
            score = float(np.mean((y_val - pred) ** 2))
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()
            if score < gbest_score:
                gbest_score = score
                gbest_position = positions[i].copy()
        history.append({"Iterasi": iteration + 1, "Best MSE": gbest_score})
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = 0.5 * velocities + 1.5 * r1 * (pbest_positions - positions) + 1.5 * r2 * (gbest_position - positions)
        positions = np.clip(positions + velocities, lower, upper)

    best_units, best_lr, best_batch = decode(gbest_position)
    # Final model trained on all supplied training data.
    final_model, final_history = train_baseline(
        X_train, y_train, best_units, layers, dropout, best_lr,
        best_batch, epochs, validation_split
    )
    return final_model, final_history, {
        "units": best_units, "learning_rate": best_lr, "batch_size": best_batch,
        "best_validation_mse_scaled": float(gbest_score),
        "pso_history": history,
    }
