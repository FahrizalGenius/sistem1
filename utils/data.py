import io
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls"]

def read_uploaded_file(uploaded_file, sheet_name=None):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        # Try common encodings and delimiters.
        for encoding in ("utf-8-sig", "utf-8", "latin1"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
            except Exception:
                continue
        raise ValueError("CSV tidak dapat dibaca. Periksa encoding dan pemisah kolom.")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), sheet_name=sheet_name or 0)
    raise ValueError("Format tidak didukung. Gunakan CSV, XLSX, atau XLS.")

def infer_datetime_columns(df):
    cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            cols.append(col)
        elif df[col].dtype == "object":
            sample = df[col].dropna().head(30)
            if len(sample):
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                if parsed.notna().mean() >= 0.8:
                    cols.append(col)
    return cols

def coerce_numeric_columns(df, columns):
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

def missing_summary(df):
    return pd.DataFrame({
        "Kolom": df.columns,
        "Tipe data": [str(df[c].dtype) for c in df.columns],
        "Missing value": [int(df[c].isna().sum()) for c in df.columns],
        "Persentase missing (%)": [
            round(float(df[c].isna().mean() * 100), 2) for c in df.columns
        ],
        "Nilai unik": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })

def validate_dataset(df):
    problems = []
    if df is None or df.empty:
        problems.append("Dataset kosong.")
    if df is not None and len(df.columns) < 2:
        problems.append("Dataset harus memiliki minimal dua kolom untuk fitur dan target.")
    if df is not None and df.columns.duplicated().any():
        problems.append("Terdapat nama kolom yang duplikat.")
    return problems

def generate_sample_paddy_data():
    """Menghasilkan dataset deret waktu bulanan simulasi panen padi realistis (10 tahun: 120 baris)."""
    np.random.seed(42)
    dates = pd.date_range(start="2014-01-01", periods=120, freq="MS")
    n = len(dates)
    months = dates.month

    # Musiman curah hujan (Puncak basah Des-Feb, kering Jun-Ags)
    rainfall = 220 + 130 * np.sin((months + 2) * np.pi / 6) + np.random.normal(0, 20, n)
    rainfall = np.clip(rainfall, 35.0, 480.0).round(1)

    # Suhu rata-rata (°C)
    temp = 27.2 - 0.8 * np.sin((months + 2) * np.pi / 6) + np.random.normal(0, 0.35, n)
    temp = np.clip(temp, 25.2, 30.5).round(1)

    # Kelembaban udara (%)
    humidity = 76 + 9 * np.sin((months + 2) * np.pi / 6) + np.random.normal(0, 2.0, n)
    humidity = np.clip(humidity, 60.0, 95.0).round(1)

    # Luas panen (Hektar) - Siklus tanam 2-3 kali setahun (Puncak panen Mar-Mei & Sep-Okt)
    harvest_area = 28500 + 11500 * np.sin((months - 1) * np.pi / 3) + np.random.normal(0, 1200, n)
    harvest_area = np.clip(harvest_area, 15000, 48000).round(0)

    # Produktivitas padi (Ton/ha) dengan tren peningkatan teknologi dan pengaruh iklim
    trend = np.linspace(0, 0.3, n)
    yield_per_ha = 5.25 + trend + (rainfall - 200) * 0.0008 - np.abs(temp - 27.0) * 0.05 + np.random.normal(0, 0.1, n)
    yield_per_ha = np.clip(yield_per_ha, 4.6, 6.2)

    # Total Produksi Padi (Ton)
    production = (harvest_area * yield_per_ha).round(0)

    df = pd.DataFrame({
        "Tanggal": dates.strftime("%Y-%m-%d"),
        "Luas_Panen_ha": harvest_area,
        "Curah_Hujan_mm": rainfall,
        "Suhu_Rata2_C": temp,
        "Kelembaban_Persen": humidity,
        "Produksi_Padi_Ton": production
    })
    return df

