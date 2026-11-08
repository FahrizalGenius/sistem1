import io
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS = ["csv", "xlsx", "xls"]

def read_uploaded_file(uploaded_file, sheet_name=None):
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
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

def find_kabupaten_column(df):
    """Mencari kolom kabupaten/wilayah dalam dataframe secara otomatis."""
    if df is None:
        return None
    candidates = [
        "kabupaten", "kab", "kab_kota", "kabupaten/kota", "kabupaten_kota",
        "kota", "wilayah", "daerah", "regency", "nama_kabupaten", "nama kabupaten"
    ]
    for col in df.columns:
        if str(col).strip().lower() in candidates:
            return col
    for col in df.columns:
        clean_col = str(col).strip().lower()
        if ("kab" in clean_col or "wilayah" in clean_col) and (df[col].dtype == object or isinstance(df[col].dtype, pd.CategoricalDtype)):
            return col
    return None

def get_kabupaten_list(df, col=None):
    """Mengambil daftar nilai unik kabupaten yang ada di dataset."""
    if df is None:
        return []
    target_col = col if col else find_kabupaten_column(df)
    if target_col and target_col in df.columns:
        vals = [str(x).strip() for x in df[target_col].dropna().unique() if str(x).strip()]
        return sorted(list(dict.fromkeys(vals)))
    return []

def infer_data_frequency(df, date_col=None):
    """
    Mendeteksi secara otomatis apakah data deret waktu bersifat:
    - Bulanan (Monthly)
    - Tahunan (Yearly)
    - Musim Tanam / Triwulan
    - Harian / Mingguan
    Mengembalikan tuple: (frekuensi_label, alasan_deteksi, rekomendasi_time_step)
    """
    if df is None or df.empty:
        return "Ikuti urutan data", "Dataset kosong", 4

    # 1. Analisis kolom tanggal eksplisit jika ada
    col_to_check = date_col
    if not col_to_check:
        dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            col_to_check = dt_cols[0]
        else:
            for c in df.columns:
                c_clean = str(c).strip().lower()
                if any(k in c_clean for k in ["tanggal", "date", "waktu", "periode", "tgl"]):
                    sample = df[c].dropna().head(30)
                    parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                    if parsed.notna().mean() >= 0.8:
                        col_to_check = c
                        break

    if col_to_check and col_to_check in df.columns:
        dt_series = pd.to_datetime(df[col_to_check], errors="coerce").dropna().drop_duplicates().sort_values()
        if len(dt_series) >= 3:
            diffs = dt_series.diff().dt.days.dropna()
            med_diff = diffs.median()
            if 25 <= med_diff <= 35:
                return "Bulanan (Monthly)", f"Interval tanggal median {med_diff:.0f} hari (deret bulanan)", 6
            elif 340 <= med_diff <= 390:
                return "Tahunan (Yearly)", f"Interval tanggal median {med_diff:.0f} hari (deret tahunan)", 2
            elif 75 <= med_diff <= 125:
                return "Musim Tanam / Triwulan", f"Interval tanggal median {med_diff:.0f} hari (siklus triwulanan/musim)", 3
            elif 6 <= med_diff <= 8:
                return "Mingguan", f"Interval tanggal median {med_diff:.0f} hari (deret mingguan)", 4
            elif med_diff <= 2:
                return "Harian", f"Interval tanggal median {med_diff:.0f} hari (deret harian)", 7

    # 2. Cek kolom bulan eksplisit (misal: 'bulan', 'bulan tanam', 'bulan panen', 'month')
    month_cols = [c for c in df.columns if "bulan" in str(c).strip().lower() or "month" in str(c).strip().lower()]
    year_cols = [c for c in df.columns if "tahun" in str(c).strip().lower() or "year" in str(c).strip().lower()]

    if month_cols:
        target_m = month_cols[0]
        unique_m = df[target_m].dropna().unique()
        if len(unique_m) > 1:
            return "Bulanan (Monthly)", f"Terdeteksi kolom bulan '{str(target_m).strip()}' dengan variasi data bulanan", 6

    # 3. Cek jika hanya ada kolom tahun tanpa pemecahan bulan
    if year_cols and not month_cols:
        target_y = year_cols[0]
        y_vals = pd.to_numeric(df[target_y], errors="coerce").dropna()
        if len(y_vals) > 0 and (y_vals.nunique() >= len(df) * 0.6 or len(df) <= 40):
            return "Tahunan (Yearly)", f"Terdeteksi kolom tahun '{str(target_y).strip()}' tanpa rincian bulanan", 2

    # 4. Cek kolom siklus tanam / musim
    season_cols = [c for c in df.columns if any(k in str(c).strip().lower() for k in ["tanam ke", "musim", "triwulan", "quarter"])]
    if season_cols:
        return "Musim Tanam / Triwulan", f"Terdeteksi kolom siklus tanam '{str(season_cols[0]).strip()}'", 3

    # 5. Heuristik berdasarkan jumlah baris
    if len(df) <= 35:
        return "Tahunan (Yearly)", f"Dataset memiliki {len(df)} baris (pola observasi tahunan)", 2
    else:
        return "Bulanan (Monthly)", f"Dataset memiliki {len(df)} baris observasi deret waktu", 6

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

    rainfall = 220 + 130 * np.sin((months + 2) * np.pi / 6) + np.random.normal(0, 20, n)
    rainfall = np.clip(rainfall, 35.0, 480.0).round(1)

    temp = 27.2 - 0.8 * np.sin((months + 2) * np.pi / 6) + np.random.normal(0, 0.35, n)
    temp = np.clip(temp, 25.2, 30.5).round(1)

    humidity = 76 + 9 * np.sin((months + 2) * np.pi / 6) + np.random.normal(0, 2.0, n)
    humidity = np.clip(humidity, 60.0, 95.0).round(1)

    harvest_area = 28500 + 11500 * np.sin((months - 1) * np.pi / 3) + np.random.normal(0, 1200, n)
    harvest_area = np.clip(harvest_area, 15000, 48000).round(0)

    trend = np.linspace(0, 0.3, n)
    yield_per_ha = 5.25 + trend + (rainfall - 200) * 0.0008 - np.abs(temp - 27.0) * 0.05 + np.random.normal(0, 0.1, n)
    yield_per_ha = np.clip(yield_per_ha, 4.6, 6.2)

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
