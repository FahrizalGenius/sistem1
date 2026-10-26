import streamlit as st

STAGES = [
    "1. Upload Dataset",
    "2. Pengecekan Data",
    "3. Preprocessing",
    "4. Pengaturan Data",
    "5. Pemilihan Model",
    "6. Training & Prediksi",
    "7. Evaluasi",
]

def initialize_state():
    defaults = {
        "raw_df": None,
        "source_name": None,
        "quality_checked": False,
        "clean_df": None,
        "preprocessed_df": None,
        "feature_cols": [],
        "target_col": None,
        "date_col": None,
        "frequency": "Ikuti urutan data",
        "time_step": 6,
        "split_ratio": 0.8,
        "model_choice": "LSTM Baseline",
        "model_params": {},
        "trained_model": None,
        "scaler_x": None,
        "scaler_y": None,
        "y_test": None,
        "y_pred": None,
        "eval_metrics": None,
        "training_info": None,
        "completed": set(),
        "last_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def invalidate_from(stage_number: int):
    """Clear downstream completion/results when upstream choices change."""
    completed = set(st.session_state.completed)
    completed = {s for s in completed if int(s.split(".")[0]) < stage_number}
    st.session_state.completed = completed

    if stage_number <= 2:
        st.session_state.clean_df = None
        st.session_state.preprocessed_df = None
        st.session_state.feature_cols = []
        st.session_state.target_col = None
    if stage_number <= 3:
        st.session_state.preprocessed_df = None
    if stage_number <= 4:
        st.session_state.trained_model = None
        st.session_state.y_test = None
        st.session_state.y_pred = None
        st.session_state.eval_metrics = None
        st.session_state.training_info = None
    if stage_number <= 5:
        st.session_state.trained_model = None
        st.session_state.y_test = None
        st.session_state.y_pred = None
        st.session_state.eval_metrics = None
        st.session_state.training_info = None

def mark_complete(stage_number: int):
    completed = set(st.session_state.completed)
    completed.add(STAGES[stage_number - 1])
    st.session_state.completed = completed

def is_complete(stage_number: int) -> bool:
    return STAGES[stage_number - 1] in st.session_state.completed

def can_open(stage_number: int) -> bool:
    if stage_number == 1:
        return True
    return all(is_complete(n) for n in range(1, stage_number))
