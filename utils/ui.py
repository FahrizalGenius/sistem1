import streamlit as st
from utils.state import STAGES, is_complete, can_open, initialize_state, invalidate_from

# Definisi halaman untuk navigasi cepat
PAGE_FILES = [
    "pages/1_Upload_Dataset.py",
    "pages/2_Pengecekan_Data.py",
    "pages/3_Preprocessing.py",
    "pages/4_Pengaturan_Data.py",
    "pages/5_Pemilihan_Model.py",
    "pages/6_Training_Prediksi.py",
    "pages/7_Evaluasi.py",
]

PAGE_ICONS = ["📂", "🔍", "⚙️", "⏱️", "🧠", "🚀", "📊"]

def apply_custom_css():
    """Menginjeksi CSS kustom modern dengan estetika Agriculture AI yang elegan."""
    st.markdown("""
    <style>
    /* Import Google Fonts modern */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0f172a !important;
        background-color: #f8fafc !important;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* General Typography Contrast Fixes */
    p, span, div, li, label, h1, h2, h3, h4, h5, h6 {
        color: inherit;
    }

    .stMarkdown, [data-testid="stMarkdownContainer"] p {
        color: #1e293b !important;
    }

    [data-testid="stCaptionContainer"] p {
        color: #475569 !important;
        font-weight: 500 !important;
    }

    /* Hide redundant native Streamlit navigation on top of sidebar */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdf4 0%, #f8fafc 100%) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2) !important;
    }

    [data-testid="stSidebar"] * {
        color: #1e293b;
    }

    /* Streamlit Page Links (st.page_link) styling */
    [data-testid="stPageLink-NavLink"], [data-testid="stPageLink"] a {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 8px 14px !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        display: flex !important;
        align-items: center !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stPageLink-NavLink"]:hover, [data-testid="stPageLink"] a:hover {
        background-color: #ecfdf5 !important;
        border-color: #10b981 !important;
        color: #047857 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15) !important;
    }

    [data-testid="stPageLink-NavLink"] *,
    [data-testid="stPageLink"] a * {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    [data-testid="stPageLink-NavLink"]:hover *,
    [data-testid="stPageLink"] a:hover * {
        color: #047857 !important;
    }

    /* Sidebar specific navigation links */
    [data-testid="stSidebar"] [data-testid="stPageLink-NavLink"],
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 6px !important;
    }

    /* Metric Card Modernization */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.05) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease !important;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        border-color: #10b981 !important;
        box-shadow: 0 10px 24px -4px rgba(16, 185, 129, 0.18) !important;
    }
    [data-testid="stMetricLabel"] * {
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        color: #475569 !important;
    }
    [data-testid="stMetricValue"] * {
        font-size: 1.65rem !important;
        font-weight: 800 !important;
        color: #064e3b !important;
        letter-spacing: -0.5px !important;
    }
    [data-testid="stMetricDelta"] * {
        color: #059669 !important;
        font-weight: 600 !important;
    }

    /* Button Enhancements */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        padding: 0.5rem 1.25rem !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        color: #1e293b !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        border-color: #10b981 !important;
        color: #059669 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(5, 150, 105, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(5, 150, 105, 0.45) !important;
        color: #ffffff !important;
    }
    .stButton > button:disabled {
        background: #f1f5f9 !important;
        border-color: #e2e8f0 !important;
        color: #94a3b8 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Widget Labels & Inputs */
    [data-testid="stWidgetLabel"] p, label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border-color: #cbd5e1 !important;
    }
    div[data-baseweb="input"] input, div[data-baseweb="base-input"] input {
        background-color: #ffffff !important;
        color: #0f172a !important;
    }

    /* Card Containers */
    .agri-card {
        background: #ffffff;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(16, 185, 129, 0.18);
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.04);
        margin-bottom: 20px;
        transition: all 0.2s ease;
    }
    .agri-card:hover {
        box-shadow: 0 10px 28px rgba(16, 185, 129, 0.1);
        border-color: rgba(16, 185, 129, 0.35);
    }

    /* Stage Header Hero Banner */
    .stage-banner {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 60%, #047857 100%);
        border-radius: 18px;
        padding: 26px 30px;
        color: white !important;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(6, 78, 59, 0.2);
        position: relative;
        overflow: hidden;
    }
    .stage-banner::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(52, 211, 153, 0.2) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .stage-banner .breadcrumb {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #a7f3d0 !important;
        margin-bottom: 6px;
    }
    .stage-banner h1 {
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin: 0 0 8px 0 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }
    .stage-banner p {
        font-size: 0.95rem;
        margin: 0;
        color: #d1fae5 !important;
        line-height: 1.5;
        max-width: 850px;
    }

    /* Badge & Tag styling */
    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-success {
        background-color: #d1fae5;
        color: #065f46;
        border: 1px solid #6ee7b7;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
    .badge-info {
        background-color: #e0f2fe;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }

    /* Sidebar Progress Stepper */
    .step-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 10px;
        font-size: 0.88rem;
        font-weight: 500;
        margin-bottom: 4px;
        color: #334155;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }
    .step-item.completed {
        background: #ecfdf5;
        border-color: #a7f3d0;
        color: #065f46;
        font-weight: 600;
    }
    .step-item.active {
        background: #f0fdf4;
        border-color: #10b981;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.15);
    }

    /* Modern Tabs & Expander */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #e2e8f0 !important;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 8px 16px !important;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #059669 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }

    /* Alerts / Callouts */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(0,0,0,0.06) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
    }
    .stAlert * {
        color: #0f172a !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #cbd5e1;
        background: #ffffff;
    }

    /* Bottom Navigation Bar */
    .step-nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 36px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Merender bilah samping (sidebar) yang terstruktur, elegan, dan informatif."""
    apply_custom_css()

    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; padding: 10px 0 16px 0;">
            <div style="font-size: 2.2rem; background: #ecfdf5; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; border-radius: 12px; border: 1px solid #a7f3d0;">🌾</div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #064e3b; letter-spacing: -0.3px;">SYSTEM 1</h3>
                <p style="margin: 0; font-size: 0.76rem; font-weight: 600; color: #059669; text-transform: uppercase; letter-spacing: 0.5px;">Peramalan Panen Padi AI</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Hitung kemajuan tahapan
        completed_count = sum(1 for n in range(1, 8) if is_complete(n))
        pct = int((completed_count / 7) * 100)

        st.caption(f"**Progres Pipeline:** {completed_count}/7 Tahap Selesai ({pct}%)")
        st.progress(completed_count / 7.0)

        st.markdown("---")
        st.markdown("<p style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.8px; margin-bottom: 8px;'>Navigasi Pipeline</p>", unsafe_allow_html=True)

        # Navigasi ke Beranda
        st.page_link("app.py", label="Beranda (Dashboard)", icon="🏠")

        # Navigasi ke setiap tahap
        for n, stage in enumerate(STAGES, start=1):
            done = is_complete(n)
            openable = can_open(n)
            page_path = PAGE_FILES[n - 1]
            icon = "✅" if done else (PAGE_ICONS[n - 1] if openable else "🔒")
            
            # Label tahap tanpa nomor ganda
            clean_name = stage.split(". ", 1)[1] if ". " in stage else stage
            label = f"{n}. {clean_name}"
            
            if openable:
                st.page_link(page_path, label=label, icon=icon)
            else:
                st.markdown(f"<div style='opacity: 0.55; padding: 4px 8px; font-size: 0.88rem; color: #64748b;'>🔒 {label}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Status Ringkas Sistem
        st.markdown("<p style='font-size: 0.78rem; font-weight: 700; text-transform: uppercase; color: #64748b; letter-spacing: 0.8px; margin-bottom: 8px;'>Status Sistem</p>", unsafe_allow_html=True)
        raw_df = st.session_state.get("raw_df")
        model = st.session_state.get("trained_model")
        metrics = st.session_state.get("eval_metrics")

        if raw_df is not None:
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #d1fae5; border-radius: 10px; padding: 10px; margin-bottom: 8px; font-size: 0.8rem;">
                <div style="color: #065f46; font-weight: 700;">📊 Dataset Siap</div>
                <div style="color: #475569; margin-top: 2px;">{raw_df.shape[0]:,} baris × {raw_df.shape[1]} kolom</div>
                <div style="color: #64748b; font-size: 0.74rem; margin-top: 2px;">File: {st.session_state.get('source_name', 'Dataset')}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size: 0.8rem; color: #94a3b8; font-style: italic;'>Belum ada dataset dimuat.</div>", unsafe_allow_html=True)

        if model is not None:
            r2_val = metrics.get('R²', None) if metrics else None
            r2_str = f"R²: {r2_val:.4f}" if (r2_val is not None and not str(r2_val) == 'nan') else "Telah dilatih"
            st.markdown(f"""
            <div style="background: #ffffff; border: 1px solid #bae6fd; border-radius: 10px; padding: 10px; margin-bottom: 12px; font-size: 0.8rem;">
                <div style="color: #0369a1; font-weight: 700;">🧠 {st.session_state.get('model_choice', 'Model')}</div>
                <div style="color: #0284c7; font-weight: 600; margin-top: 2px;">{r2_str}</div>
            </div>
            """, unsafe_allow_html=True)

        # Tombol Reset
        if st.button("🔄 Reset Seluruh Sesi", use_container_width=True, help="Menghapus semua data dan memulai dari awal"):
            invalidate_from(1)
            st.session_state.raw_df = None
            st.session_state.source_name = None
            st.session_state.completed = set()
            st.rerun()

def stage_header(title, description, icon="🌾", stage_number=None):
    """Menampilkan hero banner untuk setiap halaman tahap."""
    initialize_state()
    apply_custom_css()
    render_sidebar()

    # Ekstraksi nomor tahap dari title jika tidak diberikan
    if stage_number is None and title and title[0].isdigit():
        try:
            stage_number = int(title.split(".")[0].strip())
        except Exception:
            stage_number = None

    stage_badge = f"Tahap {stage_number} dari 7" if stage_number else "Modul Pipeline"
    
    st.markdown(f"""
    <div class="stage-banner">
        <div class="breadcrumb">🌾 SYSTEM 1 &nbsp;•&nbsp; {stage_badge}</div>
        <h1>{title}</h1>
        <p>{description}</p>
    </div>
    """, unsafe_allow_html=True)

def require_previous(stage_number: int):
    """Memastikan tahap sebelumnya telah selesai, jika belum tampilkan antarmuka ramah."""
    if not can_open(stage_number):
        initialize_state()
        apply_custom_css()
        render_sidebar()

        prereq = stage_number - 1
        prereq_name = STAGES[prereq - 1] if prereq >= 1 else "Tahap 1"
        prereq_page = PAGE_FILES[prereq - 1] if prereq >= 1 else PAGE_FILES[0]

        st.markdown(f"""
        <div class="agri-card" style="text-align: center; padding: 40px 20px;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🔒</div>
            <h3 style="color: #0f172a; margin-bottom: 8px;">Tahap Ini Masih Terkunci</h3>
            <p style="color: #64748b; max-width: 500px; margin: 0 auto 20px auto;">
                Untuk menjaga integritas alur penelitian deret waktu, Anda perlu menyelesaikan 
                <strong>{prereq_name}</strong> terlebih dahulu.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.page_link(prereq_page, label=f"Buka {prereq_name} ➔", icon="📂", use_container_width=True)
        st.stop()

def render_step_navigation(current_stage: int, next_page: str = None, next_label: str = None):
    """Merender tombol navigasi bawah untuk berpindah antar tahap dengan mulus."""
    st.markdown("<div style='margin-top: 32px;'></div>", unsafe_allow_html=True)
    c_prev, c_space, c_next = st.columns([2, 3, 2])

    if current_stage > 1:
        prev_page = PAGE_FILES[current_stage - 2]
        prev_name = STAGES[current_stage - 2].split(". ", 1)[1]
        with c_prev:
            st.page_link(prev_page, label=f"⬅️ Tahap {current_stage-1}: {prev_name}", use_container_width=True)

    if current_stage < 7 and is_complete(current_stage):
        target_page = next_page or PAGE_FILES[current_stage]
        target_name = next_label or f"Lanjut ke Tahap {current_stage+1} ➔"
        with c_next:
            st.page_link(target_page, label=target_name, icon="➡️", use_container_width=True)
