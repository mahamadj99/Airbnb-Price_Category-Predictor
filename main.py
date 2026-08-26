import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ─────────────────────────────────────────────
#  Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Airbnb Price Category Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  Load Model & Preprocessor
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    try:
        with open(os.path.join(BASE_DIR, "airbnb_model.pkl"), "rb") as f:
            model = pickle.load(f)
        with open(os.path.join(BASE_DIR, "airbnb_preprocessor.pkl"), "rb") as f:
            preprocessor = pickle.load(f)
        return model, preprocessor, None
    except FileNotFoundError as e:
        return None, None, f"Model file not found: {e}"
    except Exception as e:
        return None, None, f"Error loading model: {e}"

model, preprocessor, load_error = load_artifacts()

# ─────────────────────────────────────────────
#  Neighbourhood Data (grouped by borough)
# ─────────────────────────────────────────────
NEIGHBOURHOODS = {
    "Manhattan": [
        "Harlem", "East Harlem", "Upper West Side", "Upper East Side",
        "Midtown", "Hell's Kitchen", "Chelsea", "Greenwich Village",
        "SoHo", "Tribeca", "Financial District", "Lower East Side",
        "East Village", "West Village", "Chinatown", "Inwood",
        "Washington Heights", "Morningside Heights", "Murray Hill",
        "Gramercy", "Flatiron District", "Nolita", "NoHo"
    ],
    "Brooklyn": [
        "Williamsburg", "Bushwick", "Bedford-Stuyvesant", "Crown Heights",
        "Park Slope", "Sunset Park", "Flatbush", "Greenpoint",
        "DUMBO", "Brooklyn Heights", "Cobble Hill", "Carroll Gardens",
        "Boerum Hill", "Bay Ridge", "Bensonhurst", "Coney Island",
        "East New York", "Flatlands", "Sheepshead Bay"
    ],
    "Queens": [
        "Astoria", "Long Island City", "Flushing", "Jamaica",
        "Jackson Heights", "Forest Hills", "Ridgewood", "Sunnyside",
        "Woodside", "Corona", "Elmhurst"
    ],
    "Bronx": [
        "Riverdale", "Fordham", "Tremont", "Mott Haven",
        "Concourse", "Pelham Bay", "Wakefield"
    ],
    "Staten Island": [
        "St. George", "Stapleton", "New Dorp", "Tottenville",
        "Great Kills", "Eltingville"
    ]
}

# ─────────────────────────────────────────────
#  CSS — Zodiac AI Visual Identity
#  Primary  : #0D1B2A  (deep navy)
#  Secondary: #00B4D8  (cyan)
#  Highlight: #FFFFFF
#  Text     : #E0E0E0
#  Accent   : #90E0EF  (light cyan)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

/* ── Base ── */
html, body, .stApp {
    background-color: #0D1B2A !important;
    font-family: 'Inter', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #091422 !important;
    border-right: 1px solid rgba(0,180,216,0.15);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

/* ── Hide default streamlit elements ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2rem 4rem 2rem !important;
    max-width: 960px !important;
}

/* ── Sidebar Logo Area ── */
.sidebar-logo-wrap {
    background: linear-gradient(160deg, #0a2540 0%, #091422 100%);
    border-bottom: 1px solid rgba(0,180,216,0.15);
    padding: 28px 20px 22px 20px;
    text-align: center;
    margin-bottom: 8px;
}
.sidebar-logo-ring {
    width: 72px;
    height: 72px;
    margin: 0 auto 12px auto;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}
.sidebar-logo-ring svg {
    width: 72px;
    height: 72px;
}
.sidebar-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 3px;
    color: #E0E0E0 !important;
    text-transform: uppercase;
    margin: 0;
}
.sidebar-tagline {
    font-size: 10px;
    letter-spacing: 2px;
    color: #00B4D8 !important;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Sidebar Sections ── */
.sidebar-nav {
    padding: 6px 14px;
}
.sidebar-section-title {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #00B4D8 !important;
    margin: 18px 0 8px 4px;
}
.sidebar-info-card {
    background: rgba(0,180,216,0.06);
    border: 1px solid rgba(0,180,216,0.12);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.sidebar-info-card p {
    font-size: 12px;
    color: #a0c4d8 !important;
    margin: 0 0 5px 0;
    line-height: 1.6;
}
.sidebar-info-card p:last-child { margin-bottom: 0; }
.sidebar-info-card strong {
    color: #E0E0E0 !important;
}

/* ── Tech Badges ── */
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 4px;
}
.badge {
    background: rgba(0,180,216,0.1);
    border: 1px solid rgba(0,180,216,0.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    color: #90E0EF !important;
    font-weight: 500;
}

/* ── Divider ── */
.s-divider {
    border: none;
    border-top: 1px solid rgba(0,180,216,0.1);
    margin: 14px 0;
}

/* ── Team Card ── */
.team-member {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 4px;
    border-bottom: 1px solid rgba(0,180,216,0.08);
}
.team-member:last-child { border-bottom: none; }
.team-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: linear-gradient(135deg, #00B4D8, #0077A8);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
    color: #fff !important;
    flex-shrink: 0;
    font-family: 'Space Grotesk', sans-serif;
}
.team-info { flex: 1; min-width: 0; }
.team-name {
    font-size: 12px;
    font-weight: 600;
    color: #E0E0E0 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.team-links {
    display: flex;
    gap: 8px;
    margin-top: 2px;
}
.team-link {
    font-size: 10px;
    color: #00B4D8 !important;
    text-decoration: none;
    letter-spacing: 0.3px;
}
.team-link:hover { color: #90E0EF !important; }

/* ── Page Header ── */
.page-header {
    text-align: center;
    padding: 36px 0 32px 0;
    position: relative;
}
.page-header-logo {
    margin: 0 auto 18px auto;
    width: 80px;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.page-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(22px, 4vw, 34px);
    font-weight: 700;
    color: #FFFFFF !important;
    letter-spacing: -0.5px;
    margin: 0 0 6px 0;
    line-height: 1.2;
}
.page-title span { color: #00B4D8 !important; }
.page-subtitle {
    font-size: clamp(13px, 1.8vw, 15px);
    color: #6a8fa8 !important;
    letter-spacing: 0.2px;
    margin: 0;
}
.header-line {
    width: 48px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00B4D8, transparent);
    margin: 16px auto 0 auto;
    border-radius: 2px;
}

/* ── Form Section Label ── */
.form-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #00B4D8 !important;
    margin: 24px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.form-section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(0,180,216,0.15);
}

/* ── Inputs ── */
.stSelectbox label,
.stTextInput label,
.stNumberInput label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #7a9db8 !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0a2035 !important;
    border: 1px solid rgba(0,180,216,0.18) !important;
    border-radius: 8px !important;
    color: #E0E0E0 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
.stSelectbox > div > div:focus-within,
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within {
    border-color: rgba(0,180,216,0.5) !important;
    box-shadow: 0 0 0 3px rgba(0,180,216,0.08) !important;
}
div[data-baseweb="select"] > div {
    background: #0a2035 !important;
    border-color: rgba(0,180,216,0.18) !important;
}

/* ── Predict Button ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #00B4D8 0%, #0077A8 100%);
    color: #ffffff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    border-radius: 8px !important;
    border: none !important;
    padding: 14px 24px !important;
    margin-top: 16px;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00c8f0 0%, #0088c2 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,180,216,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Card ── */
.result-wrap {
    margin-top: 28px;
    background: linear-gradient(160deg, #0a2540 0%, #0D1B2A 100%);
    border: 1px solid rgba(0,180,216,0.25);
    border-radius: 14px;
    padding: 32px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-wrap::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00B4D8, transparent);
}
.result-eyebrow {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00B4D8 !important;
    margin-bottom: 10px;
}
.result-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(28px, 5vw, 42px);
    font-weight: 700;
    color: #FFFFFF !important;
    margin: 0 0 20px 0;
    line-height: 1.1;
}
.conf-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6a8fa8 !important;
    margin-bottom: 8px;
}
.conf-bar-track {
    background: rgba(0,180,216,0.1);
    border-radius: 20px;
    height: 6px;
    max-width: 280px;
    margin: 0 auto 10px auto;
    overflow: hidden;
    border: 1px solid rgba(0,180,216,0.12);
}
.conf-bar-fill {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(90deg, #00B4D8, #90E0EF);
}
.conf-pct {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #00B4D8 !important;
}
.conf-status {
    font-size: 12px;
    color: #6a8fa8 !important;
    margin-top: 4px;
    letter-spacing: 0.3px;
}

/* ── Alert Cards ── */
.alert-error {
    background: rgba(220,38,38,0.08);
    border: 1px solid rgba(220,38,38,0.25);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #fca5a5 !important;
}
.alert-warning {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 13px;
    color: #fcd34d !important;
}
.alert-error strong, .alert-warning strong {
    display: block;
    margin-bottom: 2px;
    font-size: 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Responsive ── */
@media (max-width: 768px) {
    .block-container { padding: 1rem 1rem 3rem 1rem !important; }
    .page-header { padding: 20px 0 24px 0; }
    .result-wrap { padding: 24px 16px; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Zodiac Logo SVG (Cyan version — matches UI)
# ─────────────────────────────────────────────
ZODIAC_LOGO_SVG = """
<svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <circle cx="50" cy="50" r="38" stroke="#00B4D8" stroke-width="1.2" stroke-dasharray="2 3"/>
  <circle cx="50" cy="12" r="2.2" fill="#00B4D8"/>
  <circle cx="50" cy="88" r="2.2" fill="#00B4D8"/>
  <circle cx="12" cy="50" r="2.2" fill="#90E0EF"/>
  <circle cx="88" cy="50" r="2.2" fill="#90E0EF"/>
  <path d="M50 8 L51.5 11 L50 10 L48.5 11 Z" fill="#FFFFFF"/>
  <path d="M50 92 L51.5 89 L50 90 L48.5 89 Z" fill="#00B4D8"/>
  <text x="50" y="63" font-family="Georgia,serif" font-size="36" font-weight="700"
        text-anchor="middle" fill="url(#zgrad)">Z</text>
  <defs>
    <linearGradient id="zgrad" x1="30" y1="30" x2="70" y2="70" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="50%" stop-color="#90E0EF"/>
      <stop offset="100%" stop-color="#00B4D8"/>
    </linearGradient>
  </defs>
</svg>
"""

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # Logo + Brand
    st.markdown(f"""
    <div class="sidebar-logo-wrap">
        <div class="sidebar-logo-ring">{ZODIAC_LOGO_SVG}</div>
        <p class="sidebar-brand">The Zodiac</p>
        <p class="sidebar-tagline">Beyond Time, Beyond Limits</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)

    # About
    st.markdown('<div class="sidebar-section-title">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card">
        <p>A <strong>Machine Learning</strong> model trained on 48,000+ NYC Airbnb listings to classify listings into price categories.</p>
        <p>Fill in listing details and get an instant prediction with a confidence score.</p>
    </div>
    """, unsafe_allow_html=True)

    # How it Works
    st.markdown('<div class="sidebar-section-title">How it Works</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card">
        <p>1 — Enter the listing location and details</p>
        <p>2 — Click <strong>Predict</strong></p>
        <p>3 — View the price category and confidence score</p>
    </div>
    """, unsafe_allow_html=True)

    # Dataset
    st.markdown('<div class="sidebar-section-title">Dataset</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-info-card">
        <p><strong>Source:</strong> NYC Airbnb Open Data 2019</p>
        <p><strong>Records:</strong> ~48,895 listings</p>
        <p><strong>Location:</strong> New York City, USA</p>
    </div>
    """, unsafe_allow_html=True)

    # Tech Stack
    st.markdown('<div class="sidebar-section-title">Tech Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="badge-row">
        <span class="badge">Python</span>
        <span class="badge">Scikit-learn</span>
        <span class="badge">Streamlit</span>
        <span class="badge">Pandas</span>
        <span class="badge">NumPy</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="s-divider" style="margin-top:18px">', unsafe_allow_html=True)

    # Team
    st.markdown('<div class="sidebar-section-title">Team</div>', unsafe_allow_html=True)

    TEAM_HTML = """
    <style>
    .tm-card {
        background: rgba(0,180,216,0.06);
        border: 1px solid rgba(0,180,216,0.12);
        border-radius: 10px;
        padding: 10px 12px;
        margin-bottom: 0px;
    }
    .tm-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 0;
        border-bottom: 1px solid rgba(0,180,216,0.08);
    }
    .tm-row:last-child { border-bottom: none; padding-bottom: 0; }
    .tm-row:first-child { padding-top: 0; }
    .tm-avatar {
        width: 34px;
        height: 34px;
        min-width: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #00B4D8, #0077A8);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
    }
    .tm-info { flex: 1; min-width: 0; }
    .tm-name {
        font-size: 12px;
        font-weight: 600;
        color: #E0E0E0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .tm-links { display: flex; gap: 8px; margin-top: 3px; }
    .tm-link {
        font-size: 10px;
        color: #00B4D8;
        text-decoration: none;
        letter-spacing: 0.3px;
        padding: 2px 8px;
        border: 1px solid rgba(0,180,216,0.3);
        border-radius: 20px;
        transition: all 0.2s;
    }
    .tm-link:hover {
        background: rgba(0,180,216,0.15);
        color: #90E0EF;
    }
    </style>

    <div class="tm-card">
        <div class="tm-row">
            <div class="tm-avatar">MO</div>
            <div class="tm-info">
                <div class="tm-name">Mahamed Osama</div>
                <div class="tm-links">
                    <a class="tm-link" href="https://www.linkedin.com/in/mahamed-osama-80b4ab3b1/" target="_blank">LinkedIn</a>
                    <a class="tm-link" href="https://github.com/mahamadj99" target="_blank">GitHub</a>
                </div>
            </div>
        </div>
        <div class="tm-row">
            <div class="tm-avatar">SS</div>
            <div class="tm-info">
                <div class="tm-name">Sally Sobhy</div>
            </div>
        </div>
        <div class="tm-row">
            <div class="tm-avatar">NA</div>
            <div class="tm-info">
                <div class="tm-name">Nour Ahmed</div>
            </div>
        </div>
    </div>
    """

    import streamlit.components.v1 as components
    components.html(TEAM_HTML, height=175, scrolling=False)

    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN — Header
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="page-header">
    <div class="page-header-logo">{ZODIAC_LOGO_SVG}</div>
    <h1 class="page-title">Airbnb <span>Price Category</span> Predictor</h1>
    <p class="page-subtitle">New York City · Machine Learning · 2019 Dataset</p>
    <div class="header-line"></div>
</div>
""", unsafe_allow_html=True)

# Model load error
if load_error:
    st.markdown(f"""
    <div class="alert-error">
        <strong>Model Error</strong>
        {load_error} — Make sure airbnb_model.pkl and airbnb_preprocessor.pkl are present.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
#  FORM
# ─────────────────────────────────────────────

# ── Location ──
st.markdown('<div class="form-section-label">Location</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="medium")
with col1:
    neighbourhood_group = st.selectbox(
        "Borough",
        list(NEIGHBOURHOODS.keys())
    )
with col2:
    neighbourhood_options = NEIGHBOURHOODS[neighbourhood_group]
    neighbourhood_input = st.selectbox(
        "Neighbourhood",
        options=["— Select or type —"] + sorted(neighbourhood_options),
        index=0
    )
    custom_neighbourhood = st.text_input(
        "Or type a custom neighbourhood",
        placeholder="e.g. Astoria, Crown Heights..."
    )

# Resolve neighbourhood value
if custom_neighbourhood.strip():
    neighbourhood_value = custom_neighbourhood.strip()
elif neighbourhood_input != "— Select or type —":
    neighbourhood_value = neighbourhood_input
else:
    neighbourhood_value = ""

# ── Coordinates ──
st.markdown('<div class="form-section-label">Coordinates</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="medium")
with col1:
    latitude = st.number_input("Latitude", value=40.7128, format="%.4f", help="NYC range: 40.49 – 40.91")
with col2:
    longitude = st.number_input("Longitude", value=-74.0060, format="%.4f", help="NYC range: -74.25 – -73.70")

# ── Room Details ──
st.markdown('<div class="form-section-label">Room Details</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    room_type = st.selectbox(
        "Room Type",
        ["Entire home/apt", "Private room", "Shared room"]
    )
with col2:
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=3)
with col3:
    availability_365 = st.number_input("Availability (days/year)", min_value=0, max_value=365, value=100)

# ── Reviews & Host ──
st.markdown('<div class="form-section-label">Reviews & Host</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=10)
with col2:
    reviews_per_month = st.number_input("Reviews Per Month", min_value=0.0, value=1.0, format="%.2f")
with col3:
    calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, value=1)

# ─────────────────────────────────────────────
#  VALIDATION
# ─────────────────────────────────────────────
def validate_inputs():
    errors, warnings = [], []
    if not neighbourhood_value:
        errors.append("Neighbourhood is required — select from the list or type a custom value.")
    if not (40.49 <= latitude <= 40.91):
        warnings.append(f"Latitude {latitude:.4f} is outside the NYC range (40.49 – 40.91). Prediction accuracy may be reduced.")
    if not (-74.25 <= longitude <= -73.70):
        warnings.append(f"Longitude {longitude:.4f} is outside the NYC range (-74.25 – -73.70). Prediction accuracy may be reduced.")
    if minimum_nights > 180:
        warnings.append("Minimum nights above 180 is uncommon and may affect prediction accuracy.")
    return errors, warnings

# ─────────────────────────────────────────────
#  CONFIDENCE HELPER
# ─────────────────────────────────────────────
def confidence_status(conf):
    if conf >= 0.85: return "High confidence"
    if conf >= 0.70: return "Moderate confidence"
    if conf >= 0.55: return "Fair confidence"
    return "Low confidence — consider reviewing inputs"

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

if st.button("Predict Price Category"):
    errors, warnings = validate_inputs()

    if errors:
        for e in errors:
            st.markdown(f'<div class="alert-error"><strong>Input Required</strong>{e}</div>', unsafe_allow_html=True)
    else:
        for w in warnings:
            st.markdown(f'<div class="alert-warning"><strong>Notice</strong>{w}</div>', unsafe_allow_html=True)

        try:
            input_df = pd.DataFrame({
                "neighbourhood_group": [neighbourhood_group],
                "neighbourhood": [neighbourhood_value.lower().strip()],
                "room_type": [room_type],
                "latitude": [latitude],
                "longitude": [longitude],
                "minimum_nights": [minimum_nights],
                "number_of_reviews": [number_of_reviews],
                "reviews_per_month": [reviews_per_month],
                "calculated_host_listings_count": [calculated_host_listings_count],
                "availability_365": [availability_365]
            })

            processed = preprocessor.transform(input_df)
            prediction = model.predict(processed)[0]

            # Confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(processed)[0]
                confidence = float(np.max(proba))
                conf_pct = int(confidence * 100)
                conf_status_text = confidence_status(confidence)
            else:
                conf_pct = None
                conf_status_text = "Confidence score unavailable for this model type."

            # Build confidence block HTML
            if conf_pct is not None:
                conf_html = f"""
                <div style="margin-top:16px;">
                    <div style="font-size:10px;font-weight:600;letter-spacing:2px;text-transform:uppercase;
                                color:#6a8fa8;margin-bottom:8px;">Confidence Score</div>
                    <div style="background:rgba(0,180,216,0.1);border-radius:20px;height:6px;
                                max-width:280px;margin:0 auto 10px auto;overflow:hidden;
                                border:1px solid rgba(0,180,216,0.12);">
                        <div style="height:100%;width:{conf_pct}%;border-radius:20px;
                                    background:linear-gradient(90deg,#00B4D8,#90E0EF);"></div>
                    </div>
                    <div style="font-family:'Space Grotesk',sans-serif;font-size:22px;
                                font-weight:700;color:#00B4D8;">{conf_pct}%</div>
                    <div style="font-size:12px;color:#6a8fa8;margin-top:4px;
                                letter-spacing:0.3px;">{conf_status_text}</div>
                </div>
                """
            else:
                conf_html = f"""
                <div style="font-size:12px;color:#6a8fa8;margin-top:12px;">{conf_status_text}</div>
                """

            result_html = f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap');
            </style>
            <div style="
                background: linear-gradient(160deg,#0a2540 0%,#0D1B2A 100%);
                border: 1px solid rgba(0,180,216,0.25);
                border-radius: 14px;
                padding: 32px 24px;
                text-align: center;
                position: relative;
                overflow: hidden;
                margin-top: 8px;
            ">
                <div style="
                    position:absolute;top:0;left:0;right:0;height:2px;
                    background:linear-gradient(90deg,transparent,#00B4D8,transparent);
                "></div>
                <div style="font-size:10px;font-weight:600;letter-spacing:3px;
                            text-transform:uppercase;color:#00B4D8;margin-bottom:10px;">
                    Predicted Price Category
                </div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:38px;
                            font-weight:700;color:#FFFFFF;margin:0 0 4px 0;line-height:1.1;">
                    {prediction}
                </div>
                {conf_html}
            </div>
            """

            import streamlit.components.v1 as components
            components.html(result_html, height=260, scrolling=False)

        except ValueError as ve:
            st.markdown(f'<div class="alert-error"><strong>Input Error</strong>{ve}</div>', unsafe_allow_html=True)
        except AttributeError as ae:
            st.markdown(f'<div class="alert-error"><strong>Model Error</strong>{ae}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="alert-error"><strong>Unexpected Error</strong>{str(e)}</div>', unsafe_allow_html=True)

# ── Footer ──
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; border-top:1px solid rgba(0,180,216,0.1); padding-top:16px;">
    <p style="font-size:11px; color:#2a4a62 !important; letter-spacing:1px; text-transform:uppercase; margin:0;">
        The Zodiac &nbsp;·&nbsp; NYC Airbnb 2019 &nbsp;·&nbsp; Scikit-learn + Streamlit
    </p>
</div>
""", unsafe_allow_html=True)