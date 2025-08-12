import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlencode

# =========================
# Page & styling
# =========================
st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
button[data-testid="stBaseButton-secondary"] {
  background-color: #F04438 !important; color: white !important; border-radius: 16px !important;
  font-weight: 600 !important; padding: 0.6rem 1.4rem !important; border: none !important;
}
button[data-testid="stBaseButton-secondary"]:hover { background-color: #d13c30 !important; cursor: pointer; }

.card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.kpi  { font-size: 1.2rem; font-weight: 700; }
.eur  { font-variant-numeric: tabular-nums; }

/* Oranje ‘PFM’ highlight cards */
.big-card { border:1px solid #FEAC76; background: #FFF7F2; border-radius: 12px; padding: 18px 20px;}
.big-card .title { font-weight: 700; font-size: 1.25rem; }
.big-card .value { font-weight: 800; font-size: 1.4rem; margin-top: .25rem; }

/* Badges voor aanbevelingen */
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.8rem; font-weight:600; margin-left:6px;}
.badge-green  { background:#E9F9EE; color:#14804A; }
.badge-amber  { background:#FEF3C7; color:#92400E; }
.badge-red    { background:#FEE2E2; color:#991B1B; }

/* kleine spacing helper */
.mt-8 { margin-top: 8px; }
.mt-16 { margin-top: 16px; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# =========================
# Helpers
# =========================
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

# NL weekday mapping (for tooltips/heatmap)
WEEKDAY_EN_TO_NL = {
    "Monday": "maandag",
    "Tuesday": "dinsdag",
    "Wednesday": "woensdag",
    "Thursday": "donderdag",
    "Friday": "vrijdag",
    "Saturday": "zaterdag",
    "Sunday": "zondag",
}
WEEKDAY_ORDER_EN = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
WEEKDAY_ORDER_NL = [WEEKDAY_EN_TO_NL[d] for d in WEEKDAY_ORDER_EN]


def fmt_eur(x: float) -> str:
    try:
        return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except Exception:
        return "€0"


def fmt_eur2(x: float) -> str:
    try:
        return ("€{:,.2f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except Exception:
        return "€0,00"


def fmt_int(x: float) -> str:
    try:
        return ("{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return "0"


def fmt_pct(x: float, decimals: int = 1) -> str:
    try:
        return (f"{x*100:.{decimals}f}").replace(".", ",") + "%"
    except Exception:
        return "0%"


def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out


def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, [
        "turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"
    ])
    # conversie -> fractie (0..1) indien nodig
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:  # % → fractie
            out["conversion_rate"] = out["conversion_rate"] / 100.0
    else:
        out["conversion_rate"] = out.get("transactions", 0.0) / (out.get("count_in", 0.0) + EPS)
    # SPV
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover", 0.0) / (out.get("count_in", 0.0) + EPS)
    # ATV
    out["at]()
