# pages/10_Retail_Performance_Radar.py
import os, sys, datetime as dt
import requests
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------- Path setup ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------- Imports uit radar/ ----------
from radar.normalizer import normalize
from radar.actions_engine import generate_actions
from radar.best_practice import top_performers
from radar.demographics import has_gender_data, gender_insights  # optioneel, we vragen geen demodata op

# ---------- Page config & styling ----------
st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")

# ✅ Styling: paarse pills & rode knop (jouw CSS)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
    [data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
    button[data-testid="stBaseButton-secondary"] {
        background-color: #F04438 !important; color: white !important;
        border-radius: 16px !important; font-weight: 600 !important;
        font-family: "Instrument Sans", sans-serif !important;
        padding: 0.6rem 1.4rem !important; border: none !important; box-shadow: none !important;
        transition: background-color 0.2s ease-in-out;
    }
    button[data-testid="stBaseButton-secondary"]:hover { background-color: #d13c30 !important; cursor: pointer; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("## 📊 Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

PFM_PURPLE = "#762181"

# ---------- Shop mapping: jouw SHOP_NAME_MAP (id -> naam) ----------
SHOP_ID_TO_NAME, NAME_TO_ID = {}, {}

try:
    from shop_mapping import SHOP_NAME_MAP as _MAP_ID_TO_NAME  # {id:int: "Naam":str}
except Exception:
    _MAP_ID_TO_NAME = {}

try:
    from shop_mapping import SHOP_ID_TO_NAME as _ALT_ID_TO_NAME  # fallback
except Exception:
    _ALT_ID_TO_NAME = {}

if _MAP_ID_TO_NAME:
    SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP_ID_TO_NAME.items() if str(v).strip()}
elif _ALT_ID_TO_NAME:
    SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _ALT_ID_TO_NAME.items() if str(v).strip()}
else:
    SHOP_ID_TO_NAME = {}

NAME_TO_ID = {name: sid for sid, name in SHOP_ID_TO_NAME.items()}

# ---------- Inputs: rij 1 ----------
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
with c1:
    preset = st.selectbox("Periode", ["7 dagen", "30 dagen", "13 weken", "Custom"], index=0)

today = dt.date.today()
if preset == "7 dagen":
    d_from, d_to = today - dt.timedelta(days=7), today - dt.timedelta(days=1)
elif preset == "30 dagen":
    d_from, d_to = today - dt.timedelta(days=30), today - dt.timedelta(days=1)
elif preset == "13 weken":
    d_from, d_to = today - dt.timedelta(weeks=13), today - dt.timedelta(days=1)
else:
    d_from = st.date_input("Van", value=today - dt.timedelta(days=7), key="from")
    d_to   = st.date_input("Tot", value=today - dt.timedelta(days=1), key="to")

with c2:
    gran = st.selectbox("Granulariteit", ["Dag", "Uur"], index=0)
period_step = "day" if gran == "Dag" else "hour"

with c3:
    conv_target_pct = st.slider("Conversiedoel (%)", 5, 50, 20, 1)
    spv_uplift_pct  = st.slider("SPV‑uplift (%)", 5, 50, 10, 5)
with c4:
    csm2i_target = st.slider("CSm²I‑target", 0.80, 1.20, 1.00, 0.05)

# ---------- Inputs: rij 2 (shops) ----------
if NAME_TO_ID:
    names = sorted(NAME_TO_ID.keys(), key=lambda s: s.lower())
    selected_names = st.multiselect("Selecteer winkels", names, default=names[:1], key="shop_selector")
    shop_ids = [NAME_TO_ID[n] for n in selected_names]
else:
    st.warning("Geen geldige shops gevonden in shop_mapping.py (verwacht: SHOP_NAME_MAP = {id:int: 'Naam':str}).")
    shop_ids = []

# ---------- Analyseer-knop (links, onder inputs) ----------
analyze = st.button("🔍 Analyseer", key="analyze_button", type="secondary")

# ---------- Debug ----------
with st.expander("🛠️ Debug"):
    st.json({"period_step": period_step, "from": str(d_from), "to": str(d_to), "shop_ids": shop_ids})

# ---------- API-url check ----------
API_URL = st.secrets.get("API_URL", "")
if not API_URL:
    st.warning("Stel `API_URL` in via **.streamlit/secrets.toml** (bijv. https://…/get-report).")
    st.stop()

# ---------- Inline fetch: POST DIRECT naar API_URL met data[] en data_output[] ----------
def fetch_report_inline(api_url: str, shop_ids, data_outputs, date_from: str, date_to: str, period_step: str = "day", timeout: int = 60):
    if not shop_ids:
        raise ValueError("shop_ids empty")
    if not data_outputs:
        raise ValueError("data_outputs empty")

    params = {
        "source": "shops",
        "period": "date",
        "form_date_from": date_from,
        "form_date_to": date_to,
        "period_step": period_step,
        # let op []-sleutels
        "data[]": [int(s) for s in shop_ids],
        "data_output[]": list(data_outputs),
    }
    r = requests.post(api_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---------- Run analyse ----------
if analyze:
    if not shop_ids:
        st.info("Selecteer minimaal één winkel.")
        st.stop()

    # Geen demografie in deze call
    outs = [
        "count_in", "transactions", "turnover",
        "conversion_rate", "sales_per_visitor", "sq_meter",
    ]

    with st.spinner("Analyseren…"):
        try:
            resp = fetch_report_inline(
                api_url=API_URL,
                shop_ids=shop_ids,
                data_outputs=outs,
                date_from=str(d_from),
                date_to=str(d_to),
                period_step=period_step,
            )
        except Exception as e:
            st.error(f"Fout bij ophalen data: {e}")
            st.stop()

    df = normalize(resp, ts_key="timestamp")
    if df.empty:
        st.info("Geen data voor de gekozen filters/periode.")
        st.stop()

    # Namen mappen
    if SHOP_ID_TO_NAME:
        df["shop_name"] = df["shop_id"].map(SHOP_ID_TO_NAME).fillna(df.get("shop_name", ""))

    # ---------- Next Best Action ----------
    st.markdown("### ✅ Next Best Action")
    actions = generate_actions(
        df,
        conv_target=conv_target_pct / 100.0,
        spv_uplift=spv_uplift_pct / 100.0,
        csm2i_target=csm2i_target,
    )
    if actions.empty:
        st.success("Geen urgente issues gevonden. 🎉")
    else:
        total = actions["expected_impact_eur"].sum()
        st.markdown(
            f"**Totale verwachte impact:** €{total:,.0f}"
            .replace(",", "X").replace(".", ",").replace("X", ".")
        )
        for sid, sdf in actions.groupby("shop_id", sort=False):
            sname = sdf["shop_name"].iloc[0]
            fig = px.bar(
                sdf.head(8), x="issue", y="expected_impact_eur",
                title=f"Top issues op impact – {sname}"
            )
            fig.update_traces(marker_color=PFM_PURPLE)
            fig.update_layout(yaxis_tickprefix="€", showlegend=False,
                              margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ---------- Best Practice Finder ----------
    st.markdown("### 🏆 Best Practice Finder")
    base = df.copy()
    if "sales_per_visitor" not in base.columns:
        base["sales_per_visitor"] = (base.get("turnover", 0) / (base.get("count_in", 0) + 1e-9))
    top = top_performers(base, kpi="sales_per_visitor", n=5)
    if top.empty:
        st.info("Onvoldoende data voor Best Practices.")
    else:
        st.dataframe(
            top.rename(columns={"sales_per_visitor_avg": "SPV (gem)"}),
            use_container_width=True
        )

    # ---------- Demografie (optioneel; niet opgevraagd in deze call) ----------
    if has_gender_data(df):
        st.markdown("### 👥 Demografiepatronen (optioneel)")
        gi = gender_insights(df, top_n=3)
        if not gi.empty:
            gi["male_share"]   = (gi["male_share"] * 100).round(1)
            gi["female_share"] = (gi["female_share"] * 100).round(1)
            gi["conv"]         = (gi["conv"] * 100).round(2)
            st.dataframe(gi.rename(columns={
                "period": "Tijdvak", "male_share": "% Man", "female_share": "% Vrouw",
                "conv": "Conversie (%)", "spv": "SPV (€)"
            }), use_container_width=True)
else:
    st.info("Selecteer winkels en klik **Analyseer**.")
