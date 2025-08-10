import os, sys, datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st

# --- Projectroot op pad zetten (één niveau boven /pages) ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from radar.data_client import fetch_report
from radar.normalizer import normalize
from radar.actions_engine import generate_actions
from radar.best_practice import top_performers
from radar.demographics import has_gender_data, gender_insights

# --- ROBUST import van shop_mapping.py met fallback ---
SHOP_NAME_MAP = {}
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # exact bestandsnaam/variabelenaam
    if isinstance(_MAP, dict):
        SHOP_NAME_MAP = _MAP
except Exception:
    # Hard fallback: laad handmatig vanaf pad (dekt edge-cases in Streamlit runner)
    import importlib.util, pathlib
    sm_path = pathlib.Path(ROOT) / "shop_mapping.py"
    if sm_path.exists():
        spec = importlib.util.spec_from_file_location("shop_mapping", sm_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        SHOP_NAME_MAP = getattr(mod, "SHOP_NAME_MAP", {}) or {}

st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")

# --- Forceer PFM-rode knop (alle st.button) ---
st.markdown("""
<style>
div.stButton > button {
  background-color:#F04438 !important;
  color:white !important;
  border:0 !important;
  border-radius:8px !important;
}
div.stButton > button:hover { filter: brightness(0.95); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
## 📊 Retail Performance Radar
Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen
""")

PFM_PURPLE = "#762181"

# Controls row 1
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.0, 1.0])
with c1:
    preset = st.selectbox("Periode", ["7 dagen","30 dagen","13 weken","Custom"], index=0)
today = dt.date.today()
if preset == "7 dagen":
    d_from, d_to = today - dt.timedelta(days=7), today - dt.timedelta(days=1)
elif preset == "30 dagen":
    d_from, d_to = today - dt.timedelta(days=30), today - dt.timedelta(days=1)
elif preset == "13 weken":
    d_from, d_to = today - dt.timedelta(weeks=13), today - dt.timedelta(days=1)
else:
    d_from = st.date_input("Van", value=today - dt.timedelta(days=7), key="from")
    d_to = st.date_input("Tot", value=today - dt.timedelta(days=1), key="to")
with c2:
    gran = st.selectbox("Granulariteit", ["Dag","Uur"], index=0)
period_step = "day" if gran == "Dag" else "hour"
with c3:
    conv_target_pct = st.slider("Conversiedoel (%)", 5, 50, 20, 1)
    spv_uplift_pct = st.slider("SPV‑uplift (%)", 5, 50, 10, 5)
with c4:
    csm2i_target = st.slider("CSm²I‑target", 0.80, 1.20, 1.00, 0.05)

# Controls row 2 — multi‑select stores from mapping
c5, c6 = st.columns([2, 1])
with c5:
    if SHOP_NAME_MAP:
        opts = sorted([(name, sid) for sid, name in SHOP_MAP_NAME.items()], key=lambda x: x[0].lower())
        names = [o[0] for o in opts]
        selected_names = st.multiselect("Selecteer winkels", names, default=names[:1])
        name_to_id = {name: sid for name, sid in opts}
        shop_ids = [name_to_id[n] for n in selected_names]
    else:
        ids_text = st.text_input("Shop IDs (komma‑gescheiden)", value="")
        shop_ids = [int(x.strip()) for x in ids_text.split(",") if x.strip().isdigit()]
with c6:
    analyze = st.button("🔍 Analyseer", type="secondary", use_container_width=True, key="an_btn", help="Voer analyse uit", on_click=None)
    # Force the button to PFM red via a wrapper div
    st.markdown('<div class="pfm-red"></div>', unsafe_allow_html=True)

# Debug
with st.expander("🛠️ Debug"):
    st.json({"period_step": period_step, "from": str(d_from), "to": str(d_to), "shop_ids": shop_ids})

API_URL = st.secrets.get("API_URL", "")
if not API_URL:
    st.warning("Stel `API_URL` in via .streamlit/secrets.toml")
    st.stop()

if analyze:
    if not shop_ids:
        st.info("Selecteer minimaal één winkel.")
        st.stop()
    outs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sqm_meter",
            "demographics_gender_male","demographics_gender_female"]
    resp = fetch_report(api_url=API_URL, shop_ids=shop_ids, data_outputs=outs,
                        date_from=str(d_from), date_to=str(d_to), period_step=period_step)
    df = normalize(resp, ts_key="timestamp")
    if df.empty:
        st.info("Geen data voor de gekozen filters.")
        st.stop()

    # Map namen vanuit shop_mapping
    if SHOP_NAME_MAP:
        df["shop_name"] = df["shop_id"].map(SHOP_NAME_MAP).fillna(df.get("shop_name",""))

    # Next Best Action
    st.markdown("### ✅ Next Best Action")
    actions = generate_actions(df, conv_target=conv_target_pct/100.0, spv_uplift=spv_uplift_pct/100.0, csm2i_target=csm2i_target)
    if actions.empty:
        st.success("Geen urgente issues gevonden. 🎉")
    else:
        total = actions["expected_impact_eur"].sum()
        st.markdown(f"**Totale verwachte impact:** €{total:,.0f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
        import plotly.express as px
        for sid, sdf in actions.groupby("shop_id", sort=False):
            sname = sdf["shop_name"].iloc[0]
            fig = px.bar(sdf.head(8), x="issue", y="expected_impact_eur", title=f"Top issues op impact – {sname}")
            fig.update_traces(marker_color=PFM_PURPLE)
            fig.update_layout(yaxis_tickprefix="€", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Best Practices
    st.markdown("### 🏆 Best Practice Finder")
    base = df.copy()
    if "sales_per_visitor" not in base.columns:
        base["sales_per_visitor"] = (base.get("turnover",0) / (base.get("count_in",0)+1e-9))
    from radar.best_practice import top_performers
    top = top_performers(base, kpi="sales_per_visitor", n=5)
    if top.empty:
        st.info("Onvoldoende data voor Best Practices.")
    else:
        st.dataframe(top.rename(columns={"sales_per_visitor_avg": "SPV (gem)"}), use_container_width=True)

    # Demografie — alleen als data
    if has_gender_data(df):
        st.markdown("### 👥 Demografiepatronen (optioneel)")
        gi = gender_insights(df, top_n=3)
        if not gi.empty:
            gi["male_share"] = (gi["male_share"]*100).round(1)
            gi["female_share"] = (gi["female_share"]*100).round(1)
            gi["conv"] = (gi["conv"]*100).round(2)
            st.dataframe(gi.rename(columns={
                "period":"Tijdvak","male_share":"% Man","female_share":"% Vrouw","conv":"Conversie (%)","spv":"SPV (€)"
            }), use_container_width=True)
else:
    st.info("Selecteer winkels en klik **Analyseer**.")
