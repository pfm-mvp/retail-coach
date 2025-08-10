
import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st

from radar.data_client import fetch_report
from radar.normalizer import normalize
from radar.actions_engine import generate_actions
from radar.best_practice import top_performers
from radar.demographics import has_gender_data, gender_insights

PFM_PURPLE = "#762181"
PFM_RED = "#F04438"

st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")

# Sidebar
st.sidebar.header("⚙️ Instellingen")
today = dt.date.today()
preset = st.sidebar.selectbox("Periode", ["7 dagen","30 dagen","13 weken","Custom"], index=0)
if preset == "7 dagen":
    d_from, d_to = today - dt.timedelta(days=7), today - dt.timedelta(days=1)
elif preset == "30 dagen":
    d_from, d_to = today - dt.timedelta(days=30), today - dt.timedelta(days=1)
elif preset == "13 weken":
    d_from, d_to = today - dt.timedelta(weeks=13), today - dt.timedelta(days=1)
else:
    c1, c2 = st.sidebar.columns(2)
    d_from = c1.date_input("Van", value=today - dt.timedelta(days=7))
    d_to = c2.date_input("Tot", value=today - dt.timedelta(days=1))

gran = st.sidebar.radio("Granulariteit", ["Dag","Uur"], horizontal=True)
period_step = "day" if gran=="Dag" else "hour"

st.sidebar.markdown("---")
conv_target_pct = st.sidebar.slider("Conversiedoel (%)", 5, 50, 20, 1)
spv_uplift_pct = st.sidebar.slider("SPV‑uplift t.o.v. eigen gem. (%)", 5, 50, 10, 5)
csm2i_target = st.sidebar.slider("CSm²I‑target", 0.80, 1.20, 1.00, 0.05)

st.sidebar.markdown("---")
shops_str = st.sidebar.text_input("Shop IDs (komma‑gescheiden)", value="")
analyze = st.sidebar.button("🔍 Analyseer Regio", type="primary")

# Header
st.markdown("## 📊 Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# Debug
with st.expander("🛠️ Debug"):
    st.write("POST /get-report met query params (geen []).")
    st.json({"period_step": period_step, "from": str(d_from), "to": str(d_to)})

API_URL = st.secrets.get("API_URL", "")
if not API_URL:
    st.warning("Stel `API_URL` in via st.secrets om live data op te halen.")
    st.stop()

def parse_shops(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip().isdigit()]

if analyze:
    shop_ids = parse_shops(shops_str)
    if not shop_ids:
        st.info("Voer 1 of meer shop IDs in.")
        st.stop()

    # Data outputs (incl. sqm & demographics)
    outs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sqm_meter",
            "demographics_gender_male","demographics_gender_female"]

    resp = fetch_report(
        api_url=API_URL,
        shop_ids=shop_ids,
        data_outputs=outs,
        date_from=str(d_from),
        date_to=str(d_to),
        period_step=period_step,
    )
    df = normalize(resp, ts_key="timestamp")

    if df.empty:
        st.info("Geen data voor de gekozen filters.")
        st.stop()

    # ====== Next Best Action
    actions = generate_actions(df, conv_target=conv_target_pct/100.0, spv_uplift=spv_uplift_pct/100.0, csm2i_target=csm2i_target)
    st.markdown("### ✅ Next Best Action")
    if actions.empty:
        st.success("Geen urgente issues gevonden. 🎉")
    else:
        total_impact = actions["expected_impact_eur"].sum()
        st.markdown(f"**Totale verwachte impact:** €{total_impact:,.0f}".replace(",", "X").replace(".", ",").replace("X","."))
        for sid, sdf in actions.groupby("shop_id", sort=False):
            sname = sdf["shop_name"].iloc[0]
            st.markdown(f"#### 🏬 {sname}")
            fig = px.bar(sdf.head(8), x="issue", y="expected_impact_eur", title=f"Top issues op impact – {sname}")
            fig.update_traces(marker_color=PFM_PURPLE)
            fig.update_layout(yaxis_tickprefix="€", showlegend=False, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

    # ====== Best Practice Finder
    st.markdown("### 🏆 Best Practice Finder")
    base = df.copy()
    # Ensure SPV exists and normalized
    if "sales_per_visitor" not in base.columns:
        base["sales_per_visitor"] = (base.get("turnover",0) / (base.get("count_in",0)+1e-9))
    top = top_performers(base, kpi="sales_per_visitor", n=5)
    if top.empty:
        st.info("Onvoldoende data voor Best Practices.")
    else:
        st.dataframe(top.rename(columns={"sales_per_visitor_avg": "SPV (gem)"}), use_container_width=True)

    # ====== Demografie (optioneel)
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
    # else: compleet verbergen (geen melding)
else:
    st.info("Kies je periode, granulariteit en shops en klik **Analyseer Regio**.")
