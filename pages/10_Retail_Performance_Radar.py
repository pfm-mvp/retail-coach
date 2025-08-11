# pages/10_Retail_Performance_Radar.py
# ──────────────────────────────────────────────────────────────────────────────
# Retail Performance Radar – Next Best Actions, KPI‑tegels, scatter en
# aanbevelingskaarten per winkel. PFM‑styling, nette getalnotaties.
# ──────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")

PFM_RED    = "#F04438"
PFM_PURPLE = "#9E77ED"
PFM_ORANGE = "#FEAC76"
PFM_EMBER  = "#F59E0B"
PFM_GREEN  = "#16A34A"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background-color: %s !important; color: #fff !important; }
button[data-testid="stBaseButton-secondary"]{
  background:%s !important;color:#fff !important;border-radius:16px !important;
  font-weight:600 !important;border:none !important;padding:.6rem 1.4rem !important;
}
button[data-testid="stBaseButton-secondary"]:hover{ background:#d13c30 !important; }
.card{border:1px solid #eee;border-radius:12px;padding:14px 16px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04);}
.kpi{font-size:1.2rem;font-weight:700;font-variant-numeric:tabular-nums;}
.badge{display:inline-block;padding:2px 8px;border-radius:12px;background:#F1F5F9;color:#111827;font-size:.8rem}
.note{color:#6b7280;font-size:.85rem}
</style>
""" % (PFM_PURPLE, PFM_RED), unsafe_allow_html=True)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(v):
    try: return ("€{:,.0f}".format(float(v))).replace(",", "X").replace(".", ",").replace("X",".")
    except: return "€0"

def fmt_eur2(v):
    try: return ("€{:,.2f}".format(float(v))).replace(",", "X").replace(".", ",").replace("X",".")
    except: return "€0,00"

def normalize_kpis(df):
    out = df.copy()
    for c in ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    if "conversion_rate" in out and not out["conversion_rate"].empty:
        if out["conversion_rate"].max()>1.5:
            out["conversion_rate"] = out["conversion_rate"]/100.0
    else:
        out["conversion_rate"] = out.get("transactions",0.0)/(out.get("count_in",0.0)+EPS)
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover",0.0)/(out.get("count_in",0.0)+EPS)
    out["atv"] = out.get("turnover",0.0)/(out.get("transactions",0.0)+EPS)
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0,np.nan).median()
        fallback = med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0,np.nan).fillna(fallback)
    else:
        out["sq_meter"]=DEFAULT_SQ_METER
    return out

def choose_ref_spv(df, uplift_pct=0.0):
    safe = df.copy()
    for c in ["turnover","count_in"]:
        if c in safe.columns:
            safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0.0)
        else:
            safe[c] = 0.0
    visitors = float(safe["count_in"].sum()); turnover = float(safe["turnover"].sum())
    base = 0.0 if visitors<=0 else turnover/(visitors+EPS)
    return max(0.0, base) * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"]/(out["visitors"]+EPS)
    out["csm2i"]      = out["actual_spv"]/(ref_spv+EPS)
    out["visitors_per_sqm"] = out["count_in"]/(out["sq_meter"]+EPS)
    out["actual_spsqm"]     = out["turnover"] /(out["sq_meter"]+EPS)
    out["expected_spsqm"]   = ref_spv * out["visitors_per_sqm"]
    out["uplift_eur_csm"]   = np.maximum(0.0, out["visitors"]*(csm2i_target*ref_spv - out["actual_spv"]))
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Shop mapping
# ──────────────────────────────────────────────────────────────────────────────
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k,v in _MAP.items() if str(v).strip()}
NAME_TO_ID       = {v:k for k,v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# ──────────────────────────────────────────────────────────────────────────────
# UI – periode, granulariteit, winkels, targets
# ──────────────────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1,1,1])
with c1:
    period_label = st.selectbox("Periode", ["7 dagen","30 dagen","last_month"], index=1)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag","Uur"], index=0)
with c3:
    proj_toggle = st.toggle("Toon projectie resterend jaar", value=True)

c4, c5, c6 = st.columns([1,1,1])
with c4:
    conv_goal_pct = st.slider("Conversiedoel (%)", 1, 80, 20, 1)
with c5:
    spv_uplift_pct = st.slider("SPV‑uplift (%)", 0, 100, 10, 1)
with c6:
    csm2i_target = st.slider("CSm²I‑target", 0.10, 2.00, 1.00, 0.05)

st.markdown("### Selecteer winkels")
selected_names = st.multiselect("Selecteer winkels", names, default=names[:5], placeholder="Kies 1 of meer winkels…")
shop_ids = [NAME_TO_ID[n] for n in selected_names]

analyze = st.button("🔍 Analyseer", type="secondary")

# ──────────────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────────────
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)), ("period_step",step)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = (resp or {}).get("data", {})
    for _, shops in data.items():
        for sid, payload in (shops or {}).items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in (dates or {}).items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date; df["hour"] = ts.dt.hour
    return df

# ──────────────────────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────────────────────
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    # Periode
    today = date.today()
    if period_label == "last_month":
        first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last  = today.replace(day=1) - timedelta(days=1)
        date_from, date_to = first, last
    elif period_label == "30 dagen":
        date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)
    else:
        date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar voor de gekozen instellingen."); st.stop()

    # Referentie‑SPV
    ref_spv = choose_ref_spv(df, uplift_pct=spv_uplift_pct/100.0)

    # Uniforme CSm²I & uplift
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # Conversie‑uplift
    conv_target = float(conv_goal_pct)/100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # Aggregatie per winkel
    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csi=("csm2i","mean"),
        spv=("actual_spv","mean"),
        conv=("conversion_rate","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # KPI‑tegels
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div>
                    <div class="kpi">{fmt_eur(agg["uplift_csm"].sum())}</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div>
                    <div class="kpi">{fmt_eur(agg["uplift_conv"].sum())}</div></div>""", unsafe_allow_html=True)
    total_now = agg["uplift_total"].sum()
    k3.markdown(f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
                    <div class="kpi">{fmt_eur(total_now)}</div></div>""", unsafe_allow_html=True)

    # Projectie resterend jaar (kleine badge rechtsboven tegels)
    if proj_toggle:
        days_len  = max(1,(date_to - date_from).days+1)
        days_left = (date(today.year,12,31) - today).days
        per_day   = total_now / days_len
        proj_year = per_day * max(0, days_left)
        st.caption(f"📈 Projectie resterend jaar: **{fmt_eur(proj_year)}**")

    # Scatter – SPV vs Sales per m² (kleur tov targetband)
    rad = df.groupby(["shop_id","shop_name"]).agg(
        spv=("actual_spv","mean"),
        spsqm=("actual_spsqm","mean"),
        csi=("csm2i","mean"),
        visitors=("count_in","sum"),
    ).reset_index()
    size_series = agg.set_index("shop_id")["uplift_total"]
    rad["size_metric"] = rad["shop_id"].map(size_series).fillna(0.0)
    low_thr, high_thr = float(csm2i_target)*0.95, float(csm2i_target)*1.05
    rad["band"] = np.select([rad["csi"]<low_thr, rad["csi"]>high_thr], ["Onder target","Boven target"], default="Rond target")
    color_map  = {"Onder target": PFM_RED, "Rond target": PFM_EMBER, "Boven target": PFM_GREEN}
    symbol_map = {"Onder target":"diamond","Rond target":"circle","Boven target":"square"}
    rad["hover_spv"]   = rad["spv"].round(2).apply(fmt_eur2)
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(fmt_eur2)
    rad["hover_csi"]   = rad["csi"].round(2).map(lambda v: str(v).replace(".", ","))
    rad["hover_uplift"]= rad["size_metric"].round(0).apply(fmt_eur)

    scatter = px.scatter(
        rad, x="spv", y="spsqm", size="size_metric",
        color="band", symbol="band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv","hover_spsqm","hover_csi","hover_uplift"],
        labels={"spv":"Sales per Visitor (€/bezoeker)","spsqm":"Sales per m² (€/m²)","band":"CSm²I t.o.v. target"},
    )
    scatter.update_traces(
        text=rad["shop_name"],
        hovertemplate="<b>%{text}</b><br>SPV: %{customdata[0]}<br>Sales per m²: %{customdata[1]}<br>CSm²I: %{customdata[2]}<br>Uplift: %{customdata[3]}<extra></extra>"
    )
    scatter.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=560)
    st.plotly_chart(scatter, use_container_width=True)

    # Aanbevelingskaarten per winkel (niet invasief, alleen toevoeging)
    st.markdown("### Aanbevelingen per winkel")
    for _, r in agg.sort_values("uplift_total", ascending=False).iterrows():
        spv_vs_port  = r["spv"]/max(EPS, df["turnover"].sum()/max(EPS, df["count_in"].sum()))
        csi_badge    = f"<span class='badge'>CSm²I: {str(round(r['csi'],2)).replace('.',',')}</span>"
        conv_badge   = f"<span class='badge'>Conv: {str(round(r['conv']*100,1)).replace('.',',')}%</span>"
        st.markdown(
            f"""
            <div class="card">
              <div><b>{r['shop_name']}</b> &nbsp; {csi_badge} &nbsp; {conv_badge}</div>
              <div class="note">Potentiële uplift: <b>{fmt_eur(r['uplift_total'])}</b> 
              (CSm²I: {fmt_eur(r['uplift_csm'])}, Conversie: {fmt_eur(r['uplift_conv'])})</div>
              <ul>
                <li>CSm²I <b>{'onder' if r['csi']<csm2i_target else 'op/ boven'}</b> target → 
                    { 'train upsell/cross‑sell, focus op SPV' if r['csi']<csm2i_target else 'borg best practices' }.</li>
                <li>Conversie richting doel {conv_goal_pct}% → 
                    { 'extra bezetting piekuren & active promo bij instap' if r['conv']<conv_target else 'handhaaf performance' }.</li>
                <li>Sales per m²: {fmt_eur2(r['spsqm'])} – kijk naar <i>win‑store</i> playbook voor inspiratie.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Debug
    with st.expander("🛠️ Debug"):
        st.json({
            "period_step": step, "from": str(date_from), "to": str(date_to),
            "shop_ids": shop_ids, "ref_spv": float(ref_spv),
            "csm2i_target": float(csm2i_target), "conv_goal_pct": int(conv_goal_pct)
        })
