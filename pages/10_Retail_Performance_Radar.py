# pages/10_Retail_Performance_Radar.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

# ─────────────────────────  Page & styling  ─────────────────────────
st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
[data-baseweb="tag"] { background:#9E77ED !important; color:#fff !important; }
button[data-testid="stBaseButton-secondary"]{
  background:#F04438!important;color:#fff!important;border-radius:16px!important;
  font-weight:600!important;border:none!important;padding:0.6rem 1.4rem!important;
}
button[data-testid="stBaseButton-secondary"]:hover{background:#d13c30!important;cursor:pointer;}
.card{border:1px solid #eee;border-radius:12px;padding:14px 16px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.kpi{font-size:1.2rem;font-weight:700;font-variant-numeric:tabular-nums}
.note{color:#667085}
.block-orange{background:#FFF7ED;border:1px solid #FEAC76;border-radius:12px;padding:18px}
.badge{display:inline-block;border-radius:10px;padding:2px 8px;font-size:.78rem;color:#fff}
.badge.red{background:#F04438}.badge.amber{background:#F59E0B}.badge.green{background:#16A34A}
.lstrip{border-left:6px solid var(--clr,#F59E0B);border-radius:10px;padding:12px 14px;background:#fff}
.small{color:#667085;font-size:.85rem}
.h-gap{height:14px}
</style>
""", unsafe_allow_html=True)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# ─────────────────────────  Helpers  ─────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x):
    try: return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except: return "€0"

def coerce(df, cols):
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out.get(c,0.0), errors="coerce").fillna(0.0)
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if out["conversion_rate"].max() > 1.5:
        out["conversion_rate"] = out["conversion_rate"]/100.0
    else:
        out["conversion_rate"] = out["conversion_rate"].fillna(out["transactions"]/(out["count_in"]+EPS))
    if out["sales_per_visitor"].isna().all() or out["sales_per_visitor"].eq(0).all():
        out["sales_per_visitor"] = out["turnover"]/(out["count_in"]+EPS)
    out["atv"] = out["turnover"]/(out["transactions"]+EPS)
    sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
    med = sqm.replace(0, np.nan).median()
    out["sq_meter"] = sqm.replace(0, np.nan).fillna(med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER)
    return out

def choose_ref_spv(df: pd.DataFrame, uplift_pct: float = 0.0) -> float:
    safe = coerce(df, ["turnover","count_in"])
    visitors = float(safe["count_in"].sum())
    base = 0.0 if visitors <= 0 else float(safe["turnover"].sum())/(visitors+EPS)
    return base * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"]/(out["visitors"]+EPS)
    out["csm2i"]      = out["actual_spv"]/(ref_spv+EPS)

    out["visitors_per_sqm"] = out["count_in"]/(out["sq_meter"]+EPS)
    out["actual_spsqm"]     = out["turnover"]/(out["sq_meter"]+EPS)
    out["expected_spsqm"]   = ref_spv * out["visitors_per_sqm"]

    out["uplift_eur_csm"] = np.maximum(0.0, out["visitors"]*(csm2i_target*ref_spv - out["actual_spv"]))
    return out

# ─────────────────────────  Shop mapping  ─────────────────────────
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}

# ─────────────────────────  Inputs  ─────────────────────────
c1, c2 = st.columns([1,1])
with c1:
    period_label = st.selectbox("Periode", ["7 dagen","30 dagen","last_month"], index=1)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag","Uur"], index=0)

left, mid, right = st.columns([1,1,1])
with left:
    conv_goal_pct = st.slider("Conversiedoel (%)", 1, 80, 46, 1)
with mid:
    spv_uplift_pct = st.slider("SPV‑uplift (%)", 0, 100, 0, 1)
with right:
    csm2i_target = st.slider("CSm²I‑target", 0.10, 2.00, 1.00, 0.05)

st.markdown("### Selecteer winkels")
selected = st.multiselect("Selecteer winkels",
                          sorted(NAME_TO_ID.keys(), key=str.lower),
                          default=sorted(NAME_TO_ID.keys(), key=str.lower)[:5],
                          placeholder="Kies 1 of meer winkels…")
shop_ids = [NAME_TO_ID[n] for n in selected]

analyze = st.button("🔍 Analyseer", type="secondary")

# ─────────────────────────  API  ─────────────────────────
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"),("period","date"),
              ("form_date_from",str(dfrom)),("form_date_to",str(dto)),("period_step",step)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows=[]
    for _, shops in (resp or {}).get("data", {}).items():
        for sid, payload in (shops or {}).items():
            for ts, obj in (payload or {}).get("dates", {}).items():
                rec={"timestamp":ts,"shop_id":int(sid)}
                rec.update(((obj or {}).get("data", {})))
                rows.append(rec)
    df=pd.DataFrame(rows)
    if df.empty: return df
    ts=pd.to_datetime(df["timestamp"], errors="coerce"); df["date"]=ts.dt.date; df["hour"]=ts.dt.hour
    df["shop_name"]=df["shop_id"].map(SHOP_ID_TO_NAME).fillna(df["shop_id"].astype(str))
    return df

# ─────────────────────────  Run  ─────────────────────────
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()

    today = date.today()
    if period_label == "last_month":
        first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last  = today.replace(day=1) - timedelta(days=1)
        date_from, date_to = first, last
    elif period_label == "30 dagen":
        date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)
    else:
        date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)

    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar."); st.stop()

    # referenties en berekeningen
    ref_spv = choose_ref_spv(df, uplift_pct=spv_uplift_pct/100.0)
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)
    conv_target = float(conv_goal_pct)/100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csm2i=("csm2i","mean"),
        spv=("actual_spv","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # KPI‑tegels
    k1,k2,k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div><div class="kpi">{fmt_eur(agg["uplift_csm"].sum())}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div><div class="kpi">{fmt_eur(agg["uplift_conv"].sum())}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div><div class="kpi">{fmt_eur(agg["uplift_total"].sum())}</div></div>', unsafe_allow_html=True)

    # Oranje total widget + (optioneel) projectie
    c1, c2 = st.columns([1.25,1])
    with c1:
        st.markdown(f"""
        <div class="block-orange">
          <div style="font-weight:700;font-size:1.05rem">💰 Total extra potential in revenue</div>
          <div class="kpi" style="margin-top:4px">{fmt_eur(agg["uplift_total"].sum())}</div>
          <div class="note">Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        remain = (date(today.year,12,31) - today).days
        yearly_proj = agg["uplift_total"].sum() * max(remain,0)
        st.markdown(f"""
        <div class="block-orange">
          <div style="font-weight:700;font-size:1.05rem">📈 Projectie resterend jaar</div>
          <div class="kpi" style="margin-top:4px">{fmt_eur(yearly_proj)}</div>
          <div class="note">Huidig potentieel × resterende dagen dit jaar.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="h-gap"></div>', unsafe_allow_html=True)

    # Scatter: SPV vs Sales per m² (kleur = band t.o.v. CSm²I‑target)
    rad = agg[["shop_id","shop_name","spv","spsqm","csm2i","uplift_total","visitors"]].copy()
    low_thr  = float(csm2i_target)*0.95
    high_thr = float(csm2i_target)*1.05
    rad["band"] = np.select([rad["csm2i"] < low_thr, rad["csm2i"] > high_thr],
                            ["Onder target","Boven target"], default="Rond target")

    # hover customdata (in vaste volgorde)
    rad["hover_spv"]   = rad["spv"].round(2).map(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X","."))
    rad["hover_spsqm"] = rad["spsqm"].round(2).map(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X","."))
    rad["hover_csi"]   = rad["csm2i"].round(2).astype(str).str.replace(".", ",")
    rad["hover_upl"]   = rad["uplift_total"].map(fmt_eur)

    color_map  = {"Onder target":"#F04438","Rond target":"#F59E0B","Boven target":"#16A34A"}
    symbol_map = {"Onder target":"diamond","Rond target":"circle","Boven target":"square"}

    fig = px.scatter(rad, x="spv", y="spsqm",
                     color="band", symbol="band",
                     color_discrete_map=color_map, symbol_map=symbol_map,
                     size="uplift_total",
                     hover_data=["hover_spv","hover_spsqm","hover_csi","hover_upl"],
                     labels={"spv":"Sales per Visitor","spsqm":"Sales per m²","band":"CSm²I t.o.v. target"})
    fig.update_traces(hovertemplate="<b>%{text}</b><br>SPV: %{customdata[0]}<br>Sales per m²: %{customdata[1]}<br>CSm²I: %{customdata[2]}<br>Uplift: %{customdata[3]}<extra></extra>",
                      text=rad["shop_name"])
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=520,
                      xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
                      yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"),
                      legend_title_text="CSm²I t.o.v. target")
    st.plotly_chart(fig, use_container_width=True)

    # ───────── Aanbevelingen per winkel (incl. SPV vs Best) ─────────
    best_spv = agg.loc[agg["spv"].idxmax(), "spv"] if not agg.empty else 0.0
    st.markdown("## Aanbevelingen per winkel")
    for _, r in agg.sort_values("uplift_total", ascending=False).iterrows():
        # statuskleur
        if r["csm2i"] < low_thr:    clr, badge = "#F04438", "red"
        elif r["csm2i"] > high_thr: clr, badge = "#16A34A", "green"
        else:                       clr, badge = "#F59E0B", "amber"

        spv_gap = best_spv - r["spv"]
        spv_line = f"SPV: €{r['spv']:.2f} {'(beste: €'+format(best_spv, '.2f')+')' if best_spv>0 else ''}"
        if spv_gap > 0.05:
            spv_hint = "Focus op basket‑size & mix (SPV naar best‑practice tillen)."
        else:
            spv_hint = "SPV op niveau; focus op conversie & traffic‑benutting."

        html = f"""
        <div class="lstrip" style="--clr:{clr};margin-bottom:10px">
          <div style="display:flex;gap:10px;align-items:center">
            <div style="font-weight:700">{r['shop_name']}</div>
            <span class="badge {badge}">CSm²I {r['csm2i']:.2f}</span>
            <span class="small">Potentiële uplift: {fmt_eur(r['uplift_total'])} &nbsp;
              <span class="small">(CSm²I: {fmt_eur(r['uplift_csm'])}, Conv.: {fmt_eur(r['uplift_conv'])})</span>
            </span>
          </div>
          <ul style="margin:6px 0 0 18px">
            <li>{'CSm²I onder target → training/upsell/cross‑sell.' if badge=='red' else ('CSm²I rond target → finetune bezetting & presentatie.' if badge=='amber' else 'CSm²I boven target → schaal successen en deel playbook.')}</li>
            <li>Conversiedoel {conv_goal_pct}% → benut piekuren & active promo bij instap.</li>
            <li>{spv_line}. {spv_hint}</li>
          </ul>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    # Debug
    with st.expander("🛠️ Debug"):
        st.json({
            "period_step": step, "from": str(date_from), "to": str(date_to),
            "ref_spv": ref_spv, "csm2i_target": csm2i_target,
            "conv_goal_pct": conv_goal_pct, "shops": shop_ids
        })