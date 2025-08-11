# pages/10_Retail_Performance_Radar.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

# ────────────────────────────────────────────────────────────────────────────────
# Page setup & styling
# ────────────────────────────────────────────────────────────────────────────────
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
.kpi  { font-size: 1.25rem; font-weight: 700; }
.eur  { font-variant-numeric: tabular-nums; }
.orange { background:#FEAC7622; border:1px solid #FEAC76; border-radius:12px; padding:16px; color:#222; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; }
.badge-ok { background:#DCFCE7; color:#065F46; }
.badge-warn { background:#FEF3C7; color:#92400E; }
.badge-err { background:#FEE2E2; color:#991B1B; }
.tile { border:1px solid #eee; border-radius:12px; padding:16px; margin-bottom:12px; background:#fff; }
.tile h4 { margin:0 0 8px 0; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x: float) -> str:
    try:
        return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except Exception:
        return "€0"

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else:
            out[c] = 0.0
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    # conversie -> fractie (0..1)
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:
            out["conversion_rate"] = out["conversion_rate"] / 100.0
    else:
        out["conversion_rate"] = out.get("transactions", 0.0) / (out.get("count_in", 0.0) + EPS)
    # SPV
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover", 0.0) / (out.get("count_in", 0.0) + EPS)
    # ATV
    out["atv"] = out.get("turnover", 0.0) / (out.get("transactions", 0.0) + EPS)
    # m² fallback
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0, np.nan).median()
        fallback = med if pd.notnull(med) and med > 0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

def choose_ref_spv(df: pd.DataFrame, mode="portfolio", benchmark_shop_id=None, manual_spv=None, uplift_pct=0.0):
    # robuust + tolerant
    safe = coerce_numeric(df, ["turnover","count_in","shop_id"])
    def spv_of(frame: pd.DataFrame) -> float:
        visitors = float(frame["count_in"].sum())
        turnover = float(frame["turnover"].sum())
        return 0.0 if visitors <= 0 else turnover / (visitors + EPS)

    if mode == "benchmark" and benchmark_shop_id is not None and int(benchmark_shop_id) in safe["shop_id"].astype(int).values:
        base = spv_of(safe[safe["shop_id"].astype(int) == int(benchmark_shop_id)])
    elif mode == "manual" and manual_spv is not None:
        base = float(manual_spv)
    else:
        base = spv_of(safe)

    base = max(0.0, float(base))
    return base * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    """
    Uniform:
      actual_spv = turnover / visitors
      CSm²I      = actual_spv / ref_spv
      expected_spsqm = ref_spv * visitors_per_sqm
      uplift_eur_csm = max(0, visitors * (csm2i_target*ref_spv - actual_spv))
    """
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"] / (out["visitors"] + EPS)
    out["csm2i"]      = out["actual_spv"] / (float(ref_spv) + EPS)

    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = float(ref_spv)   * out["visitors_per_sqm"]

    out["uplift_eur_csm"] = np.maximum(0.0, out["visitors"] * (float(csm2i_target) * float(ref_spv) - out["actual_spv"]))
    return out

def period_dates(label: str):
    today = date.today()
    if label == "last_month":
        first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last  = today.replace(day=1) - timedelta(days=1)
        return first, last
    if label == "30 dagen":
        return today - timedelta(days=30), today - timedelta(days=1)
    return today - timedelta(days=7), today - timedelta(days=1)

def projected_remaining_year_factor(d_from: date, d_to: date) -> float:
    # schaal factor: resterende dagen in dit jaar / geanalyseerde dagen
    analyzed_days = (d_to - d_from).days + 1
    if analyzed_days <= 0:
        return 1.0
    last_day = date(d_to.year, 12, 31)
    days_left = (last_day - d_to).days
    return max(0.0, days_left) / analyzed_days

# ────────────────────────────────────────────────────────────────────────────────
# Shop mapping (root: shop_mapping.py)
# ────────────────────────────────────────────────────────────────────────────────
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# ────────────────────────────────────────────────────────────────────────────────
# UI – periode, granulariteit, winkels, targets
# ────────────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    period_label = st.selectbox("Periode", ["7 dagen", "30 dagen", "last_month"], index=1)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag", "Uur"], index=0)
with c3:
    conv_goal_pct = st.slider("Conversiedoel (%)", 1, 80, 35, 1)
with c4:
    spv_uplift_pct = st.slider("SPV‑uplift (%)", 0, 100, 0, 1)

c5, c6 = st.columns([1,1])
with c5:
    csm2i_target = st.slider("CSm²I‑target", 0.10, 2.00, 1.00, 0.05)
with c6:
    proj_toggle = st.toggle("Projecteer naar resterend jaar", value=False)

st.markdown("### Selecteer winkels")
selected_names = st.multiselect("Selecteer winkels", names, default=names[:5], placeholder="Kies 1 of meer winkels…")
shop_ids = [NAME_TO_ID[n] for n in selected_names]

# Analyseer‑knop
analyze = st.button("🔍 Analyseer", type="secondary")

# ────────────────────────────────────────────────────────────────────────────────
# API helpers
# ────────────────────────────────────────────────────────────────────────────────
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    # POST met queryparams; herhaal keys voor meerdere waarden
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)), ("period_step",step)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = (resp or {}).get("data", {})
    for _, shops in data.items():
        for sid, payload in (shops or {}).items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in dates.items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date; df["hour"] = ts.dt.hour
    return df

# ────────────────────────────────────────────────────────────────────────────────
# RUN
# ────────────────────────────────────────────────────────────────────────────────
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    d_from, d_to = period_dates(period_label)
    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, d_from, d_to, step, outputs)
        raw = normalize_resp(resp)
        if raw.empty:
            st.info("Geen data beschikbaar voor de gekozen periode."); st.stop()

    # Bij UUR aggregeren we naar DAG voor vergelijkbaarheid in KPI's/CSm²I
    if step == "hour":
        raw = (raw
               .groupby(["shop_id","shop_name","date"], as_index=False)
               .agg({"count_in":"sum","transactions":"sum","turnover":"sum","sq_meter":"last"}))
        raw["conversion_rate"]   = raw["transactions"] / (raw["count_in"] + EPS)
        raw["sales_per_visitor"] = raw["turnover"] / (raw["count_in"] + EPS)

    # Referentie‑SPV
    ref_spv = choose_ref_spv(raw, mode="portfolio", uplift_pct=spv_uplift_pct/100.0)

    # CSm²I & uplift (component)
    df = compute_csm2i_and_uplift(raw, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # Conversie‑uplift
    conv_target = float(conv_goal_pct)/100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # Aggregatie per winkel
    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        spv=("actual_spv","mean"),
        conv=("conversion_rate","mean"),
        csm2i=("csm2i","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # Projectie resterend jaar (optioneel)
    proj_factor = projected_remaining_year_factor(d_from, d_to) if proj_toggle else 0.0
    proj_total  = agg["uplift_total"].sum() * proj_factor if proj_toggle else None

    # ── KPI‑tegels ──
    k1, k2, k3 = st.columns(3)
    k1.markdown(
        f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_csm"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    k2.markdown(
        f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_conv"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    k3.markdown(
        f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
            <div class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</div></div>""",
        unsafe_allow_html=True
    )

    st.markdown("&nbsp;", unsafe_allow_html=True)

    # ── Oranje totaal + (optionele) projectie ──
    left, mid, right = st.columns([2,2,2])
    with left:
        st.markdown(
            f"""<div class="orange">
                <b>💰 Total extra potential in revenue</b><br/>
                <span class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</span><br/>
                <small>Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</small>
            </div>""",
            unsafe_allow_html=True
        )
    if proj_toggle and proj_total is not None:
        with mid:
            st.markdown(
                f"""<div class="orange">
                    <b>📈 Projectie resterend jaar</b><br/>
                    <span class="kpi eur">{fmt_eur(proj_total)}</span><br/>
                    <small>Huidig potentieel vermenigvuldigd met resterende dagen in {d_to.year}.</small>
                </div>""",
                unsafe_allow_html=True
            )

    # ── Aanbevelingen per winkel ──
    st.markdown("## 📌 Aanbevelingen per winkel")
    for _, r in agg.sort_values("uplift_total", ascending=False).iterrows():
        csi_badge = "badge-ok" if r["csm2i"] >= csm2i_target*0.98 else ("badge-err" if r["csm2i"] < csm2i_target*0.9 else "badge-warn")
        conv_target = float(conv_goal_pct)/100.0
        conv_badge = "badge-ok" if r["conv"] >= conv_target*0.98 else ("badge-err" if r["conv"] < conv_target*0.9 else "badge-warn")
        st.markdown(f"""<div class="tile">
            <h4>{r['shop_name']}</h4>
            <div><b>CSm²I</b> huidig vs target: {r['csm2i']:.2f} / {csm2i_target:.2f}
                <span class="badge {csi_badge}">{'≥' if r['csm2i']>=csm2i_target else '<' } target</span></div>
            <div><b>SPV</b> (€/bezoeker): {r['spv']:.2f} &nbsp;|&nbsp; <b>Sales per m²</b>: {r['spsqm']:.2f}</div>
            <div><b>Conversie</b> huidig vs doel: {r['conv']*100:.1f}% / {conv_goal_pct:.0f}%
                <span class="badge {conv_badge}">{'≥' if r['conv']>=conv_target else '<'} doel</span></div>
            <div style="margin-top:8px;"><b>Potentiële uplift</b>: <span class="eur">{fmt_eur(r['uplift_total'])}</span></div>
            <ul style="margin-top:6px;">
              <li>🧭 <b>CSm²I</b>: richt op SPV ≥ {(ref_spv*csm2i_target):.2f} via bundels/upsell en efficiënte m².</li>
              <li>🧑‍🤝‍🧑 <b>Conversie</b>: personeelsbezetting en openingsuren finetunen op piekuren.</li>
              <li>🛒 <b>SPV</b>: bundelaanbiedingen, kassaupsell en heldere prijscommunicatie.</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # ── Scatter: SPV vs Sales per m² ──
    st.markdown("## 📈 SPV vs Sales per m² (kleur = CSm²I t.o.v. target)")
    rad = agg.copy()
    low_thr  = float(csm2i_target) * 0.95
    high_thr = float(csm2i_target) * 1.05
    rad["csm2i_band"] = np.select(
        [rad["csm2i"] < low_thr, rad["csm2i"] > high_thr],
        ["Onder target", "Boven target"],
        default="Rond target",
    )
    color_map  = {"Onder target": "#F04438", "Rond target": "#F59E0B", "Boven target": "#16A34A"}
    symbol_map = {"Onder target": "diamond", "Rond target": "circle", "Boven target": "square"}
    rad["hover_spv"]   = rad["spv"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_csi"]   = rad["csm2i"].round(2).map(lambda v: str(v).replace(".", ","))
    rad["hover_uplift"]= rad["uplift_total"].round(0).map(lambda v: ("€{:,.0f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))

    scatter = px.scatter(
        rad, x="spv", y="spsqm", size="uplift_total",
        color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv","hover_spsqm","hover_csi","hover_uplift"],
        labels={"spv":"Sales per Visitor","spsqm":"Sales per m²","csm2i_band":"CSm²I t.o.v. target"},
    )
    scatter.update_traces(
        hovertemplate="<b>%{text}</b><br>" +
                      "SPV: %{customdata[0]}<br>" +
                      "Sales per m²: %{customdata[1]}<br>" +
                      "CSm²I: %{customdata[2]}<br>" +
                      "Uplift: %{customdata[3]}<extra></extra>",
        text=rad["shop_name"],
    )
    scatter.update_layout(
        margin=dict(l=20,r=20,t=10,b=10),
        height=560, legend_title_text="CSm²I t.o.v. target",
        xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
        yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"),
    )
    st.plotly_chart(scatter, use_container_width=True)

    # Debug
    with st.expander("🛠️ Debug"):
        st.write({
            "period_step": step,
            "from": str(d_from),
            "to": str(d_to),
            "shop_ids": shop_ids,
            "ref_spv": ref_spv,
            "csm2i_target": csm2i_target,
            "conv_goal_pct": conv_goal_pct,
            "projection_factor": proj_factor if proj_toggle else 0.0
        })
