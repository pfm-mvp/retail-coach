# pages/10_Retail_Performance_Radar.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument+Sans', sans-serif !important; }
[data-baseweb="tag"] { background-color: #9E77ED !important; color: #fff !important; }
button[data-testid="stBaseButton-secondary"] {
  background-color:#F04438 !important; color:#fff !important; border-radius:16px !important;
  font-weight:600 !important; padding:.6rem 1.4rem !important; border:none !important;
}
button[data-testid="stBaseButton-secondary"]:hover { background-color:#d13c30 !important; cursor:pointer; }
.card { border:1px solid #eee; border-radius:12px; padding:14px 16px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.kpi { font-size:1.2rem; font-weight:800; }
.eur { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x: float) -> str:
    return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        else: out[c] = 0.0
    return out

def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:
            out["conversion_rate"] = out["conversion_rate"] / 100.0
    else:
        out["conversion_rate"] = out.get("transactions", 0.0) / (out.get("count_in", 0.0) + EPS)
    if ("sales_per_visitor" not in out.columns) or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover", 0.0) / (out.get("count_in", 0.0) + EPS)
    out["atv"] = out.get("turnover", 0.0) / (out.get("transactions", 0.0) + EPS)
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0, np.nan).median()
        fallback = med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

def choose_ref_spv(df: pd.DataFrame, mode="portfolio", benchmark_shop_id=None, manual_spv=None, uplift_pct=0.0):
    safe = df.copy()
    for c in ["turnover","count_in","shop_id"]:
        if c in safe.columns: safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0.0)
        else: safe[c] = 0.0

    def spv_of(frame: pd.DataFrame) -> float:
        visitors = float(frame["count_in"].sum())
        turnover = float(frame["turnover"].sum())
        return 0.0 if visitors <= 0 else turnover / (visitors + EPS)

    if mode == "benchmark" and benchmark_shop_id is not None and int(benchmark_shop_id) in safe["shop_id"].astype(int).values:
        base = spv_of(safe[safe["shop_id"].astype(int)==int(benchmark_shop_id)])
    elif mode == "manual" and manual_spv is not None:
        base = float(manual_spv)
    else:
        base = spv_of(safe)

    base = max(0.0, float(base))
    return base * (1.0 + float(uplift_pct))

def compute_csm2i_and_uplift(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"] / (out["visitors"] + EPS)
    out["csm2i"]      = out["actual_spv"] / (float(ref_spv) + EPS)
    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = float(ref_spv)   * out["visitors_per_sqm"]
    out["uplift_eur_csm"]   = np.maximum(0.0, out["visitors"] * (float(csm2i_target)*float(ref_spv) - out["actual_spv"]))
    return out

def resolve_period(preset: str) -> tuple[date, date]:
    today = date.today()
    first_of_month = today.replace(day=1)
    last_month_last_day = first_of_month - timedelta(days=1)
    last_month_first_day = last_month_last_day.replace(day=1)
    p = (preset or "").lower()
    if p in ("7 dagen","last_7_days"):   return today - timedelta(days=7),  today - timedelta(days=1)
    if p in ("30 dagen","last_30_days"): return today - timedelta(days=30), today - timedelta(days=1)
    if p in ("last_month","vorige maand"): return last_month_first_day, last_month_last_day
    if p in ("month_to_date","mtd","deze maand"): return first_of_month, today
    if p in ("year_to_date","ytd","dit jaar"): return today.replace(month=1, day=1), today
    if p in ("gisteren","yesterday"): y=today-timedelta(days=1); return y,y
    return today - timedelta(days=7), today - timedelta(days=1)

def period_length_days(d_from: date, d_to: date) -> int:
    return max(1, (d_to - d_from).days + 1)

def remaining_year_days(today: date | None = None) -> int:
    today = today or date.today()
    end = date(today.year, 12, 31)
    return max(0, (end - today).days)

# Shop mapping
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# UI: periode / granulariteit / targets
c1, c2, c3 = st.columns([1,1,1])
with c1:
    period_options = ["7 dagen", "30 dagen", "last_month", "month_to_date", "year_to_date"]
    period_label = st.selectbox("Periode", period_options, index=0)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag", "Uur"], index=0)
with c3:
    proj_rest_year = st.toggle("💡 Projecteer naar resterend jaar", value=True)

date_from, date_to = resolve_period(period_label)

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

def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, timeout=60):
    params = [("source","shops"), ("period","date"),
              ("form_date_from",str(dfrom)), ("form_date_to",str(dto)), ("period_step",step)]
    for sid in shop_ids: params.append(("data", int(sid)))
    for outp in outputs: params.append(("data_output", outp))
    r = requests.post(api_url, params=params, timeout=timeout); r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = resp.get("data", {})
    for _, shops in data.items():
        for sid, payload in shops.items():
            for ts, obj in (payload or {}).get("dates", {}).items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date; df["hour"] = ts.dt.hour
    return df

if analyze:
    if not shop_ids: st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL: st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]
    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs)
        df = normalize_resp(resp)
        if df.empty: st.info("Geen data beschikbaar voor de gekozen periode."); st.stop()

    mode = "portfolio"; bm_id = None
    ref_spv = choose_ref_spv(df, mode=mode, benchmark_shop_id=bm_id, uplift_pct=spv_uplift_pct/100.0)
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    conv_target = float(conv_goal_pct) / 100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csm2i=("csm2i","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # KPI‑tegels
    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div><div class="kpi eur">{fmt_eur(agg["uplift_csm"].sum())}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div><div class="kpi eur">{fmt_eur(agg["uplift_conv"].sum())}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div><div class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</div></div>', unsafe_allow_html=True)

    # Projectie naar resterend jaar
    if proj_rest_year:
        days_left = remaining_year_days()
        days_in_period = period_length_days(date_from, date_to)
        projection = float(agg["uplift_total"].sum()) * (days_left / days_in_period)
        st.markdown(
            f"""
            <div style="margin:10px 0 0 0;padding:16px;border-radius:12px;background:#FEAC7622;border:1px solid #FEAC76">
              <div style="font-weight:700;">💰 Total extra potential in revenue (resterend jaar)</div>
              <div style="font-size:1.2rem;font-weight:800;">{fmt_eur(projection)}</div>
              <div style="opacity:.8;">Projectie: huidig {period_label.lower()}‑potentieel × (resterende dagen dit jaar / dagen in analyseperiode).</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Scatter (blijft zoals je hem nu hebt)
    rad = df.groupby(["shop_id","shop_name"]).agg(
        spv=("actual_spv","mean"),
        spsqm=("actual_spsqm","mean"),
        csi=("csm2i","mean"),
        visitors=("count_in","sum"),
    ).reset_index()

    size_series = agg.set_index("shop_id")["uplift_total"]
    rad["size_metric"] = rad["shop_id"].map(size_series).fillna(0.0)
    rad["hover_size"] = rad["size_metric"].round(0).map(lambda v: ("€{:,.0f}".format(v)).replace(",", "X").replace(".", ",").replace("X","."))
    low_thr  = float(csm2i_target) * 0.95
    high_thr = float(csm2i_target) * 1.05
    rad["csm2i_band"] = np.select([rad["csi"]<low_thr, rad["csi"]>high_thr], ["Onder target","Boven target"], default="Rond target")
    rad["hover_spv"]   = rad["spv"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X","."))
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X","."))
    rad["hover_csi"]   = rad["csi"].round(2).map(lambda v: str(v).replace(".", ","))

    color_map  = {"Onder target":"#F04438","Rond target":"#F59E0B","Boven target":"#16A34A"}
    symbol_map = {"Onder target":"diamond","Rond target":"circle","Boven target":"square"}
    fig = px.scatter(
        rad, x="spv", y="spsqm", size="size_metric",
        color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv","hover_spsqm","hover_csi","hover_size"],
        labels={"spv":"Sales per Visitor","spsqm":"Sales per m²","csm2i_band":"CSm²I t.o.v. target"},
    )
    fig.update_traces(
        hovertemplate="<b>%{text}</b><br>SPV: %{customdata[0]}<br>Sales per m²: %{customdata[1]}<br>CSm²I: %{customdata[2]}<br>Uplift: %{customdata[3]}<extra></extra>",
        text=rad["shop_name"]
    )
    fig.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=560,
                      legend_title_text="CSm²I t.o.v. target",
                      xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
                      yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("🛠️ Debug"):
        st.json({
            "period_step": step,
            "from": str(date_from),
            "to": str(date_to),
            "shop_ids": shop_ids,
            "ref_spv": ref_spv,
            "csm2i_target": csm2i_target,
            "conv_goal_pct": conv_goal_pct,
        })
