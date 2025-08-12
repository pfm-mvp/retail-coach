# pages/10_Retail_Performance_Radar.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlencode

# ============== Page & styling ==============
st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")
st.markdown("""
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
.big-card { border:1px solid #FEAC76; background:#FFF7F2; border-radius:12px; padding:18px 20px;}
.big-card .title { font-weight:700; font-size:1.05rem; }
.big-card .value { font-weight:800; font-size:1.35rem; margin-top:.25rem; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.8rem; font-weight:600; margin-left:6px;}
.badge-green{background:#E9F9EE;color:#14804A;} .badge-amber{background:#FEF3C7;color:#92400E;} .badge-red{background:#FEE2E2;color:#991B1B;}
.mt-8{margin-top:8px;} .mt-16{margin-top:16px;}
</style>
""", unsafe_allow_html=True)

st.title("Retail Performance Radar")
st.caption("Next Best Action • Best Practice Finder • (optioneel) Demografiepatronen")

# ============== Helpers ==============
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x): 
    try: return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except: return "€0"
def fmt_eur2(x):
    try: return ("€{:,.2f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    except: return "€0,00"

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    return out

def normalize_kpis(df):
    out = coerce_numeric(df, ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"])
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty and out["conversion_rate"].max() > 1.5:
        out["conversion_rate"] = out["conversion_rate"] / 100.0
    if "sales_per_visitor" not in out.columns or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover",0.0) / (out.get("count_in",0.0) + EPS)
    out["atv"] = out.get("turnover",0.0) / (out.get("transactions",0.0) + EPS)
    if "sq_meter" in out.columns:
        sqm = pd.to_numeric(out["sq_meter"], errors="coerce")
        med = sqm.replace(0,np.nan).median()
        fallback = med if pd.notnull(med) and med>0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0,np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

def choose_ref_spv(df, mode="portfolio", benchmark_shop_id=None, manual_spv=None, uplift_pct=0.0):
    safe = df.copy()
    for c in ["turnover","count_in","shop_id"]:
        if c in safe.columns: safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0.0)
        else: safe[c] = 0.0
    def spv_of(fr):
        v = float(fr["count_in"].sum()); t = float(fr["turnover"].sum())
        return 0.0 if v<=0 else t/(v+EPS)
    if mode=="benchmark" and benchmark_shop_id is not None and int(benchmark_shop_id) in safe["shop_id"].astype(int).values:
        base = spv_of(safe[safe["shop_id"].astype(int)==int(benchmark_shop_id)])
    elif mode=="manual" and manual_spv is not None:
        base = float(manual_spv)
    else:
        base = spv_of(safe)
    return max(0.0,float(base)) * (1.0+float(uplift_pct))

def compute_csm2i_and_uplift(df, ref_spv, csm2i_target):
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"] / (out["visitors"] + EPS)
    out["csm2i"]      = out["actual_spv"] / (float(ref_spv) + EPS)
    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = float(ref_spv)   * out["visitors_per_sqm"]
    out["uplift_eur_csm"]   = np.maximum(0.0, out["visitors"] * (float(csm2i_target)*float(ref_spv) - out["actual_spv"]))
    return out

# ============== Shop mapping ==============
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k,v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v:k for k,v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# ============== UI ==============
c1,c2,c3 = st.columns([1,1,1])
with c1:
    period_label = st.selectbox("Periode", ["7 dagen","30 dagen","last_month"], index=0)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag","Uur"], index=0)
with c3:
    proj_toggle = st.toggle("Toon projectie voor resterend jaar", value=False)

today = date.today()
if period_label=="last_month":
    first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last  = today.replace(day=1) - timedelta(days=1)
    date_from, date_to = first, last
elif period_label=="30 dagen":
    date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)
else:
    date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)

c4,c5,c6 = st.columns([1,1,1])
with c4:
    conv_goal_pct = st.slider("Conversiedoel (%)", 1, 80, 20, 1)
with c5:
    spv_uplift_pct = st.slider("SPV‑uplift (%)", 0, 100, 10, 1)
with c6:
    csm2i_target = st.slider("CSm²I‑target", 0.10, 2.00, 1.00, 0.05)

st.markdown("### Selecteer winkels")
selected_names = st.multiselect("Selecteer winkels", names, default=names[:5], placeholder="Kies 1 of meer winkels…")
shop_ids = [NAME_TO_ID[n] for n in selected_names]

# Openingstijden slider (alleen relevant voor 'Uur')
open_start, open_end = st.slider(
    "⏰ Openingstijden (alleen voor Uur‑drill‑down & heatmap)",
    min_value=0, max_value=24, value=(9,21), step=1, format="%02d:00"
)

analyze = st.button("🔍 Analyseer", type="secondary")

# ============== API helpers (array params with [] + doseq) ==============
def fetch_report(api_url, shop_ids, dfrom, dto, step, outputs, show_hours=None, timeout=60):
    """
    Build query with array keys using urlencode(..., doseq=True) so it becomes:
      ...&data[]=32224&data[]=30058&data_output[]=count_in&data_output[]=turnover...
    Also uses 'step' (hour|day) as required by the API.
    """
    base_params = [
        ("source","shops"),
        ("period","date"),
        ("form_date_from", str(dfrom)),
        ("form_date_to", str(dto)),
        ("step", step),
    ]
    # Multiple values
    arr_params = []
    for sid in shop_ids:    arr_params.append(("data[]", int(sid)))
    for outp in outputs:    arr_params.append(("data_output[]", outp))

    # Optional hour window (only meaningful for step=hour)
    if show_hours and step == "hour":
        arr_params.append(("show_hours_from", f"{int(show_hours[0]):02d}:00"))
        arr_params.append(("show_hours_to",   f"{int(show_hours[1]):02d}:00"))

    all_params = base_params + arr_params
    qs = urlencode(all_params, doseq=True)
    url = f"{api_url.rstrip('/')}/get-report?{qs}"

    # DEBUG line (kept — very helpful)
    st.session_state["last_url"] = url

    r = requests.post(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def normalize_resp(resp):
    rows = []
    data = resp.get("data", {})
    for _, shops in data.items():
        for sid, payload in shops.items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in dates.items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": SHOP_ID_TO_NAME.get(int(sid), str(sid))}
                rec.update((obj or {}).get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["datetime"] = ts
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    return df

# ============== RUN ==============
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()

    API_URL = st.secrets.get("API_URL","").rstrip("/")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs, show_hours=(open_start, open_end))
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar voor de gekozen periode."); 
            st.caption(f"Debug URL: {st.session_state.get('last_url','')}")
            st.stop()

    # Reference SPV
    ref_spv = choose_ref_spv(df, mode="portfolio", benchmark_shop_id=None, uplift_pct=spv_uplift_pct/100.0)

    # CSm²I + conversion uplifts
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)
    conv_target = float(conv_goal_pct) / 100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # Aggregate per shop
    agg = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        sqm=("sq_meter","mean"),
        spsqm=("actual_spsqm","mean"),
        csm2i=("csm2i","mean"),
        spv=("actual_spv","mean"),
        conv=("conversion_rate","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    agg["uplift_total"] = agg["uplift_csm"] + agg["uplift_conv"]

    # ===== KPI tiles =====
    k1,k2,k3 = st.columns(3)
    k1.markdown(f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div>
    <div class="kpi eur">{fmt_eur(agg["uplift_csm"].sum())}</div></div>""", unsafe_allow_html=True)
    k2.markdown(f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div>
    <div class="kpi eur">{fmt_eur(agg["uplift_conv"].sum())}</div></div>""", unsafe_allow_html=True)
    k3.markdown(f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
    <div class="kpi eur">{fmt_eur(agg["uplift_total"].sum())}</div></div>""", unsafe_allow_html=True)

    # ===== Orange total + optional projection =====
    cA,cB = st.columns([1,1])
    cA.markdown(f"""
    <div class="big-card"><div class="title">💰 Total extra potential in revenue</div>
    <div class="value">{fmt_eur(agg["uplift_total"].sum())}</div>
    <div class="mt-8">Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</div></div>""", unsafe_allow_html=True)
    if proj_toggle:
        today2 = date.today(); end_year = date(today2.year,12,31)
        rem_days = max(0,(end_year - today2).days)
        days_in_period = (date_to-date_from).days + 1 if period_label=="last_month" else (30 if period_label=="30 dagen" else 7)
        daily_potential = agg["uplift_total"].sum() / max(1,days_in_period)
        projection = daily_potential * rem_days
        cB.markdown(f"""
        <div class="big-card"><div class="title">📈 Projectie resterend jaar</div>
        <div class="value">{fmt_eur(projection)}</div>
        <div class="mt-8">Huidig potentieel × resterende dagen dit jaar.</div></div>""", unsafe_allow_html=True)
    else:
        cB.markdown(f"""
        <div class="big-card"><div class="title">📈 Projectie resterend jaar</div>
        <div class="value">–</div>
        <div class="mt-8">Schakel bovenaan in om projectie te tonen.</div></div>""", unsafe_allow_html=True)

    # ===== Scatter =====
    rad = agg.copy()
    low_thr, high_thr = float(csm2i_target)*0.95, float(csm2i_target)*1.05
    rad["csm2i_band"] = np.select([rad["csm2i"]<low_thr, rad["csm2i"]>high_thr], ["Onder target","Boven target"], default="Rond target")
    rad["size_metric"] = rad["uplift_total"].fillna(0.0)
    rad["hover_spv"]   = rad["spv"].round(2).apply(fmt_eur2)
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(fmt_eur2)
    rad["hover_csi"]   = rad["csm2i"].round(2).map(lambda v: str(v).replace(".",","))
    rad["hover_size"]  = rad["size_metric"].round(0).apply(fmt_eur)
    color_map={"Onder target":"#F04438","Rond target":"#F59E0B","Boven target":"#16A34A"}
    symbol_map={"Onder target":"diamond","Rond target":"circle","Boven target":"square"}
    scatter = px.scatter(
        rad, x="spv", y="spsqm", size="size_metric", color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv","hover_spsqm","hover_csi","hover_size"],
        labels={"spv":"Sales per Visitor","spsqm":"Sales per m²","csm2i_band":"CSm²I t.o.v. target"},
    )
    scatter.update_traces(
        hovertemplate="<b>%{text}</b><br>SPV: %{customdata[0]}<br>Sales per m²: %{customdata[1]}<br>CSm²I: %{customdata[2]}<br>Uplift: %{customdata[3]}<extra></extra>",
        text=rad["shop_name"], marker=dict(line=dict(width=0)),
    )
    scatter.update_layout(margin=dict(l=20,r=20,t=10,b=10), height=520,
                          legend_title_text="CSm²I t.o.v. target",
                          xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
                          yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"))
    st.plotly_chart(scatter, use_container_width=True)

    # ===== Recommendations =====
    st.markdown("## Aanbevelingen per winkel")
    best_spv = agg.loc[agg["spv"].idxmax(),"spv"] if not agg.empty else 0.0
    for _, row in agg.sort_values("uplift_total", ascending=False).iterrows():
        name=row["shop_name"]; sid=int(row["shop_id"])
        csi=float(row["csm2i"]); spv_store=float(row["spv"]); spsqm_store=float(row["spsqm"]); conv_store=float(row["conv"])
        up_csm=float(row["uplift_csm"]); up_conv=float(row["uplift_conv"]); total_up=float(row["uplift_total"])
        low_thr, high_thr = float(csm2i_target)*0.95, float(csm2i_target)*1.05
        if csi<low_thr: badge = '<span class="badge badge-red">🔴 onder target</span>'
        elif csi>high_thr: badge = '<span class="badge badge-green">🟢 boven target</span>'
        else: badge = '<span class="badge badge-amber">🟠 rond target</span>'
        spv_comp = f"{fmt_eur2(spv_store)} vs best {fmt_eur2(best_spv)}" if best_spv>0 else fmt_eur2(spv_store)

        st.markdown(f"### {name} {badge}", unsafe_allow_html=True)
        st.markdown(
            f"**CSm²I huidig vs target:** {csi:.2f} / {csm2i_target:.2f}  \n"
            f"**Conversie huidig vs doel:** {conv_store*100:.1f}% / {conv_goal_pct:.0f}%  \n"
            f"**Sales per m² (actueel):** {fmt_eur2(spsqm_store)}  \n"
            f"**Sales per Visitor:** {spv_comp}  \n"
            f"**Potentiële uplift:** {fmt_eur(total_up)} *(CSm²I: {fmt_eur(up_csm)} • Conversie: {fmt_eur(up_conv)})*"
        )
        bullets=[]
        if csi < csm2i_target: bullets.append("CSm²I onder target → plan **upsell/cross‑sell** & coach op verkooproutine (SPV).")
        if conv_store < conv_target: bullets.append("Conversie onder doel → **extra bezetting** op piekuren & **actie bij instap**.")
        if best_spv - spv_store > 0.1: bullets.append(f"SPV {fmt_eur2(spv_store)} < best {fmt_eur2(best_spv)} → leer van **best practice** winkel.")
        if not bullets: bullets.append("Presteert op of boven target → **vasthouden** en best practices delen.")
        for b in bullets: st.write(f"- {b}")

        # ----- Hour drill‑down & heatmap (only when step=hour) -----
        if step == "hour":
            st.markdown("**Uur‑profielen (drill‑down & heatmap)**")
            st.caption(f"Heatmap binnen openingstijd {open_start:02d}:00–{open_end:02d}:00 (gemiddeld in de gekozen periode).")

            sub = df[df["shop_id"]==sid].copy()
            if sub.empty:
                st.info("Geen uurdata voor deze winkel in de gekozen periode."); 
                st.markdown("---")
                continue

            # Filter opening hours
            sub = sub[(sub["hour"]>=open_start) & (sub["hour"]<open_end)]
            if sub.empty:
                st.info("Binnen de opgegeven openingstijd is geen uurdata.")
                st.markdown("---")
                continue

            sub["weekday"] = sub["datetime"].dt.dayofweek  # 0=Mon
            sub["weekday_name"] = sub["datetime"].dt.day_name()
            # Visitors per hour per weekday (mean)
            hm = sub.groupby(["weekday","hour"])["count_in"].mean().reset_index()
            # Ensure full grid to avoid blank figure artefacts
            hrs = list(range(open_start, open_end))
            wds = list(range(0,7))
            full = pd.MultiIndex.from_product([wds, hrs], names=["weekday","hour"])
            hm = hm.set_index(["weekday","hour"]).reindex(full).reset_index()
            hm["weekday_name"] = hm["weekday"].map({0:"Monday",1:"Tuesday",2:"Wednesday",3:"Thursday",4:"Friday",5:"Saturday",6:"Sunday"})
            hm["hour_label"] = hm["hour"].map(lambda h: f"{int(h):02d}:00")

            z = hm["count_in"].fillna(0.0).values.reshape(len(wds), len(hrs))
            fig_hm = go.Figure(data=go.Heatmap(
                z=z, x=[f"{h:02d}:00" for h in hrs], y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                colorscale="Viridis", colorbar=dict(title="Bezoekers (gem.)")
            ))
            fig_hm.update_layout(
                height=260, margin=dict(l=20,r=20,t=10,b=10),
                xaxis_title="Uur", yaxis_title="Weekdag"
            )
            # unique key per chart avoids duplicate id
            st.plotly_chart(fig_hm, use_container_width=True, key=f"hm_{sid}")

            st.markdown("---")

    # Debug
    with st.expander("🛠️ Debug"):
        st.json({
            "step": step, "from": str(date_from), "to": str(date_to), "shop_ids": shop_ids,
            "ref_spv": ref_spv, "csm2i_target": csm2i_target, "conv_goal_pct": conv_goal_pct,
            "opening_from": open_start, "opening_to": open_end,
            "last_url": st.session_state.get("last_url","")
        })
