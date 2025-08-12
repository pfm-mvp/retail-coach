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
    safe = df.copy()
    for c in ["turnover","count_in","shop_id"]:
        if c in safe.columns:
            safe[c] = pd.to_numeric(safe[c], errors="coerce").fillna(0.0)
        else:
            safe[c] = 0.0

    def spv_of(frame: pd.DataFrame) -> float:
        visitors = float(frame["count_in"].sum())
        turnover = float(frame["turnover"].sum())
        return 0.0 if visitors <= 0 else turnover / (visitors + EPS)

    if mode == "benchmark" and benchmark_shop_id is not None and int(benchmark_shop_id) in safe["shop_id"].astype(int).values:
        sub = safe[safe["shop_id"].astype(int) == int(benchmark_shop_id)]
        base = spv_of(sub)
    elif mode == "manual" and manual_spv is not None:
        base = float(manual_spv)
    else:
        base = spv_of(safe)
    base = max(0.0, float(base))
    return base * (1.0 + float(uplift_pct))


def compute_csm2i_and_uplift(df: pd.DataFrame, ref_spv: float, csm2i_target: float):
    """
    CSm²I (index) = actual_spv / ref_spv
    actual_spv    = turnover / visitors
    expected_spsqm= ref_spv * visitors_per_sqm
    Uplift (CSm²I)= max(0, visitors * (csm2i_target*ref_spv - actual_spv))
    """
    out = normalize_kpis(df)
    out["visitors"]   = out["count_in"]
    out["actual_spv"] = out["turnover"] / (out["visitors"] + EPS)
    out["csm2i"]      = out["actual_spv"] / (float(ref_spv) + EPS)
    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = float(ref_spv)   * out["visitors_per_sqm"]
    out["uplift_eur_csm"]   = np.maximum(0.0, out["visitors"] * (float(csm2i_target) * float(ref_spv) - out["actual_spv"]))
    return out

# =========================
# Shop mapping
# =========================
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam"}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}
names = sorted(NAME_TO_ID.keys(), key=str.lower)

# =========================
# UI – periode, granulariteit, winkels, targets
# =========================
PERIOD_OPTIONS = [
    ("Last week",   "last_week"),
    ("This month",  "this_month"),
    ("Last month",  "last_month"),
    ("This quarter","this_quarter"),
    ("Last quarter","last_quarter"),
    ("This year",   "this_year"),
    ("Last year",   "last_year"),
]

c1, c2, c3 = st.columns([1,1,1])
with c1:
    period_label = st.selectbox("Periode", [lbl for (lbl, _) in PERIOD_OPTIONS], index=1)
    period_token = dict(PERIOD_OPTIONS)[period_label]
with c2:
    gran = st.selectbox("Granulariteit", ["Dag", "Uur"], index=0)
with c3:
    proj_toggle = st.toggle("Toon projectie voor resterend jaar", value=False)

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

# Openingstijden (voor UUR drilldown/heatmap)
open_start, open_end = st.slider(
    "⏰ Openingstijden (alleen van toepassing bij granulariteit ‘Uur’)",
    0, 24, (9, 21), step=1, format="%02d:00"
)

# Analyseer‑knop (links)
analyze = st.button("🔍 Analyseer", type="secondary")

# =========================
# API helper (presets-first, robust fallback)
# =========================
def fetch_report(api_url, shop_ids, step, outputs, period_token=None, dfrom=None, dto=None, show_hours=None, timeout=60):
    """
    Primary: repeat keys (data, data_output) + step=hour|day + period=<token>  (Vemcount presets)
    Fallback: bracketed keys + period_step=... for server quirks.
    If period_token is None and dates given -> uses period=date with form_date_from/to.
    """
    use_dates = (period_token is None or period_token == "date")

    # --- primary (repeat keys, step)
    params = [("source", "shops"), ("step", step)]
    if use_dates and dfrom and dto:
        params.extend([("period","date"), ("form_date_from", str(dfrom)), ("form_date_to", str(dto))])
    else:
        params.append(("period", period_token or "this_month"))

    for sid in shop_ids:
        params.append(("data", int(sid)))
    for outp in outputs:
        params.append(("data_output", outp))
    if show_hours and step == "hour":
        sh, eh = show_hours
        params.append(("show_hours_from", f"{sh:02d}:00"))
        params.append(("show_hours_to",   f"{eh:02d}:00"))

    # Build URL manually to preserve repeated keys
    qs = urlencode(params, doseq=True).replace("%3A", ":")
    url = f"{api_url}?{qs}"

    r = requests.post(url, timeout=timeout)
    try:
        r.raise_for_status()
        return r.json()
    except requests.HTTPError:
        # --- fallback (bracket keys + period_step)
        params_fb = [("source","shops"), ("period_step", step)]
        if use_dates and dfrom and dto:
            params_fb.extend([("period","date"), ("form_date_from", str(dfrom)), ("form_date_to", str(dto))])
        else:
            params_fb.append(("period", period_token or "this_month"))

        for sid in shop_ids:
            params_fb.append(("data[]", int(sid)))
        for outp in outputs:
            params_fb.append(("data_output[]", outp))
        if show_hours and step == "hour":
            sh, eh = show_hours
            params_fb.append(("show_hours_from", f"{sh:02d}:00"))
            params_fb.append(("show_hours_to",   f"{eh:02d}:00"))

        qs_fb = urlencode(params_fb, doseq=True).replace("%3A", ":")
        url_fb = f"{api_url}?{qs_fb}"
        r2 = requests.post(url_fb, timeout=timeout)
        r2.raise_for_status()
        return r2.json()


def normalize_resp(resp):
    rows = []
    data = resp.get("data", {})
    for _, shops in data.items():
        for sid, payload in shops.items():
            dates = (payload or {}).get("dates", {})
            for ts, obj in dates.items():
                rec = {"timestamp": ts, "shop_id": int(sid)}
                rows.append({**rec, **((obj or {}).get("data", {}))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    # map names after creation (faster)
    try:
        from shop_mapping import SHOP_NAME_MAP as _MAP2
    except Exception:
        _MAP2 = {}
    _SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP2.items() if str(v).strip()}
    df["shop_name"] = df["shop_id"].map(_SHOP_ID_TO_NAME).fillna(df["shop_id"].astype(str))
    return df

# =========================
# RUN
# =========================
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","").rstrip("/")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(
            API_URL,
            shop_ids=shop_ids,
            step=step,
            outputs=outputs,
            period_token=period_token,                 # presets → Vemcount
            show_hours=(open_start, open_end)
        )
        df_raw = normalize_resp(resp)
        if df_raw.empty:
            st.info("Geen data beschikbaar voor de gekozen periode/granulariteit."); st.stop()

    # --- Eén bron voor berekeningen
    df = compute_csm2i_and_uplift(df_raw, ref_spv=0.0, csm2i_target=1.0)  # temp to normalize types

    # Referentie‑SPV (portfolio + uplift)
    mode = "portfolio"
    bm_id = None
    ref_spv = choose_ref_spv(df, mode=mode, benchmark_shop_id=bm_id, uplift_pct=spv_uplift_pct/100.0)

    # Recompute with actual ref/target for correct uplift
    df = compute_csm2i_and_uplift(df_raw, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # Conversie‑uplift (gewogen gebaseerd op counts)
    conv_target = float(conv_goal_pct) / 100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # ===== Aggregatie per winkel (PERIODETrouw — onafhankelijk van granulariteit)
    g = df.groupby(["shop_id","shop_name"], as_index=False).agg(
        visitors=("count_in","sum"),
        turnover=("turnover","sum"),
        transactions=("transactions","sum"),
        sqm=("sq_meter","mean"),  # aanname: constant per winkel
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    )
    # Gewogen ratio's
    g["spv"]  = g["turnover"] / (g["visitors"] + EPS)
    g["conv"] = g["transactions"] / (g["visitors"] + EPS)
    g["spsqm"] = g["turnover"] / (g["sqm"] + EPS)
    # CSm²I opnieuw uit periode‑SPV
    g["csm2i"] = g["spv"] / (float(ref_spv) + EPS)
    # Safeguards
    g["conv"] = g["conv"].clip(lower=0.0, upper=1.0)

    g["uplift_total"] = g["uplift_csm"] + g["uplift_conv"]

    # ===== KPI‑tegels =====
    k1, k2, k3 = st.columns(3)
    k1.markdown(
        f"""<div class="card"><div>🚀 <b>CSm²I potential</b><br/><small>({period_label}, target {csm2i_target:.2f})</small></div>
            <div class="kpi eur">{fmt_eur(g["uplift_csm"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    k2.markdown(
        f"""<div class="card"><div>🎯 <b>Conversion potential</b><br/><small>({period_label}, doel = {conv_goal_pct}%)</small></div>
            <div class="kpi eur">{fmt_eur(g["uplift_conv"].sum())}</div></div>""",
        unsafe_allow_html=True
    )
    k3.markdown(
        f"""<div class="card"><div>∑ <b>Total potential</b><br/><small>({period_label})</small></div>
            <div class="kpi eur">{fmt_eur(g["uplift_total"].sum())}</div></div>""",
        unsafe_allow_html=True
    )

    # ===== Oranje total + optionele projectie =====
    cA, cB = st.columns([1,1])
    cA.markdown(
        f"""
        <div class="big-card">
          <div class="title">💰 Total extra potential in revenue</div>
          <div class="value">{fmt_eur(g["uplift_total"].sum())}</div>
          <div class="mt-8">Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if proj_toggle:
        # resterende dagen dit jaar (projectie baseer je op # unieke dagen in dataset)
        today2 = date.today()
        end_year = date(today2.year, 12, 31)
        rem_days = (end_year - today2).days

        try:
            days_in_period = int(pd.to_datetime(df["date"], errors="coerce").nunique())
            if days_in_period <= 0:
                days_in_period = 1
        except Exception:
            days_in_period = 1

        daily_potential = g["uplift_total"].sum() / max(1, days_in_period)
        projection = daily_potential * max(0, rem_days)
        cB.markdown(
            f"""
            <div class="big-card">
              <div class="title">📈 Projectie resterend jaar</div>
              <div class="value">{fmt_eur(projection)}</div>
              <div class="mt-8">Huidig potentieel × resterende dagen dit jaar.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        cB.markdown(
            f"""
            <div class="big-card">
              <div class="title">📈 Projectie resterend jaar</div>
              <div class="value">–</div>
              <div class="mt-8">Schakel bovenaan in om projectie te tonen.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===== Scatter: SPV vs Sales per m² (kleur = CSm²I t.o.v. target) =====
    rad = g.copy()
    low_thr  = float(csm2i_target) * 0.95
    high_thr = float(csm2i_target) * 1.05
    rad["csm2i_band"] = np.select(
        [rad["csm2i"] < low_thr, rad["csm2i"] > high_thr],
        ["Onder target", "Boven target"],
        default="Rond target",
    )
    # maat: total uplift
    rad["size_metric"] = rad["uplift_total"].fillna(0.0)
    rad["hover_spv"]   = rad["spv"].round(2).apply(fmt_eur2)
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(fmt_eur2)
    rad["hover_csi"]   = rad["csm2i"].round(2).map(lambda v: str(v).replace(".", ","))
    rad["hover_size"]  = rad["size_metric"].round(0).apply(fmt_eur)

    color_map  = {"Onder target": "#F04438", "Rond target": "#F59E0B", "Boven target": "#16A34A"}
    symbol_map = {"Onder target": "diamond", "Rond target": "circle", "Boven target": "square"}

    scatter = px.scatter(
        rad,
        x="spv", y="spsqm", size="size_metric",
        color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=["hover_spv", "hover_spsqm", "hover_csi", "hover_size"],
        labels={"spv": "Sales per Visitor", "spsqm": "Sales per m²", "csm2i_band": "CSm²I t.o.v. target"},
    )
    scatter.update_traces(
        hovertemplate="<b>%{text}</b><br>" +
                      "SPV: %{customdata[0]}<br>" +
                      "Sales per m²: %{customdata[1]}<br>" +
                      "CSm²I: %{customdata[2]}<br>" +
                      "Uplift: %{customdata[3]}<extra></extra>",
        text=rad["shop_name"],
        marker=dict(line=dict(width=0))
    )
    scatter.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=520,
        legend_title_text="CSm²I t.o.v. target",
        xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
        yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"),
    )
    st.plotly_chart(scatter, use_container_width=True)

    # ===== Aanbevelingen per winkel =====
    st.markdown("## Aanbevelingen per winkel")
    # benchmark SPV (best performer) t.b.v. vergelijk
    best_spv = g.loc[g["spv"].idxmax(), "spv"] if not g.empty else 0.0

    for _, row in g.sort_values("uplift_total", ascending=False).iterrows():
        name = row["shop_name"]; sid = int(row["shop_id"])
        csi = float(row["csm2i"]); spv_store = float(row["spv"]); spsqm_store = float(row["spsqm"])
        conv_store = float(row["conv"])
        up_csm = float(row["uplift_csm"]); up_conv = float(row["uplift_conv"])
        total_up = float(row["uplift_total"])

        # kleurindicatie
        if csi < low_thr:
            badge = '<span class="badge badge-red">🔴 onder target</span>'
        elif csi > high_thr:
            badge = '<span class="badge badge-green">🟢 boven target</span>'
        else:
            badge = '<span class="badge badge-amber">🟠 rond target</span>'

        # vergelijk t.o.v. beste SPV
        spv_comp = f"{fmt_eur2(spv_store)} vs best {fmt_eur2(best_spv)}" if best_spv > 0 else fmt_eur2(spv_store)

        st.markdown(f"### {name} {badge}", unsafe_allow_html=True)
        st.markdown(
            f"""
            **CSm²I huidig vs target:** {csi:.2f} / {csm2i_target:.2f}  
            **Conversie huidig vs doel:** {conv_store*100:.1f}% / {conv_goal_pct:.0f}%  
            **Sales per m² (actueel):** {fmt_eur2(spsqm_store)}  
            **Sales per Visitor:** {spv_comp}  
            **Potentiële uplift:** {fmt_eur(total_up)} *(CSm²I: {fmt_eur(up_csm)} • Conversie: {fmt_eur(up_conv)})*
            """.strip()
        )
        bullets = []
        if csi < csm2i_target:
            bullets.append("CSm²I onder target → plan **upsell/cross‑sell** & coach op verkooproutine (SPV).")
        if conv_store < conv_target:
            bullets.append("Conversie onder doel → **extra bezetting** op piekuren & **actie bij instap**.")
        if spv_store < best_spv:
            bullets.append("SPV lager dan best performer → **best practices** ophalen & toepassen.")
        if not bullets:
            bullets.append("Presteert op of boven target → **vasthouden** en best practices delen.")
        for b in bullets:
            st.write(f"- {b}")
        st.markdown("---")

    # ===== Multi‑store hour drilldown (vergelijking) =====
    if step == "hour":
        st.markdown("## 🔎 Multi‑store hour drilldown (vergelijking)")
        st.caption("Gemiddelden per uur over de gekozen periode, gewogen op bezoekers. Handig voor snelle vergelijking tussen vestigingen.")
        sub_all = df.copy()
        sub_all = sub_all[(sub_all['hour'] >= open_start) & (sub_all['hour'] < open_end)]
        sub_all['date_ts'] = pd.to_datetime(sub_all['date'], errors='coerce')
        sub_all = sub_all.dropna(subset=['date_ts'])
        # per store-hour: sommen + dagen aanwezig → gewogen metrics
        grp_all = sub_all.groupby(['shop_id','shop_name','hour'], as_index=False).agg(
            visitors_sum=('count_in','sum'),
            turnover_sum=('turnover','sum'),
            transactions_sum=('transactions','sum'),
            days_present=('date_ts','nunique'),
        )
        grp_all['visitors_avg'] = grp_all['visitors_sum'] / grp_all['days_present'].replace(0, np.nan)
        grp_all['spv_hour'] = grp_all['turnover_sum'] / (grp_all['visitors_sum'] + 1e-9)
        grp_all['conv_hour'] = grp_all['transactions_sum'] / (grp_all['visitors_sum'] + 1e-9)
        # Tabs voor 3 perspectieven
        t1, t2, t3 = st.tabs(["Bezoekers/uur", "SPV/uur", "Conversie/uur"])
        with t1:
            fig1 = px.bar(grp_all, x='hour', y='visitors_avg', color='shop_name', barmode='group', labels={'visitors_avg':'Gem. bezoekers'})
            fig1.update_layout(height=360, margin=dict(l=20,r=20,t=10,b=10), xaxis=dict(dtick=1))
            st.plotly_chart(fig1, use_container_width=True)
        with t2:
            fig2 = px.line(grp_all, x='hour', y='spv_hour', color='shop_name', markers=True, labels={'spv_hour':'SPV (€)'})
            fig2.update_layout(height=360, margin=dict(l=20,r=20,t=10,b=10), xaxis=dict(dtick=1), yaxis=dict(tickformat=',.2f'))
            st.plotly_chart(fig2, use_container_width=True)
        with t3:
            dfc = grp_all.copy(); dfc['conv_pct'] = dfc['conv_hour']*100
            fig3 = px.line(dfc, x='hour', y='conv_pct', color='shop_name', markers=True, labels={'conv_pct':'Conversie (%)'})
            fig3.update_layout(height=360, margin=dict(l=20,r=20,t=10,b=10), xaxis=dict(dtick=1), yaxis=dict(tickformat=',.1f'))
            st.plotly_chart(fig3, use_container_width=True)

        # Optie: individuele secties tonen/verbergen
        show_individual = st.toggle('Toon individuele vestigingen (uurprofielen & heatmaps)', value=True)

    # ===== Uur‑drilldown (alleen wanneer 'Uur' is gekozen) =====
    if step == "hour" and show_individual:
        st.markdown("## Uur‑profielen (drill‑down & heatmap)")
        st.caption(f"Heatmap binnen openingstijd {open_start:02d}:00–{open_end:02d}:00 (gemiddeld in de gekozen periode).")

        for _, row in g.iterrows():
            sid = int(row["shop_id"]); name = row["shop_name"]
            sub = df[(df["shop_id"] == sid)].copy()
            if sub.empty:
                continue
            sub = normalize_kpis(sub)
            sub = sub[(sub["hour"] >= open_start) & (sub["hour"] < open_end)]
            sub["date_ts"] = pd.to_datetime(sub["date"])  # for #days

            with st.expander(f"⏱️ {name} — uurprofiel & heatmap", expanded=False):
                # ===== Per uur (gewogen)
                grp_h = sub.groupby("hour", as_index=False).agg(
                    visitors_sum=("count_in","sum"),
                    turnover_sum=("turnover","sum"),
                    transactions_sum=("transactions","sum"),
                    days_present=("date_ts","nunique"),
                ).sort_values("hour")
                grp_h["visitors_avg"] = grp_h["visitors_sum"] / grp_h["days_present"].replace(0, np.nan)
                grp_h["spv_hour"] = grp_h["turnover_sum"] / (grp_h["visitors_sum"] + EPS)
                grp_h["conv_hour"] = grp_h["transactions_sum"] / (grp_h["visitors_sum"] + EPS)

                # Hovertext (EU‑notatie)
                grp_h["visitors_txt"] = grp_h["visitors_avg"].round(0).apply(fmt_int) + " bezoekers"
                grp_h["spv_txt"] = grp_h["spv_hour"].apply(fmt_eur2)
                grp_h["conv_txt"] = grp_h["conv_hour"].apply(lambda v: fmt_pct(v, 1))
                custom_bar = np.stack([grp_h["visitors_txt"], grp_h["spv_txt"], grp_h["conv_txt"]], axis=1)
                custom_spv = np.stack([grp_h["spv_txt"].values], axis=1)
                custom_conv = np.stack([grp_h["conv_txt"].values], axis=1)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=grp_h["hour"], y=grp_h["visitors_avg"], name="Bezoekers (gem.)", yaxis="y2", opacity=0.30,
                    customdata=custom_bar,
                    hovertemplate="Uur %{x:02d}:00<br>%{customdata[0]}<br>SPV: %{customdata[1]}<br>Conversie: %{customdata[2]}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=grp_h["hour"], y=grp_h["spv_hour"], name="SPV (€)", mode="lines+markers",
                    customdata=custom_spv,
                    hovertemplate="Uur %{x:02d}:00<br>SPV: %{customdata[0]}<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=grp_h["hour"], y=grp_h["conv_hour"]*100, name="Conversie (%)", mode="lines+markers",
                    customdata=custom_conv,
                    hovertemplate="Uur %{x:02d}:00<br>Conversie: %{customdata[0]}<extra></extra>",
                ))
                fig.update_layout(
                    height=380, margin=dict(l=20,r=20,t=10,b=10),
                    xaxis=dict(title="Uur", tickmode='linear', dtick=1),
                    yaxis=dict(title="SPV (€) / Conversie (%)", rangemode="tozero"),
                    yaxis2=dict(title="Bezoekers (gem.)", overlaying="y", side="right", rangemode="tozero"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                # ===== Heatmap bezoekers
                sub = sub.dropna(subset=["date"]).copy()
                sub["date_ts"] = pd.to_datetime(sub["date"], errors="coerce")
                sub = sub.dropna(subset=["date_ts"]).copy()
                sub["weekday_en"] = sub["date_ts"].dt.day_name()
                sub["weekday_en"] = pd.Categorical(sub["weekday_en"], ordered=True, categories=WEEKDAY_ORDER_EN)

                grp_wh = sub.groupby(["weekday_en","hour"], observed=True, as_index=False).agg(
                    visitors_sum=("count_in","sum"),
                    days_present=("date_ts","nunique"),
                )
                grp_wh["visitors_avg"] = grp_wh["visitors_sum"] / grp_wh["days_present"].replace(0, np.nan)
                grp_wh["weekday_nl"] = grp_wh["weekday_en"].astype(str).map(lambda d: WEEKDAY_EN_TO_NL.get(d, d))

                pivot = grp_wh.pivot_table(index="weekday_nl", columns="hour", values="visitors_avg", aggfunc="mean")
                pivot = pivot.reindex(WEEKDAY_ORDER_NL)
                pivot = pivot.fillna(0)

                if pivot.empty:
                    st.info("Binnen de opgegeven openingstijd is geen uurdata.")
                else:
                    hours_sorted = sorted(pivot.columns.tolist())
                    pivot = pivot[hours_sorted]
                    text_matrix = [[f"{rowlabel} {int(col):02d}:00 — {fmt_int(val)} bezoekers" for col, val in zip(pivot.columns, row)] for rowlabel, row in zip(pivot.index, pivot.values)]

                    fig_hm = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=pivot.columns,
                        y=pivot.index,
                        colorscale="Viridis",
                        colorbar=dict(title="Gem. bezoekers"),
                        text=text_matrix,
                        hovertemplate="%{text}<extra></extra>",
                    ))
                    fig_hm.update_layout(
                        height=360, margin=dict(l=20,r=20,t=10,b=10),
                        xaxis=dict(title="Uur", tickmode='linear', dtick=1),
                        yaxis=dict(title="Weekdag"),
                    )
                    st.plotly_chart(fig_hm, use_container_width=True)

    # ===== Debug (optioneel inklapbaar)
    with st.expander("🛠️ Debug"):
        dbg = {
            "period_token": period_token,
            "period_step": step,
            "shop_ids": shop_ids,
            "ref_spv": ref_spv,
            "csm2i_target": csm2i_target,
            "conv_goal_pct": conv_goal_pct,
            "open_start": open_start,
            "open_end": open_end,
        }
        st.json(dbg)
