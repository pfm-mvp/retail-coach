# pages/10_Retail_Performance_Radar.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta
import plotly.express as px

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

def fmt_eur(x: float) -> str:
    return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")

def coerce_numeric(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
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
        fallback = med if pd.notnull(med) and med > 0 else DEFAULT_SQ_METER
        out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)
    else:
        out["sq_meter"] = DEFAULT_SQ_METER
    return out

# 🔧 robuuste referentie‑SPV
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
c1, c2, c3 = st.columns([1,1,1])
with c1:
    period_label = st.selectbox("Periode", ["7 dagen", "30 dagen", "last_month"], index=0)
with c2:
    gran = st.selectbox("Granulariteit", ["Dag", "Uur"], index=0)
with c3:
    pass

today = date.today()
if period_label == "last_month":
    first = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last  = today.replace(day=1) - timedelta(days=1)
    date_from, date_to = first, last
elif period_label == "30 dagen":
    date_from, date_to = today - timedelta(days=30), today - timedelta(days=1)
else:
    date_from, date_to = today - timedelta(days=7), today - timedelta(days=1)

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

# =========================
# API helpers
# =========================
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

# =========================
# RUN
# =========================
if analyze:
    if not shop_ids:
        st.warning("Selecteer minimaal één winkel."); st.stop()
    API_URL = st.secrets.get("API_URL","")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml"); st.stop()

    step = "hour" if gran.lower().startswith("u") else "day"
    outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Data ophalen…"):
        resp = fetch_report(API_URL, shop_ids, date_from, date_to, step, outputs)
        df = normalize_resp(resp)
        if df.empty:
            st.info("Geen data beschikbaar voor de gekozen periode."); st.stop()

    # Referentie‑SPV (portfolio + uplift)
    mode = "portfolio"
    bm_id = None
    ref_spv = choose_ref_spv(df, mode=mode, benchmark_shop_id=bm_id, uplift_pct=spv_uplift_pct/100.0)

    # Uniforme CSm²I + uplift
    df = compute_csm2i_and_uplift(df, ref_spv=ref_spv, csm2i_target=csm2i_target)

    # Conversie‑uplift
    conv_target = float(conv_goal_pct) / 100.0
    df["uplift_eur_conv"] = np.maximum(0.0, (conv_target - df["conversion_rate"]) * df["count_in"]) * df["atv"]

    # Aggregatie per winkel
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

    # --- SPV benchmark t.o.v. beste winkel ---
    spv_by_shop = df.groupby(["shop_id", "shop_name"])["actual_spv"].mean().reset_index()
    best_row = spv_by_shop.loc[spv_by_shop["actual_spv"].idxmax()]
    best_shop_name = str(best_row["shop_name"])
    best_spv = float(best_row["actual_spv"])
    agg = agg.merge(spv_by_shop.rename(columns={"actual_spv": "spv_current"}), on=["shop_id", "shop_name"], how="left")
    agg["uplift_vs_best"] = np.maximum(0.0, (best_spv - agg["spv_current"])) * agg["visitors"]
    total_uplift_vs_best = float(agg["uplift_vs_best"].sum())

    # ===== KPI‑tegels =====
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
    # Extra SPV‑tegel
    st.markdown(
        f"""<div class="card" style="margin-top:10px;">
              <div>📈 <b>SPV vs Topwinkel</b><br/>
              <small>Beste: {best_shop_name} (SPV {("€{:,.2f}".format(best_spv)).replace(",", "X").replace(".", ",").replace("X",".")})</small></div>
              <div class="kpi eur">{fmt_eur(total_uplift_vs_best)}</div>
            </div>""",
        unsafe_allow_html=True
    )

    # --- Oranje total widget ---
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    total_eur = agg["uplift_total"].sum()
    st.markdown(
        f"""
        <div style="
            margin: 6px 0 18px 0;
            padding: 16px 18px;
            background: #FEAC76;
            border-radius: 12px;
            border: 1px solid rgba(0,0,0,0.06);
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        ">
          <div style="font-weight:700; font-size:15px; margin-bottom:6px;">💰 Total extra potential in revenue</div>
          <div style="font-size:26px; font-weight:800; line-height:1;" class="eur">{fmt_eur(total_eur)}</div>
          <div style="opacity:.85; margin-top:4px;">
            Som van CSm²I‑ en conversie‑potentieel voor de geselecteerde periode.
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Aanbevelingen per winkel (incl. SPV/beste, Sales/m²/verwacht, Conversie/doel) ---
    st.markdown("## 🔎 Aanbevelingen per winkel")
    store_stats = df.groupby(["shop_id","shop_name"]).agg(
        visitors=("count_in","sum"),
        sqm=("sq_meter","mean"),
        atv=("atv","mean"),
        conv=("conversion_rate","mean"),
        spv=("actual_spv","mean"),
        spsqm=("actual_spsqm","mean"),
        expected_spsqm=("expected_spsqm","mean"),
        uplift_csm=("uplift_eur_csm","sum"),
        uplift_conv=("uplift_eur_conv","sum"),
    ).reset_index()
    store_stats["csm2i_current"] = store_stats["spv"] / (float(ref_spv) + EPS)
    store_stats["uplift_total"]  = store_stats["uplift_csm"] + store_stats["uplift_conv"]
    conv_target_f = float(conv_goal_pct)/100.0

    def eur0(x): return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X",".")
    def idx2(v):  return f"{v:.2f}".replace(".", ",")

    for _, r in store_stats.sort_values("uplift_total", ascending=False).iterrows():
        ok_csm  = r["csm2i_current"] >= float(csm2i_target)
        ok_sqm  = r["spsqm"]         >= r["expected_spsqm"] - 1e-6
        ok_conv = r["conv"]          >= conv_target_f
        ok_spv_best = r["spv"]       >= best_spv - 1e-6
        b = lambda ok: "🟢" if ok else "🟠"

        tips = []
        if not ok_csm:  tips.append("Verhoog SPV (bundels/upsell, kassa‑acties).")
        if not ok_sqm:  tips.append("Benut m² beter (hotzones, productmix, zichtbaardere acties).")
        if not ok_conv: tips.append("Conversie‑push (extra bezetting piek, actieve benadering, drempelpromos).")
        if not ok_spv_best: tips.append("Spiegel aan topwinkel‑SPV (vaardigheden/aanbod/acties).")
        if not tips:
            tips = ["Behoud performance; borg best practices."]

        spv_best_eur = ("€{:,.2f}".format(best_spv)).replace(",", "X").replace(".", ",").replace("X",".")
        spv_cur_eur  = ("€{:,.2f}".format(r['spv'])).replace(",", "X").replace(".", ",").replace("X",".")
        spsqm_cur    = ("€{:,.2f}".format(r['spsqm'])).replace(",", "X").replace(".", ",").replace("X",".")
        spsqm_exp    = ("€{:,.2f}".format(r['expected_spsqm'])).replace(",", "X").replace(".", ",").replace("X",".")

        st.markdown(
            f"""
            <div style="border:1px solid #eee; border-radius:12px; padding:16px; margin:10px 0; background:#fff;">
              <div style="font-weight:700; font-size:20px; margin-bottom:8px;">{r['shop_name']}</div>

              <div style="display:grid; grid-template-columns: repeat(3,minmax(220px,1fr)); gap:18px; align-items:start;">
                <div>
                  <div style="opacity:.75; font-size:13px;">CSm²I (huidig / target)</div>
                  <div style="font-weight:800; font-size:22px;">{idx2(r['csm2i_current'])} / {idx2(float(csm2i_target))} <span>{b(ok_csm)}</span></div>
                </div>
                <div>
                  <div style="opacity:.75; font-size:13px;">Sales per m² (huidig / verwacht)</div>
                  <div style="font-weight:800; font-size:22px;">{spsqm_cur} / {spsqm_exp} <span>{b(ok_sqm)}</span></div>
                </div>
                <div>
                  <div style="opacity:.75; font-size:13px;">Conversie (huidig / doel)</div>
                  <div style="font-weight:800; font-size:22px;">{idx2(r['conv']*100)}% / {idx2(conv_target_f*100)}% <span>{b(ok_conv)}</span></div>
                </div>
              </div>

              <div style="margin-top:8px; opacity:.85;">
                <b>SPV (huidig / topwinkel):</b> {spv_cur_eur} / {spv_best_eur} <span>{b(ok_spv_best)}</span>
              </div>

              <div style="margin-top:12px; display:flex; gap:24px; align-items:center; flex-wrap:wrap;">
                <div>
                  <div style="opacity:.75; font-size:13px;">Potentiële uplift (totaal)</div>
                  <div style="font-weight:800; font-size:24px;" class="eur">{eur0(r['uplift_total'])}</div>
                </div>
                <div style="opacity:.95;">✅ { ' '.join(tips) }</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ===== Scatter (laatste visual): SPV vs Sales per m² =====
    rad = df.groupby(["shop_id", "shop_name"]).agg(
        spv=("actual_spv", "mean"),
        spsqm=("actual_spsqm", "mean"),
        csi=("csm2i", "mean"),
        visitors=("count_in", "sum"),
    ).reset_index()

    if "uplift_total" in agg.columns:
        size_series = agg.set_index("shop_id")["uplift_total"]
        rad["size_metric"] = rad["shop_id"].map(size_series).fillna(0.0)
        size_label = "Uplift"
        rad["hover_size"] = rad["size_metric"].round(0).map(
            lambda v: ("€{:,.0f}".format(v)).replace(",", "X").replace(".", ",").replace("X", ".")
        )
    else:
        rad["size_metric"] = rad["visitors"].fillna(0.0)
        size_label = "Bezoekers"
        rad["hover_size"] = rad["size_metric"].round(0).map(
            lambda v: "{:,.0f}".format(v).replace(",", ".")
        )

    low_thr  = float(csm2i_target) * 0.95
    high_thr = float(csm2i_target) * 1.05
    rad["csm2i_band"] = np.select(
        [rad["csi"] < low_thr, rad["csi"] > high_thr],
        ["Onder target", "Boven target"],
        default="Rond target",
    )

    rad["hover_spv"]   = rad["spv"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_spsqm"] = rad["spsqm"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    rad["hover_csi"]   = rad["csi"].round(2).map(lambda v: str(v).replace(".", ","))

    color_map  = {"Onder target": "#F04438", "Rond target": "#F59E0B", "Boven target": "#16A34A"}
    symbol_map = {"Onder target": "diamond", "Rond target": "circle", "Boven target": "square"}

    scatter = px.scatter(
        rad,
        x="spv", y="spsqm", size="size_metric",
        color="csm2i_band", symbol="csm2i_band",
        color_discrete_map=color_map, symbol_map=symbol_map,
        hover_data=None,
        custom_data=rad[["hover_spv","hover_spsqm","hover_csi","hover_size"]],
        labels={"spv": "Sales per Visitor", "spsqm": "Sales per m²", "csm2i_band": "CSm²I t.o.v. target"},
    )
    scatter.update_traces(
        hovertemplate="<b>%{text}</b><br>"
                      "SPV: %{customdata[0]}<br>"
                      "Sales per m²: %{customdata[1]}<br>"
                      "CSm²I: %{customdata[2]}<br>"
                      f"{size_label}: " + "%{customdata[3]}<extra></extra>",
        text=rad["shop_name"],
    )
    scatter.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        height=560,
        legend_title_text="CSm²I t.o.v. target",
        xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
        yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"),
    )
    st.plotly_chart(scatter, use_container_width=True)

    # ===== Debug (optioneel inklapbaar)
    with st.expander("🛠️ Debug"):
        dbg = {
            "period_step": step,
            "from": str(date_from),
            "to": str(date_to),
            "shop_ids": shop_ids,
            "ref_spv": ref_spv,
            "csm2i_target": csm2i_target,
            "conv_goal_pct": conv_goal_pct,
            "best_shop_name": best_shop_name,
            "best_spv": best_spv,
        }
        st.json(dbg)
