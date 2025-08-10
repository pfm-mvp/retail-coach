import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import date, timedelta

# ========== Page & Styling ==========
st.set_page_config(page_title="Retail Performance Radar", page_icon="📊", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Instrument Sans', sans-serif !important; }
/* 🎨 Multiselect pills in paars */
[data-baseweb="tag"] { background-color: #9E77ED !important; color: white !important; }
/* 🔴 Analyseer knop in PFM-rood */
button[data-testid="stBaseButton-secondary"] {
  background-color: #F04438 !important; color: white !important;
  border-radius: 16px !important; font-weight: 600 !important;
  padding: 0.6rem 1.4rem !important; border: none !important; box-shadow: none !important;
  transition: background-color 0.2s ease-in-out;
}
button[data-testid="stBaseButton-secondary"]:hover {
  background-color: #d13c30 !important; cursor: pointer;
}
/* Cards */
.card { border: 1px solid #eee; border-radius: 12px; padding: 14px 16px; background:#fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.card h4 { margin: 0 0 6px 0; font-size: 1.0rem; }
.card .muted { color:#666; font-size: 0.85rem; }
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.badge.green { background:#E6F4EA; color:#137333; }
.badge.orange{ background:#FFF4E5; color:#A04A00; }
.badge.red   { background:#FEECEA; color:#B42318; }
.kpi { font-size: 1.2rem; font-weight:700; }
.eur { font-variant-numeric: tabular-nums; }
</style>
""", unsafe_allow_html=True)

# ========== Helpers ==========
EPS = 1e-9
DEFAULT_SQ_METER = 1.0

def fmt_eur(x: float) -> str:
    return ("€{:,.0f}".format(float(x))).replace(",", "X").replace(".", ",").replace("X", ".")

def sev_badge(sev: str) -> str:
    cls = {"red":"red","orange":"orange"}.get(str(sev), "green")
    return f'<span class="badge {cls}">{cls.capitalize()}</span>'

# ========== Shop mapping ==========
try:
    from shop_mapping import SHOP_NAME_MAP as _MAP  # {id:int: "Naam":str}
except Exception:
    _MAP = {}
SHOP_ID_TO_NAME = {int(k): str(v) for k, v in _MAP.items() if str(v).strip()}
NAME_TO_ID = {v: k for k, v in SHOP_ID_TO_NAME.items()}

# ========== Titel & inputs ==========
st.title("📊 Retail Performance Radar")

today = date.today()
default_from = today - timedelta(days=7)
c1, c2 = st.columns(2)
with c1: date_from = st.date_input("Van", default_from)
with c2: date_to   = st.date_input("Tot", today - timedelta(days=1))

names = sorted(NAME_TO_ID.keys(), key=str.lower)
selected_names = st.multiselect("Selecteer winkels", names, default=names[:1], key="shop_selector")
shop_ids = [NAME_TO_ID[n] for n in selected_names]

c3, c4 = st.columns(2)
with c3:
    gran = st.selectbox("Granulariteit", ["Dag","Uur"], index=0)
    period_step = "day" if gran == "Dag" else "hour"
with c4:
    # Sliders (incl. SPV-uplift slider terug)
    conv_target_pct = st.slider("Conversiedoel (%)", 1, 80, 20, 1)
    spv_uplift_pct  = st.slider("SPV‑uplift (%)", 5, 50, 10, 5)
c5, c6 = st.columns(2)
with c5:
    csm2i_target = st.slider("CSm²I‑target", 0.10, 2.00, 1.00, 0.05)

analyze = st.button("🔍 Analyseer", key="analyze_button", type="secondary")

# ========== API & Normalizer ==========
def fetch_report_inline(api_url, shop_ids, date_from, date_to, period_step, data_outputs, timeout=30):
    params = [
        ("source", "shops"),
        ("period", "date"),
        ("form_date_from", str(date_from)),
        ("form_date_to", str(date_to)),
        ("period_step", period_step)
    ]
    for sid in shop_ids:      params.append(("data", int(sid)))
    for out in data_outputs:  params.append(("data_output", out))
    r = requests.post(api_url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def normalize(resp, shop_map):
    rows = []
    data = resp.get("data", {})
    for _, shops in data.items():
        for sid, payload in shops.items():
            meta = payload.get("meta", {})
            dates = payload.get("dates", {})
            for ts, obj in dates.items():
                rec = {"timestamp": ts, "shop_id": int(sid), "shop_name": shop_map.get(int(sid), str(sid))}
                rec.update(obj.get("data", {}))
                rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty: return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    return df

# ========== KPI verrijking ==========
def ensure_basics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # 1) Numeriek forceren
    for col in ["turnover","transactions","count_in","sales_per_visitor","conversion_rate","sq_meter"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    # 2) Conversie naar fractie (niet naar %)
    if "conversion_rate" in out.columns and not out["conversion_rate"].empty:
        if out["conversion_rate"].max() > 1.5:  # waarschijnlijk in %
            out["conversion_rate"] = out["conversion_rate"] / 100.0
    else:
        out["conversion_rate"] = out.get("transactions", 0.0) / (out.get("count_in", 0.0) + EPS)
    # 3) SPV & ATV
    if "sales_per_visitor" not in out.columns or out["sales_per_visitor"].isna().all():
        out["sales_per_visitor"] = out.get("turnover", 0.0) / (out.get("count_in", 0.0) + EPS)
    out["atv"] = out.get("turnover", 0.0) / (out.get("transactions", 0.0) + EPS)
    return out

def add_csm2i(df: pd.DataFrame, target_index: float = 1.0) -> pd.DataFrame:
    out = ensure_basics(df)
    sqm = pd.to_numeric(out.get("sq_meter"), errors="coerce")
    median_sqm = sqm.replace(0, np.nan).median()
    fallback = median_sqm if pd.notnull(median_sqm) and median_sqm > 0 else DEFAULT_SQ_METER
    out["sq_meter"] = sqm.replace(0, np.nan).fillna(fallback)

    out["visitors_per_sqm"] = out["count_in"] / (out["sq_meter"] + EPS)
    out["actual_spsqm"]     = out["turnover"]  / (out["sq_meter"] + EPS)
    out["expected_spsqm"]   = out["sales_per_visitor"] * out["visitors_per_sqm"]
    out["csm2i"]            = out["actual_spsqm"] / (out["expected_spsqm"] + EPS)
    out["uplift_eur_csm"]   = np.maximum(0.0, (target_index * out["expected_spsqm"] - out["actual_spsqm"])) * out["sq_meter"]
    return out

# ========== Next Best Action rules ==========
def impact_severity(eur: float) -> str:
    if eur >= 2000: return "red"
    if eur >=  500: return "orange"
    return "green"

def rule_high_traffic_low_conv(df: pd.DataFrame, conv_target: float) -> pd.DataFrame:
    items = []
    for (sid, sname), g in df.groupby(["shop_id","shop_name"], dropna=False):
        if g["count_in"].sum() <= 0: continue
        thr  = g["count_in"].quantile(0.75)
        mean = g["conversion_rate"].mean()
        std  = g["conversion_rate"].std(ddof=0) or 1e-9
        target = conv_target if isinstance(conv_target,(int,float)) else mean
        sel = g[(g["count_in"]>=thr) & ((g["conversion_rate"]-mean)/std <= -0.5)]
        for _, r in sel.iterrows():
            delta_conv = max(0.0, target - r["conversion_rate"])
            delta = r["count_in"] * r["atv"] * delta_conv
            items.append(dict(
                shop_id=int(sid), shop_name=sname, period=str(r.get("timestamp","")),
                issue="Veel traffic × lage conversie",
                action=f"Zet extra verkoper/promo in dit tijdvak; til conversie naar {target:.1%}.",
                expected_impact_eur=float(delta), severity=impact_severity(delta)
            ))
    return pd.DataFrame(items)

def rule_low_spv_ok_conv(df: pd.DataFrame, spv_uplift: float = 0.10) -> pd.DataFrame:
    items = []
    for (sid, sname), g in df.groupby(["shop_id","shop_name"], dropna=False):
        if g["count_in"].sum() <= 0: continue
        thr = g["sales_per_visitor"].quantile(0.25)
        target_spv = g["sales_per_visitor"].mean() * (1 + spv_uplift)
        sel = g[(g["sales_per_visitor"]<=thr) & (g["conversion_rate"]>=0.03)]
        for _, r in sel.iterrows():
            delta_spv = max(0.0, target_spv - r["sales_per_visitor"])
            delta = r["count_in"] * delta_spv
            items.append(dict(
                shop_id=int(sid), shop_name=sname, period=str(r.get("timestamp","")),
                issue="Lage SPV bij oké conversie",
                action=f"Bundel/upsell bij entree/kassa; mik op +{int(spv_uplift*100)}% SPV.",
                expected_impact_eur=float(delta), severity=impact_severity(delta)
            ))
    return pd.DataFrame(items)

def rule_csm2i_gap(df: pd.DataFrame, target_index: float = 1.0) -> pd.DataFrame:
    items = []
    for (sid, sname), g in df.groupby(["shop_id","shop_name"], dropna=False):
        gap = (target_index - g["csm2i"]).clip(lower=0)
        delta = (gap * g["expected_spsqm"] * g["sq_meter"]).sum()
        if delta > 0:
            items.append(dict(
                shop_id=int(sid), shop_name=sname, period="—",
                issue="CSm²I onder target",
                action="Optimaliseer vloerbenutting/flow richting topwinkelniveau.",
                expected_impact_eur=float(delta), severity=impact_severity(delta)
            ))
    return pd.DataFrame(items)

def generate_actions(df: pd.DataFrame, conv_target: float, spv_uplift: float, csm2i_target: float) -> pd.DataFrame:
    parts = [
        rule_high_traffic_low_conv(df, conv_target),
        rule_low_spv_ok_conv(df, spv_uplift),
        rule_csm2i_gap(df, csm2i_target)
    ]
    parts = [p for p in parts if not p.empty]
    if not parts: return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values("expected_impact_eur", ascending=False).reset_index(drop=True)

# ========== Best Practice Finder ==========
def top_performers_enriched(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    g = df.groupby(["shop_id","shop_name"], dropna=False).agg(
        **{
            "SPV (gem)": ("sales_per_visitor","mean"),
            "Conversie (gem)": ("conversion_rate","mean"),
            "ATV (gem)": ("atv","mean"),
            "CSm²I (gem)": ("csm2i","mean"),
            "Bezoekers (som)": ("count_in","sum"),
        }
    ).reset_index()
    if g.empty: return g
    g["SPV (gem)"]       = g["SPV (gem)"].round(2)
    g["ATV (gem)"]       = g["ATV (gem)"].round(2)
    g["CSm²I (gem)"]     = g["CSm²I (gem)"].round(2)
    g["Conversie (gem)"] = (g["Conversie (gem)"]*100).round(2)
    return g.sort_values("SPV (gem)", ascending=False).head(n)

# ========== Run ==========
if analyze:
    if not shop_ids:
        st.info("Selecteer minimaal één winkel.")
        st.stop()

    API_URL = st.secrets.get("API_URL", "")
    if not API_URL:
        st.warning("Stel `API_URL` in via .streamlit/secrets.toml (bijv. https://…/get-report).")
        st.stop()

    data_outputs = ["count_in","transactions","turnover","conversion_rate","sales_per_visitor","sq_meter"]

    with st.spinner("Analyseren…"):
        try:
            resp = fetch_report_inline(API_URL, shop_ids, date_from, date_to, period_step, data_outputs)
            df = normalize(resp, SHOP_ID_TO_NAME)
        except Exception as e:
            st.error(f"Fout bij ophalen data: {e}")
            st.stop()

    if df.empty:
        st.info("Geen data voor de gekozen filters/periode.")
        st.stop()

    # Verrijk KPI's en CSm²I
    df = ensure_basics(df)
    df = add_csm2i(df, target_index=csm2i_target)

    # ===== 📐 CSm²I impact cards =====
    st.markdown("### 📐 CSm²I impact (per winkel)")
    csi_by_shop = df.groupby(["shop_id","shop_name"], dropna=False).agg(
        csi=("csm2i","mean"),
        uplift=("uplift_eur_csm","sum")
    ).reset_index().sort_values("uplift", ascending=False)
    if csi_by_shop.empty:
        st.info("Geen CSm²I-data beschikbaar voor deze selectie.")
    else:
        cols_per_row = 3
        rows = (len(csi_by_shop) + cols_per_row - 1) // cols_per_row
        for r in range(rows):
            slice_ = csi_by_shop.iloc[r*cols_per_row:(r+1)*cols_per_row]
            cols = st.columns(len(slice_))
            for col, (_, rec) in zip(cols, slice_.iterrows()):
                badge = "green" if rec.csi >= csm2i_target else ("orange" if rec.csi >= csm2i_target*0.9 else "red")
                col.markdown(f"""
                    <div class="card">
                      <h4>{rec.shop_name}</h4>
                      <div class="muted">CSm²I huidig vs target</div>
                      <div class="kpi">{rec.csi:.2f} / {csm2i_target:.2f} &nbsp;
                        <span class="badge {badge}">{'≥' if rec.csi>=csm2i_target else '<'}</span>
                      </div>
                      <div class="muted">Potentiële uplift</div>
                      <div class="kpi eur">{fmt_eur(rec.uplift)}</div>
                    </div>
                """, unsafe_allow_html=True)

    # ===== 🎯 Top 3 verbeterpunten per winkel =====
    st.markdown("### 🎯 Top 3 verbeterpunten per winkel")
    actions = generate_actions(
        df,
        conv_target=conv_target_pct/100.0,
        spv_uplift=spv_uplift_pct/100.0,  # <-- slider gebruikt
        csm2i_target=csm2i_target
    )
    if actions.empty:
        st.info("Geen urgente verbeterpunten gevonden voor deze selectie.")
    else:
        for sid, sdf in actions.groupby("shop_id", sort=False):
            sname = sdf["shop_name"].iloc[0]
            st.markdown(f"#### {sname}")
            top3 = sdf.sort_values("expected_impact_eur", ascending=False).head(3)
            cols = st.columns(3) if len(top3) >= 3 else st.columns(len(top3))
            for col, (_, row) in zip(cols, top3.iterrows()):
                col.markdown(f"""
                    <div class="card">
                      <h4>{row.issue}</h4>
                      <div class="muted">{row.period}</div>
                      <div style="margin:6px 0 8px 0;">{row.action}</div>
                      <div>{sev_badge(row.severity)} &nbsp; <span class="eur">{fmt_eur(row.expected_impact_eur)}</span></div>
                    </div>
                """, unsafe_allow_html=True)

    # ===== 🏆 Best Practice Finder =====
    st.markdown("### 🏆 Best Practice Finder")
    bp = top_performers_enriched(df, n=5)
    if bp.empty:
        st.info("Onvoldoende data voor Best Practices.")
    else:
        st.dataframe(bp, use_container_width=True)