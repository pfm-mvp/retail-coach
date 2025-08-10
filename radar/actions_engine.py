
from typing import List, Dict, Optional
import pandas as pd

def safe_div(n, d):
    try:
        n = float(n); d = float(d)
        return n / d if d != 0 else 0.0
    except Exception:
        return 0.0

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "conversion_rate" in out.columns:
        cr = out["conversion_rate"].astype(float)
        out["conversion_rate"] = cr/100.0 if cr.max() > 1.5 else cr
    else:
        out["conversion_rate"] = out.apply(lambda r: safe_div(r.get("transactions",0), r.get("count_in",0)), axis=1)

    if "sales_per_visitor" not in out.columns:
        out["sales_per_visitor"] = out.apply(lambda r: safe_div(r.get("turnover",0), r.get("count_in",0)), axis=1)
    if "atv" not in out.columns:
        out["atv"] = out.apply(lambda r: safe_div(r.get("turnover",0), r.get("transactions",0)), axis=1)
    return out.fillna(0.0)

def impact_severity(eur: float) -> str:
    if eur >= 2000: return "red"
    if eur >= 500: return "orange"
    return "green"

def rule_high_traffic_low_conv(df: pd.DataFrame, conv_target: Optional[float]) -> List[Dict]:
    res = []
    if df.empty: return res
    for (sid, sname), g in df.groupby(["shop_id","shop_name"], dropna=False):
        if g["count_in"].sum() <= 0: continue
        thr = g["count_in"].quantile(0.75)
        mean = g["conversion_rate"].mean()
        std = g["conversion_rate"].std(ddof=0) or 1e-9
        target = conv_target if isinstance(conv_target,(int,float)) else mean
        sel = g[(g["count_in"]>=thr) & ((g["conversion_rate"]-mean)/std <= -0.5)]
        for _, r in sel.iterrows():
            delta_conv = max(0.0, target - r["conversion_rate"])
            delta = r["count_in"] * r["atv"] * delta_conv
            res.append(dict(shop_id=int(sid), shop_name=sname,
                            issue="Veel traffic × lage conversie",
                            action=f"Extra verkoper/promo in dit tijdvak; til conversie naar {target:.1%}.",
                            expected_impact_eur=float(delta),
                            severity=impact_severity(delta),
                            period=str(r.get("period_label",""))))
    return res

def rule_low_spv_ok_conv(df: pd.DataFrame, spv_uplift: float) -> List[Dict]:
    res = []
    if df.empty: return res
    for (sid, sname), g in df.groupby(["shop_id","shop_name"], dropna=False):
        if g["count_in"].sum() <= 0: continue
        thr = g["sales_per_visitor"].quantile(0.25)
        target_spv = g["sales_per_visitor"].mean() * (1 + spv_uplift)
        sel = g[(g["sales_per_visitor"]<=thr) & (g["conversion_rate"]>=0.03)]
        for _, r in sel.iterrows():
            delta_spv = max(0.0, target_spv - r["sales_per_visitor"])
            delta = r["count_in"] * delta_spv
            res.append(dict(shop_id=int(sid), shop_name=sname,
                            issue="Lage SPV bij oké conversie",
                            action=f"Bundel/upsell bij entree/kassa; mik op +{int(spv_uplift*100)}% SPV.",
                            expected_impact_eur=float(delta),
                            severity=impact_severity(delta),
                            period=str(r.get("period_label",""))))
    return res

def add_csm2i(df: pd.DataFrame, target_index: float = 1.0) -> pd.DataFrame:
    out = df.copy()
    if "sqm_meter" not in out.columns:
        out["sqm_meter"] = 0.0
    out["sales_per_sqm"] = out.apply(lambda r: safe_div(r.get("turnover",0), r.get("sqm_meter",0)), axis=1)
    bench = out.groupby("shop_id")["sales_per_sqm"].transform("mean").replace([None], 0.0)
    out["csm2i"] = out.apply(lambda r: safe_div(r["sales_per_sqm"], bench.loc[r.name] if r.name in bench.index else 0.0), axis=1)
    out["csm2i_gap"] = out.apply(lambda r: max(0.0, target_index - (r.get('csm2i') if r.get('csm2i')==r.get('csm2i') else 0.0)), axis=1)
    out["csm2i_uplift_eur"] = out["csm2i_gap"] * bench * out["sqm_meter"]
    return out

def generate_actions(df: pd.DataFrame, conv_target: Optional[float], spv_uplift: float, csm2i_target: float) -> pd.DataFrame:
    base = enrich(df)
    base = add_csm2i(base, csm2i_target)
    items = []
    items += rule_high_traffic_low_conv(base, conv_target)
    items += rule_low_spv_ok_conv(base, spv_uplift)
    for (sid, sname), g in base.groupby(["shop_id","shop_name"], dropna=False):
        miss = g[g["csm2i_gap"] > 0]
        if miss.empty: continue
        delta = miss["csm2i_uplift_eur"].sum()
        items.append(dict(shop_id=int(sid), shop_name=sname,
                          issue="CSm²I onder target",
                          action=f"Optimaliseer vloerbenutting/flow richting topwinkelniveau.",
                          expected_impact_eur=float(delta),
                          severity=impact_severity(delta),
                          period="—"))
    if not items:
        return pd.DataFrame(columns=["shop_id","shop_name","issue","action","expected_impact_eur","severity","period"])
    out = pd.DataFrame(items).sort_values("expected_impact_eur", ascending=False).reset_index(drop=True)
    return out
