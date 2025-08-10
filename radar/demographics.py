
import pandas as pd

def has_gender_data(df: pd.DataFrame) -> bool:
    cols = [c for c in df.columns if c.startswith("demographics_gender")]
    if not cols: return False
    s = df[cols].fillna(0).sum().sum()
    return s > 0

def gender_insights(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Looks for time slots where gender share deviates from shop average
    and KPI (conversion/SPV) is above own average.
    """
    if df.empty: return pd.DataFrame()
    cols = [c for c in df.columns if c.startswith("demographics_gender")]
    if not cols: return pd.DataFrame()

    out_rows = []
    for (sid, sname), g in df.groupby(["shop_id","shop_name"], dropna=False):
        if g[cols].fillna(0).sum().sum() <= 0:
            continue
        g2 = g.copy()
        g2["female_share"] = g2.get("demographics_gender_female", 0) / (g2.get("demographics_gender_female", 0) + g2.get("demographics_gender_male", 0) + 1e-9)
        g2["male_share"] = 1 - g2["female_share"]
        conv_avg = g2.get("conversion_rate", 0).mean()
        spv_avg = g2.get("sales_per_visitor", 0).mean()
        # Deviation scoring
        g2["score"] = 0.0
        g2["score"] += (g2.get("conversion_rate",0) > conv_avg).astype(float)
        g2["score"] += (g2.get("sales_per_visitor",0) > spv_avg).astype(float)
        g2["score"] += (abs(g2["female_share"] - g2["female_share"].mean()) > 0.1).astype(float)  # >10pp shift

        top = g2.sort_values("score", ascending=False).head(top_n)
        for _, r in top.iterrows():
            out_rows.append(dict(
                shop_id=int(sid),
                shop_name=sname,
                period=str(r.get("period_label","")),
                male_share=float(r.get("male_share",0)),
                female_share=float(r.get("female_share",0)),
                conv=float(r.get("conversion_rate",0)),
                spv=float(r.get("sales_per_visitor",0)),
            ))
    return pd.DataFrame(out_rows)
