
import pandas as pd

def top_performers(df: pd.DataFrame, kpi: str = "sales_per_visitor", n: int = 5) -> pd.DataFrame:
    if df.empty or kpi not in df.columns:
        return pd.DataFrame()
    g = df.groupby(["shop_id","shop_name"], dropna=False)[kpi].mean().sort_values(ascending=False).head(n)
    return g.reset_index().rename(columns={kpi: f"{kpi}_avg"})
