from typing import Dict
import pandas as pd

def normalize(resp_json: Dict, ts_key: str = "timestamp"):
    rows = []
    data_block = resp_json.get("data") or resp_json
    for date_key, per_shop in (data_block or {}).items():
        if not isinstance(per_shop, dict):
            continue
        for shop_id, shop_obj in per_shop.items():
            if not isinstance(shop_obj, dict):
                continue
            meta = shop_obj.get("meta", {})
            shop_name = meta.get("shop_name", str(shop_id))
            dates = shop_obj.get("dates") or {}
            for ts, d in dates.items():
                kpis = d.get("data") or {}
                row = {"shop_id": int(str(shop_id)), "shop_name": shop_name, ts_key: ts}
                for k, v in kpis.items():
                    try:
                        row[k] = float(v) if v is not None else 0.0
                    except Exception:
                        row[k] = 0.0
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    ts = pd.to_datetime(df[ts_key], errors="coerce")
    df["date"] = ts.dt.date
    df["hour"] = ts.dt.hour
    df["period_label"] = df[ts_key]
    return df
