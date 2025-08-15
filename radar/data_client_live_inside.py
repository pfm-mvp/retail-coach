import requests
from typing import Dict, List, Optional

class RadarAPIError(Exception):
    pass

def fetch_report(api_url: str, shop_ids: List[int], data_outputs: List[str],
                 date_from: str, date_to: str, period_step: str = "day",
                 show_hours_from: Optional[str] = None, show_hours_to: Optional[str] = None,
                 timeout: int = 60) -> Dict:
    if not api_url:
        raise RadarAPIError("Empty API_URL; set .streamlit/secrets.toml")
    if not shop_ids:
        raise RadarAPIError("shop_ids empty.")
    if not data_outputs:
        raise RadarAPIError("data_outputs empty.")

    params = {
        "source": "shops",
        "period": "date",
        "form_date_from": date_from,
        "form_date_to": date_to,
        "period_step": period_step,
    }
    for sid in shop_ids:
        params.setdefault("data", []).append(int(sid))
    for out in data_outputs:
        params.setdefault("data_output", []).append(out)

    if show_hours_from:
        params["show_hours_from"] = show_hours_from
    if show_hours_to:
        params["show_hours_to"] = show_hours_to

    resp = requests.post(f"{api_url.rstrip('/')}/get-report", params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RadarAPIError(f"Proxy returned {resp.status_code}: {resp.text[:400]}")
    try:
        return resp.json()
    except Exception as e:
        raise RadarAPIError(f"Invalid JSON: {e}")


def fetch_live_inside(api_url: str, location_ids: List[int], source: str = "locations", timeout: int = 30) -> Dict:
    """
    Fetch live inside data via the FastAPI proxy.
    Endpoint: POST {API_URL}/live-inside
    Query params (no [] in param names):
      - source = "locations" | "zones"
      - data   = repeated per location/zone id
    """
    if not api_url:
        raise RadarAPIError("Empty API_URL; set .streamlit/secrets.toml")
    if not location_ids:
        raise RadarAPIError("location_ids empty.")
    if source not in ("locations", "zones"):
        raise RadarAPIError("source must be 'locations' or 'zones'")

    # Build params without [] to match existing proxy convention
    params = {"source": source, "data": []}
    for loc_id in location_ids:
        params.setdefault("data", []).append(int(loc_id))

    resp = requests.post(f"{api_url.rstrip('/')}/live-inside", params=params, timeout=timeout)
    if resp.status_code != 200:
        raise RadarAPIError(f"Proxy returned {resp.status_code}: {resp.text[:400]}")
    try:
        return resp.json()
    except Exception as e:
        raise RadarAPIError(f"Invalid JSON: {e}")
