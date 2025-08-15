import streamlit as st
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # project root = parent van /pages
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------

from radar.data_client_live_inside import fetch_live_inside  # uses copy with live-inside added
from shop_mapping import SHOP_NAME_MAP

st.set_page_config(page_title="Live Inside Test", layout="wide")
st.title("🔴 Live Inside – quick test")

API_URL = st.secrets.get("API_URL", "").rstrip("/")
if not API_URL:
    st.error("API_URL missing in secrets. Add to .streamlit/secrets.toml")
    st.stop()

# drop-down with names, but submit IDs
options = [(name, sid) for sid, name in SHOP_NAME_MAP.items()]
if not options:
    st.warning("SHOP_NAME_MAP is empty.")
location_label, location_id = st.selectbox(
    "Kies locatie",
    options=options,
    format_func=lambda x: x[0] if isinstance(x, tuple) else x,
    index=0 if options else None,
)

col1, col2 = st.columns([1, 1])
with col1:
    st.caption("Source")
    source = st.radio("Bron (Vemcount)", options=["locations","zones"], index=0, horizontal=True, label_visibility="collapsed")
with col2:
    st.caption("Actie")
    run = st.button("🔄 Fetch Live Inside", use_container_width=True)

if run:
    try:
        data = fetch_live_inside(API_URL, [int(location_id)], source=source)
        st.success("Live inside response ontvangen ✅")
        st.json(data)
    except Exception as e:
        st.error(f"Fout: {e}")
