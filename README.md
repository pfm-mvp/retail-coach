
# Retail Performance Radar – Full Bundle

## Quickstart
1) Vul **shop_mapping.py** in de root:
```python
SHOP_ID_TO_NAME = {37953: "Putte", 37952: "Den Bosch"}
```
2) Zet je proxy in `.streamlit/secrets.toml`:
```toml
API_URL = "https://<fastapi-proxy>/get-report"
```
3) Start via Streamlit (launcher) of met:
```bash
pip install -r requirements.txt
streamlit run pages/10_Retail_Performance_Radar.py
```

- POST met query params (geen `[]`).
- `data_output` bevat `sqm_meter` en (optioneel) `demographics_gender_*`.
- Demografie-sectie verschijnt **alleen** als er data > 0 is.
