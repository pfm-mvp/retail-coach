# ===== Scatter: SPV vs Sales per m² (duidelijk, met CSm²I vs target) =====
rad = df.groupby(["shop_id", "shop_name"]).agg(
    spv=("actual_spv", "mean"),
    spsqm=("actual_spsqm", "mean"),
    csi=("csm2i", "mean"),
    visitors=("count_in", "sum"),
).reset_index()

# Bubbelgrootte: uplift_total als die er is, anders bezoekers
if 'agg' in locals() and isinstance(agg, pd.DataFrame) and "uplift_total" in agg.columns:
    size_series = agg.set_index("shop_id")["uplift_total"]
    rad["size_metric"] = rad["shop_id"].map(size_series).fillna(0.0)
else:
    rad["size_metric"] = rad["visitors"].fillna(0.0)

# Band t.o.v. target (±5%)
low_thr  = float(csm2i_target) * 0.95
high_thr = float(csm2i_target) * 1.05
rad["csm2i_band"] = np.select(
    [rad["csi"] < low_thr, rad["csi"] > high_thr],
    ["Onder target", "Boven target"],
    default="Rond target",
)

# Nette hover‑waarden
rad["spv_fmt"]   = rad["spv"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
rad["spsqm_fmt"] = rad["spsqm"].round(2).apply(lambda v: ("€{:,.2f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
rad["csi_fmt"]   = rad["csi"].round(2).astype(str).str.replace(".", ",")
rad["size_fmt"]  = (
    rad["size_metric"].round(0).map(lambda v: ("€{:,.0f}".format(v)).replace(",", "X").replace(".", ",").replace("X", "."))
    if ('agg' in locals() and isinstance(agg, pd.DataFrame) and "uplift_total" in agg.columns)
    else rad["size_metric"].round(0).map(lambda v: "{:,.0f}".format(v).replace(",", "."))
)

color_map  = {"Onder target": "#F04438", "Rond target": "#F59E0B", "Boven target": "#16A34A"}
symbol_map = {"Onder target": "diamond", "Rond target": "circle", "Boven target": "square"}

rad_scatter = px.scatter(
    rad,
    x="spv", y="spsqm", size="size_metric",
    color="csm2i_band", symbol="csm2i_band",
    color_discrete_map=color_map, symbol_map=symbol_map,
    hover_data={
        "shop_name": True, "spv": False, "spsqm": False, "csi": False, "size_metric": False,
        "spv_fmt": True, "spsqm_fmt": True, "csi_fmt": True, "size_fmt": True,
    },
    labels={"spv": "Sales per Visitor", "spsqm": "Sales per m²", "csm2i_band": "CSm²I t.o.v. target"},
)

hover_title = "Uplift" if ('agg' in locals() and isinstance(agg, pd.DataFrame) and "uplift_total" in agg.columns) else "Bezoekers"
rad_scatter.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>"
                  "SPV: %{customdata[5]}<br>"
                  "Sales per m²: %{customdata[6]}<br>"
                  "CSm²I: %{customdata[7]}<br>"
                  f"{hover_title}: " + "%{customdata[8]}<extra></extra>"
)

rad_scatter.update_layout(
    margin=dict(l=20, r=20, t=10, b=10),
    height=560,
    legend_title_text="CSm²I t.o.v. target",
    xaxis=dict(title="Sales per Visitor (€/bezoeker)", tickformat=",.2f"),
    yaxis=dict(title="Sales per m² (€/m²)", tickformat=",.2f"),
)
st.plotly_chart(rad_scatter, use_container_width=True)
