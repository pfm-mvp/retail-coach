# ===== Uur‑drilldown (alleen wanneer 'Uur' is gekozen) =====
if step == "hour":
    st.markdown("## Uur‑profielen (drill‑down & heatmap)")

    # 👉 openingstijden-filter (alleen relevant voor urenweergave)
    open_start, open_end = st.slider(
        "Heatmap binnen openingstijd (uurbereik)",
        min_value=0, max_value=23, value=(9, 21), step=1
    )
    st.caption(f"Heatmap binnen openingstijd {open_start:02d}:00–{open_end:02d}:00 (gemeten in de gekozen periode).")

    # dag-namen voor y‑as heatmap
    dow_names = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]

    # per winkel: lijnprofiel + heatmap
    for _, r in agg.iterrows():
        sid = int(r["shop_id"])
        name = r["shop_name"]

        sub = df[df["shop_id"] == sid].copy()
        if sub.empty:
            continue

        # Zorg dat tijdkolommen beschikbaar zijn
        sub["ts"] = pd.to_datetime(sub["timestamp"], errors="coerce")
        sub["hour"] = sub["ts"].dt.hour
        sub["dow"]  = sub["ts"].dt.dayofweek  # 0=Ma … 6=Zo

        # Filter op openingstijden
        sub = sub[(sub["hour"] >= open_start) & (sub["hour"] <= open_end)]
        if sub.empty:
            with st.expander(f"⏱️ {name} — uurprofiel", expanded=False):
                st.info("Geen meetpunten binnen het gekozen uur‑bereik.")
            continue

        sub = normalize_kpis(sub)

        # ---- Lijnprofiel (gemiddelde per uur over alle dagen) ----
        line_df = sub.groupby("hour").agg(
            visitors=("count_in", "mean"),
            spv=("sales_per_visitor", "mean"),
            conv=("conversion_rate", "mean"),
        ).reset_index().sort_values("hour")

        with st.expander(f"⏱️ {name} — uurprofiel", expanded=False):
            fig_line = go.Figure()
            fig_line.add_trace(
                go.Bar(
                    x=line_df["hour"], y=line_df["visitors"],
                    name="Bezoekers", yaxis="y2", opacity=0.30,
                    marker_color="#94a3b8"
                )
            )
            fig_line.add_trace(
                go.Scatter(
                    x=line_df["hour"], y=line_df["spv"],
                    name="SPV (€)", mode="lines+markers"
                )
            )
            fig_line.add_trace(
                go.Scatter(
                    x=line_df["hour"], y=line_df["conv"]*100.0,
                    name="Conversie (%)", mode="lines+markers"
                )
            )
            fig_line.update_layout(
                height=360, margin=dict(l=20, r=20, t=10, b=10),
                xaxis=dict(title="Uur"),
                yaxis=dict(title="SPV (€) / Conversie (%)", rangemode="tozero"),
                yaxis2=dict(
                    title="Bezoekers", overlaying="y", side="right", rangemode="tozero"
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            # ✅ unieke key per winkel
            st.plotly_chart(fig_line, use_container_width=True, key=f"line_{sid}")

        # ---- Heatmap (gemiddeld bezoekers/uur per dag van de week) ----
        hm_df = sub.groupby(["dow", "hour"]).agg(
            visitors=("count_in", "mean"),
            spv=("sales_per_visitor", "mean"),
            conv=("conversion_rate", "mean"),
        ).reset_index()

        # pivot naar dag x uur raster, alleen gekozen openingstijden
        hours = list(range(open_start, open_end + 1))
        pivot = (
            hm_df.pivot(index="dow", columns="hour", values="visitors")
            .reindex(index=range(7))  # zorg dat alle dagen aanwezig zijn
            .reindex(columns=hours)
            .fillna(0.0)
        )

        # Schaal robuust kiezen (95e percentiel) → betere kleurverdeling
        z = pivot.values
        zmax = float(np.nanpercentile(z, 95)) if np.isfinite(z).all() else float(np.nanmax(z))
        zmax = max(zmax, 1.0)  # vermijd vlakke schaal

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=pivot.values,
                x=pivot.columns,               # uren
                y=[dow_names[i] for i in pivot.index],  # Ma..Zo
                colorscale="YlOrRd",
                zmin=0, zmax=zmax,
                colorbar=dict(title="Bezoekers/uur"),
                hovertemplate=(
                    "Dag: %{y}<br>"
                    "Uur: %{x}:00<br>"
                    "Bezoekers/uur: %{z:.1f}<extra></extra>"
                ),
            )
        )
        fig_hm.update_layout(
            title=f"{name} — Heatmap bezoekers",
            height=300, margin=dict(l=20, r=20, t=35, b=10),
            xaxis=dict(title="Uur"),
            yaxis=dict(title="", autorange="reversed")  # Ma bovenaan
        )
        # ✅ unieke key per winkel
        st.plotly_chart(fig_hm, use_container_width=True, key=f"hm_{sid}")
