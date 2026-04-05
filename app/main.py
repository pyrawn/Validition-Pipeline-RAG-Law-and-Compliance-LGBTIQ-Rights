"""
app/main.py
-----------
Streamlit dashboard: Mexico LGBTIQ+ Rights Compliance Atlas.

Data source: MongoDB Atlas (falls back to latest scores JSON in data/output/).
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yaml
from dotenv import load_dotenv

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Mexico LGBTIQ+ Compliance Atlas",
    page_icon="🏳️‍🌈",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE       = pathlib.Path(__file__).resolve().parent
CONFIG_PATH = _HERE.parent / "config" / "mongo.yaml"
OUTPUT_DIR  = _HERE.parent / "data" / "output"

MARKERS      = ["L1", "L2", "L3", "C1", "C2", "C3"]
MARKER_NAMES = {
    "L1": "Non-discrimination",
    "L2": "Gender identity recognition",
    "L3": "Same-sex union / marriage",
    "C1": "Hate-crime provisions",
    "C2": "Anti-discrimination enforcement",
    "C3": "Healthcare access",
}

GEOJSON_URL = (
    "https://raw.githubusercontent.com/angelnmara/geojson/master/mexicoHigh.json"
)

# Maps pipeline state names (ASCII, from legal_inventory.csv) to the exact
# "name" property values used in the mexicoHigh GeoJSON.
STATE_NAME_MAP = {
    "Aguascalientes":    "Aguascalientes",
    "Baja California":   "Baja California",
    "Baja California Sur": "Baja California Sur",
    "Campeche":          "Campeche",
    "Chiapas":           "Chiapas",
    "Chihuahua":         "Chihuahua",
    "Ciudad de Mexico":  "Ciudad de México",
    "Coahuila":          "Coahuila",
    "Colima":            "Colima",
    "Durango":           "Durango",
    "Estado de Mexico":  "México",
    "Guanajuato":        "Guanajuato",
    "Guerrero":          "Guerrero",
    "Hidalgo":           "Hidalgo",
    "Jalisco":           "Jalisco",
    "Michoacan":         "Michoacán",
    "Morelos":           "Morelos",
    "Nayarit":           "Nayarit",
    "Nuevo Leon":        "Nuevo León",
    "Oaxaca":            "Oaxaca",
    "Puebla":            "Puebla",
    "Queretaro":         "Querétaro",
    "Quintana Roo":      "Quintana Roo",
    "San Luis Potosi":   "San Luis Potosí",
    "Sinaloa":           "Sinaloa",
    "Sonora":            "Sonora",
    "Tabasco":           "Tabasco",
    "Tamaulipas":        "Tamaulipas",
    "Tlaxcala":          "Tlaxcala",
    "Veracruz":          "Veracruz",
    "Yucatan":           "Yucatán",
    "Zacatecas":         "Zacatecas",
}

# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def _load_from_mongo() -> List[Dict]:
    load_dotenv(_HERE.parent / ".env", override=False)
    if not CONFIG_PATH.exists():
        st.sidebar.warning(f"Config not found: {CONFIG_PATH}")
        return []
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["mongo"]
    try:
        from pymongo import MongoClient
        client     = MongoClient(cfg["host"], serverSelectionTimeoutMS=5_000)
        collection = client[cfg["database"]][cfg["collection"]]
        records    = list(collection.find({}, {"_id": 0}))
        client.close()
        return records
    except Exception as e:
        st.sidebar.error(f"MongoDB connection failed: {e}")
        return []


@st.cache_data(ttl=300)
def _load_from_json() -> List[Dict]:
    jsons = sorted(OUTPUT_DIR.glob("scores_*.json"), reverse=True)
    if not jsons:
        return []
    with jsons[0].open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    records = _load_from_mongo()
    source  = "MongoDB Atlas"
    if not records:
        records = _load_from_json()
        source  = "local JSON"
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    st.sidebar.caption(f"Data source: {source}  |  {len(df)} records")
    return df


@st.cache_data(ttl=3600)
def load_geojson() -> Any:
    try:
        resp = requests.get(GEOJSON_URL, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def pivot_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return wide-format DataFrame: one row per state, one column per marker."""
    return (
        df.pivot_table(index="state", columns="marker", values="score", aggfunc="first")
          .reindex(columns=MARKERS, fill_value=0)
          .reset_index()
    )


def wide_with_totals(wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.copy()
    wide["total"]   = wide[MARKERS].sum(axis=1)
    wide["legal"]   = wide[["L1", "L2", "L3"]].sum(axis=1)
    wide["criminal"] = wide[["C1", "C2", "C3"]].sum(axis=1)
    return wide


# ── Main app ──────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ────────────────────────────────────────────────────────────────
    st.title("Mexico LGBTIQ+ Rights Compliance Atlas")
    st.markdown(
        "Legal gap analysis across 32 Mexican states · "
        "Markers scored 0 (none) · 1 (partial) · 2 (full)"
    )
    st.divider()

    df = load_data()
    if df.empty:
        st.warning(
            "No data available. Run the Airflow pipeline or a fallback notebook "
            "first to generate scores."
        )
        return

    geojson = load_geojson()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    st.sidebar.header("Filters")
    selected_markers = st.sidebar.multiselect(
        "Markers", MARKERS, default=MARKERS,
        format_func=lambda m: f"{m} — {MARKER_NAMES[m]}",
    )
    min_score = st.sidebar.slider("Minimum score per marker", 0, 2, 0)

    if not selected_markers:
        st.warning("Select at least one marker in the sidebar.")
        return

    # Apply filters
    df_filtered = df[df["marker"].isin(selected_markers) & (df["score"] >= min_score)]

    wide_all      = wide_with_totals(pivot_scores(df))
    wide_filtered = wide_with_totals(pivot_scores(
        df[df["marker"].isin(selected_markers)]
    ))

    n_states   = wide_all["state"].nunique()
    nat_avg    = wide_all[MARKERS].mean()
    weakest    = nat_avg.idxmin()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    pct_threshold = (wide_all["total"] >= 6).sum() / n_states * 100
    pct_l2        = (wide_all["L2"] > 0).sum()  / n_states * 100
    pct_l3        = (wide_all["L3"] > 0).sum()  / n_states * 100

    kpi1.metric("States with score ≥ 6", f"{pct_threshold:.0f}%",
                help="Minimum protection threshold (total ≥ 6 / 12)")
    kpi2.metric("States with L2 > 0", f"{pct_l2:.0f}%",
                help="Any gender identity recognition")
    kpi3.metric("States with L3 > 0", f"{pct_l3:.0f}%",
                help="Any same-sex union recognition")
    kpi4.metric("Weakest marker nationally", weakest,
                help=f"{MARKER_NAMES[weakest]} · avg {nat_avg[weakest]:.2f}/2")

    st.divider()

    # ── National bar chart ────────────────────────────────────────────────────
    bar_data = pd.DataFrame({
        "marker": selected_markers,
        "total":  [wide_all[m].sum() for m in selected_markers],
        "max":    [n_states * 2] * len(selected_markers),
    })
    bar_data["label"] = bar_data.apply(
        lambda r: f"{int(r['total'])}/{int(r['max'])}", axis=1
    )
    bar_data["pct"] = bar_data["total"] / bar_data["max"]

    fig_bar = px.bar(
        bar_data,
        x="total",
        y="marker",
        orientation="h",
        text="label",
        color="pct",
        color_continuous_scale=[[0, "#d73027"], [0.5, "#fee08b"], [1, "#1a9850"]],
        range_color=[0, 1],
        labels={"total": "Sum of scores", "marker": ""},
        title=f"National compliance by marker (out of {n_states * 2} possible points)",
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        coloraxis_showscale=False,
        yaxis={"categoryorder": "array", "categoryarray": list(reversed(selected_markers))},
        margin=dict(l=60, r=40, t=50, b=40),
        height=320,
    )

    # ── Choropleth map ────────────────────────────────────────────────────────
    hover_cols = {m: True for m in MARKERS}

    wide_all["state_geo"] = wide_all["state"].map(STATE_NAME_MAP)

    if geojson:
        fig_map = px.choropleth(
            wide_all,
            geojson=geojson,
            locations="state_geo",
            featureidkey="properties.name",
            color="total",
            color_continuous_scale=[[0, "#d73027"], [0.5, "#fee08b"], [1, "#1a9850"]],
            range_color=[0, 12],
            hover_name="state",
            hover_data={**{"total": True}, **hover_cols},
            labels={"total": "Total score"},
            title="Total compliance score by state (0–12)",
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=50, b=0),
            coloraxis_colorbar=dict(title="Score", tickvals=[0, 3, 6, 9, 12]),
            height=500,
        )

    col_bar, col_map = st.columns([1, 2])
    with col_bar:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_map:
        if geojson:
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("Could not load GeoJSON for the choropleth map (network unavailable).")
            st.dataframe(wide_all[["state", "total"] + MARKERS].sort_values("total", ascending=False))

    st.divider()

    # ── State detail panel ────────────────────────────────────────────────────
    st.subheader("State detail")
    state_options = sorted(wide_all["state"].tolist())
    selected_state = st.selectbox("Select a state", state_options)

    row = wide_all[wide_all["state"] == selected_state].iloc[0]

    detail_k1, detail_k2, detail_k3, detail_k4 = st.columns(4)
    nat_avg_total = wide_all["total"].mean()
    detail_k1.metric("Total score", f"{int(row['total'])}/12",
                     delta=f"{row['total'] - nat_avg_total:+.1f} vs national avg")
    detail_k2.metric("Legal score (L1+L2+L3)", f"{int(row['legal'])}/6")
    detail_k3.metric("Criminal score (C1+C2+C3)", f"{int(row['criminal'])}/6")
    gaps = sum(1 for m in MARKERS if row[m] == 0)
    full = sum(1 for m in MARKERS if row[m] == 2)
    detail_k4.metric("Critical gaps / Full compliance", f"{gaps} / {full}")

    col_radar, col_scatter = st.columns(2)

    # Radar chart
    with col_radar:
        radar_vals = [row[m] for m in MARKERS] + [row[MARKERS[0]]]
        radar_labels = [f"{m}\n{MARKER_NAMES[m]}" for m in MARKERS] + [f"{MARKERS[0]}\n{MARKER_NAMES[MARKERS[0]]}"]
        fig_radar = go.Figure(go.Scatterpolar(
            r=radar_vals,
            theta=radar_labels,
            fill="toself",
            name=selected_state,
            line_color="#2563eb",
            fillcolor="rgba(37,99,235,0.2)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 2])),
            showlegend=False,
            title=f"{selected_state} — compliance radar",
            margin=dict(l=60, r=60, t=60, b=60),
            height=380,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Scatter: Legal vs Criminal
    with col_scatter:
        scatter_df = wide_all.copy()
        scatter_df["selected"] = scatter_df["state"] == selected_state
        fig_scatter = px.scatter(
            scatter_df,
            x="legal",
            y="criminal",
            color="selected",
            color_discrete_map={True: "#dc2626", False: "#94a3b8"},
            hover_name="state",
            hover_data={"legal": True, "criminal": True, "selected": False},
            labels={"legal": "Legal score (L1+L2+L3)", "criminal": "Criminal score (C1+C2+C3)"},
            title="Legal vs Criminal compliance across all states",
            range_x=[-0.2, 6.2],
            range_y=[-0.2, 6.2],
        )
        fig_scatter.add_shape(type="line", x0=0, y0=0, x1=6, y1=6,
                              line=dict(dash="dot", color="#cbd5e1"))
        fig_scatter.update_layout(showlegend=False, height=380,
                                  margin=dict(l=40, r=20, t=50, b=40))
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Detail table
    st.markdown(f"**Marker breakdown — {selected_state}**")
    state_rows = df[df["state"] == selected_state].set_index("marker")
    table_data = []
    for m in MARKERS:
        if m in state_rows.index:
            r = state_rows.loc[m]
            table_data.append({
                "Marker": f"{m} — {MARKER_NAMES[m]}",
                "Score":  int(r["score"]),
                "Cited article": r.get("cited_article", "N/A"),
                "Justification": r.get("justification", ""),
            })

    table_df = pd.DataFrame(table_data)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%d / 2"),
            "Justification": st.column_config.TextColumn(width="large"),
        },
    )


if __name__ == "__main__":
    main()
