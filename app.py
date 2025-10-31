import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")
st.title("DNA from BAV — Converter")
st.caption("Upload WPP BAV raw export → get Distinction, Nuance, Adventure, YoY deltas, and auto-insights.")
st.caption("**Tip:** Thresholds only affect level labels (High/Mid/Low) – not raw scores or averages.")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")
high_threshold = st.sidebar.number_input("High threshold (≥)", min_value=0, max_value=100, value=65, step=1)
mid_threshold  = st.sidebar.number_input("Mid threshold (≥, < High)", min_value=0, max_value=100, value=50, step=1)

k_knowledge = st.sidebar.slider("Adventure: Knowledge weight (k)", 0.0, 1.0, 0.40, 0.01)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = {k_knowledge:.2f}×Knowledge + {k_innovation:.2f}×Innovation")

st.sidebar.markdown("---")
st.sidebar.write("**Download:** use the 'Export CSV' button at the bottom.")

# ---------------------------
# File upload
# ---------------------------
up = st.file_uploader("Upload BAV CSV", type=["csv"])
if up is None:
    st.info("Upload a CSV to begin. Required raw columns (lowercase OK): "
            "brand, category, market, year, differentiation_rank, relevance_rank, esteem_rank, knowledge_rank, mib_innovation_rank")
    st.stop()

# ---------------------------
# Read + normalize columns
# ---------------------------
df_raw = pd.read_csv(up)
df_raw.columns = df_raw.columns.str.strip().str.lower()

def need(name: str):
    if name not in df_raw.columns:
        st.error(f"Missing required column: `{name}`")
        st.write("Detected headers:", list(df_raw.columns))
        st.stop()

for c in ["brand","category","market","year",
          "differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]:
    need(c)

# Canonical working frame (using *rank* columns only)
df = pd.DataFrame({
    "Brand":           df_raw["brand"],
    "Category":        df_raw["category"],
    "Market":          df_raw["market"],
    "Year":            df_raw["year"],
    "Differentiation": df_raw["differentiation_rank"],
    "Relevance":       df_raw["relevance_rank"],
    "Esteem":          df_raw["esteem_rank"],
    "Knowledge":       df_raw["knowledge_rank"],
    "Innovation":      df_raw["mib_innovation_rank"],
})

# Helpers
def to_num(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return np.nan

def compute_levels(x, mid, high):
    if pd.isna(x): return "Low"
    if x >= high:  return "High"
    if x >= mid:   return "Mid"
    return "Low"

def auto_insight(d_level, n_level, a_level):
    d_txt = {"High":"Distinctive","Mid":"Some distinctiveness","Low":"Blends in"}.get(d_level,"Blends in")
    n_txt = {"High":"Emotionally resonant","Mid":"Some resonance","Low":"Emotionally flat"}.get(n_level,"Emotionally flat")
    a_txt = {"High":"Progressive","Mid":"Some novelty","Low":"Conservative"}.get(a_level,"Conservative")
    return f"{d_txt} | {n_txt} | {a_txt}"

def pattern_to_action(pattern):
    mapping = {
        "High / High / High": "Maintain momentum; codify codes; scale fame-driving acts.",
        "High / Mid / High": "Dial up emotional resonance and mainstream cues without blunting edge.",
        "High / Low / High": "Build warmth and everyday relevance through service and community moments.",
        "High / High / Mid": "Inject fresh provocations to avoid drift into safe heritage-only.",
        "Mid / High / High": "Crystallise Distinction (codes) to cut through; clarify POV.",
        "High / Mid / Mid": "Distinct but stagnant; pilot bolder innovation and experiences.",
        "Mid / High / Mid": "Sharpen codes and simplify message; anchor in a felt need.",
        "Mid / Mid / High": "Anchor proposition and proof; reduce vagueness while keeping momentum.",
        "Low / Low / Low": "Back to basics: define who you are; craft minimum viable brand world.",
    }
    return mapping.get(pattern, "Set foundational codes; then build warmth and momentum.")

# Clean numerics
for col in ["Differentiation","Relevance","Esteem","Knowledge","Innovation"]:
    df[col] = df[col].apply(to_num)

# Clean integer year for all downstream views
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

# DNA pillars
df["D"]   = df["Differentiation"]
df["N"]   = (df["Relevance"] + df["Esteem"]) / 2
df["A"]   = df["Knowledge"] * k_knowledge + df["Innovation"] * (1 - k_knowledge)
df["DNA"] = df[["D","N","A"]].mean(axis=1)

# Levels & pattern
df["DLevel"] = df["D"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
df["NLevel"] = df["N"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
df["ALevel"] = df["A"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
df["Pattern"] = df["DLevel"] + " / " + df["NLevel"] + " / " + df["ALevel"]
df["AutoInsight"] = df.apply(lambda r: auto_insight(r["DLevel"], r["NLevel"], r["ALevel"]), axis=1)
df["AutoAction"]  = df["Pattern"].apply(pattern_to_action)

# Category averages by Year
cat_avg = (df.groupby(["Category","Year"])[["D","N","A","DNA"]]
             .mean()
             .rename(columns=lambda c: f"CategoryAvg_{c}")
             .reset_index())
view = df.merge(cat_avg, on=["Category","Year"], how="left")

# Deltas vs category average (points)
for m in ["D","N","A","DNA"]:
    view[f"vsCategory_{m}"] = view[m] - view[f"CategoryAvg_{m}"]

# YoY % change by Brand+Category
view = view.sort_values(["Brand","Category","Year"])
for (b,c), grp in view.groupby(["Brand","Category"], dropna=False):
    idx = grp.index.tolist()
    for p, q in zip(idx, idx[1:]):
        for m in ["D","N","A","DNA"]:
            prev = view.loc[p, m]
            curr = view.loc[q, m]
            if pd.notna(prev) and prev != 0 and pd.notna(curr):
                pct = (curr - prev) / abs(prev) * 100.0
            else:
                pct = np.nan
            view.loc[q, f"YoY%_{m}"] = pct

# ---------------------------
# View controls (left)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("View controls")

brands_all = sorted(view["Brand"].dropna().unique().tolist())
focus_brand = st.sidebar.selectbox("Focus brand", brands_all, index=0 if brands_all else None)

# Optional: choose a custom "peer set" for Peer Avg (for plots)
peer_brands = st.sidebar.multiselect(
    "Peer brands for 'Category average' (choose any set)",
    options=brands_all, default=[]
)

# Metrics chips
metric_options = ["D","N","A","DNA"]
metrics_to_show = st.sidebar.multiselect(
    "Metrics to show", options=metric_options, default=["D","N","A"], label_visibility="visible"
)
if len(metrics_to_show) == 0:
    metrics_to_show = ["D","N","A"]

# Year range
yr_min = int(pd.to_numeric(view["Year"], errors="coerce").dropna().min()) if view["Year"].notna().any() else 2020
yr_max = int(pd.to_numeric(view["Year"], errors="coerce").dropna().max()) if view["Year"].notna().any() else 2025
yr_from, yr_to = st.sidebar.slider("Year range", min_value=yr_min, max_value=yr_max, value=(yr_min, yr_max))

# Apply year filter for plots/table
in_year = view["Year"].between(yr_from, yr_to)
focus_df = view[(view["Brand"] == focus_brand) & in_year].copy()

# ---------------------------
# Results Table (interactive)
# ---------------------------
# Brand filter for the table
table_brands = st.sidebar.multiselect(
    "Filter table to brands", options=brands_all, default=brands_all
)
table_df = view[in_year & view["Brand"].isin(table_brands)].copy()

# Ensure Year appears as 2023 (not 2023.0)
table_df["Year"] = pd.to_numeric(table_df["Year"], errors="coerce").astype("Int64")

# Drop BrandAvg_* and vsBrand_* if present; keep Category averages
cols_to_drop = [c for c in table_df.columns if c.startswith("BrandAvg_") or c.startswith("vsBrand_")]
table_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# Build grouped headers (multi-index)
display_cols = [
    # Identifiers
    ("Identifiers","Brand"),
    ("Identifiers","Category"),
    ("Identifiers","Market"),
    ("Identifiers","Year"),

    # Raw pull (ranks)
    ("Raw pull (ranks)","Differentiation"),
    ("Raw pull (ranks)","Relevance"),
    ("Raw pull (ranks)","Esteem"),
    ("Raw pull (ranks)","Knowledge"),
    ("Raw pull (ranks)","Innovation"),

    # DNA calc
    ("DNA calc","D"),
    ("DNA calc","N"),
    ("DNA calc","A"),
    ("DNA calc","DNA"),

    # Levels & pattern
    ("Levels & pattern","DLevel"),
    ("Levels & pattern","NLevel"),
    ("Levels & pattern","ALevel"),
    ("Levels & pattern","Pattern"),
    ("Levels & pattern","AutoInsight"),
    ("Levels & pattern","AutoAction"),

    # Category averages
    ("Category avg","CategoryAvg_D"),
    ("Category avg","CategoryAvg_N"),
    ("Category avg","CategoryAvg_A"),
    ("Category avg","CategoryAvg_DNA"),

    # Δ vs category (points)
    ("Δ vs category (pts)","vsCategory_D"),
    ("Δ vs category (pts)","vsCategory_N"),
    ("Δ vs category (pts)","vsCategory_A"),
    ("Δ vs category (pts)","vsCategory_DNA"),

    # YoY change %
    ("YoY change %","YoY%_D"),
    ("YoY change %","YoY%_N"),
    ("YoY change %","YoY%_A"),
    ("YoY change %","YoY%_DNA"),
]
display_cols = [(g,c) for (g,c) in display_cols if c in table_df.columns]
multi_cols = pd.MultiIndex.from_tuples(display_cols, names=["",""])
table_to_show = table_df[[c for (_,c) in display_cols]].copy()
table_to_show.columns = multi_cols

st.markdown("### Results Table")
st.dataframe(
    table_to_show,
    use_container_width=True,
    column_config={
        ("Identifiers","Year"):     st.column_config.NumberColumn(format="%d"),
        ("YoY change %","YoY%_D"):   st.column_config.NumberColumn(format="%.1f%%"),
        ("YoY change %","YoY%_N"):   st.column_config.NumberColumn(format="%.1f%%"),
        ("YoY change %","YoY%_A"):   st.column_config.NumberColumn(format="%.1f%%"),
        ("YoY change %","YoY%_DNA"): st.column_config.NumberColumn(format="%.1f%%"),
    }
)

# ---------------------------
# Charts
# ---------------------------
st.markdown("### Charts")

def integer_year_axis(fig):
    # Make x-axis show 2023, 2024... no half ticks
    fig.update_xaxes(dtick=1, tickformat="d")
    fig.update_layout(xaxis=dict(type='linear'))

# -- Latest-year bar chart for focus brand
latest = focus_df.dropna(subset=["Year"]).sort_values("Year").tail(1)
if latest.empty:
    st.info("No data for the selected brand/year range.")
else:
    r0 = latest.iloc[0]
    latest_year = int(r0["Year"])
    vals = [float(r0[m]) for m in metrics_to_show]
    yoy_labels = []
    for m in metrics_to_show:
        pct = r0.get(f"YoY%_{m}", np.nan)
        yoy_labels.append("" if pd.isna(pct) else f"{pct:.1f}%")

    fig_bar = px.bar(
        x=metrics_to_show, y=vals,
        labels={"x":"Metric","y":"Score"},
        title=f"{focus_brand} — latest year {latest_year} (labels show YoY % change)",
        text=yoy_labels
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_bar, use_container_width=True)

# -- Trend line(s) for selected metrics + Category Average overlay
# Build long df for trend
focus_trend = focus_df.sort_values("Year")
if not focus_trend.empty:
    long_focus = focus_trend.melt(id_vars=["Year","Brand"], value_vars=metrics_to_show,
                                  var_name="Metric", value_name="Score")
    long_focus["Brand"] = focus_brand

    # Optional: Category Average overlay for the focus category
    current_cat = str(focus_trend["Category"].dropna().iloc[0]) if not focus_trend["Category"].dropna().empty else None
    lines = [long_focus]

    if current_cat:
        cat_slice = view[(view["Category"] == current_cat) & in_year]
        cat_avg_year = (cat_slice.groupby("Year")[["D","N","A","DNA"]]
                        .mean()
                        .reset_index())
        cat_long = cat_avg_year.melt(id_vars=["Year"], value_vars=metrics_to_show,
                                     var_name="Metric", value_name="Score")
        cat_long["Brand"] = f"{current_cat} Avg"
        lines.append(cat_long)

    combo = pd.concat(lines, ignore_index=True)

    fig_line = px.line(
        combo, x="Year", y="Score", color="Brand", line_dash="Metric",
        markers=True, title=f"Trend ({', '.join(metrics_to_show)}) — {focus_brand} vs Category Avg"
    )
    fig_line.update_layout(yaxis=dict(range=[0,100]))
    integer_year_axis(fig_line)
    st.plotly_chart(fig_line, use_container_width=True)

# ---------------------------
# Export
# ---------------------------
csv = view.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV (all rows)", data=csv, file_name="dna_from_bav_output.csv")
