import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------- #
# App chrome
# ----------------------------- #
st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")
st.title("DNA from BAV — Converter")
st.caption(
    "Upload WPP BAV raw export (rank fields) → get Distinction, Nuance, Adventure, "
    "YoY deltas, and auto-insights. Uses only *_rank columns."
)

# ----------------------------- #
# Sidebar controls (with help)
# ----------------------------- #
st.sidebar.header("Controls")

high_threshold = st.sidebar.number_input(
    "High threshold (≥)",
    min_value=0, max_value=100, value=65, step=1,
    help="Scores at or above this value are classified as **High** for level labels."
)
mid_threshold = st.sidebar.number_input(
    "Mid threshold (≥, < High)",
    min_value=0, max_value=100, value=50, step=1,
    help="Scores at or above this value (but below High) are **Mid**; otherwise **Low**."
)

k_knowledge = st.sidebar.slider(
    "Adventure: Knowledge weight (k)",
    0.0, 1.0, 0.40, 0.01,
    help="Adventure = k×Knowledge_rank + (1−k)×MIB_Innovation_rank."
)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = **{k_knowledge:.2f}×Knowledge** + **{k_innovation:.2f}×Innovation**")

st.sidebar.markdown("---")
st.sidebar.write("**Download:** use the 'Export CSV' button at the bottom.")

st.sidebar.markdown("---")
with st.sidebar.expander("What do the thresholds mean?"):
    st.markdown(
        "- We classify D/N/A scores into **High / Mid / Low** bands for quick reading.\n"
        "- **High threshold**: values **≥** this are **High**.\n"
        "- **Mid threshold**: values **≥** this (but **< High**) are **Mid**; values below are **Low**.\n"
        "- These labels drive the **Pattern** and the auto-insight/action text."
    )

# ----------------------------- #
# File upload
# ----------------------------- #
st.subheader("Upload BAV CSV")
up = st.file_uploader("Drag and drop file here", type=["csv"])

required_rank_cols = [
    "brand", "category", "market", "year",
    "differentiation_rank", "relevance_rank", "esteem_rank",
    "knowledge_rank", "mib_innovation_rank"
]

if up is None:
    st.info(
        "Upload a CSV to begin. **Required raw pull fields** (lower/any case fine): "
        + ", ".join(required_rank_cols)
    )
    st.stop()

# ----------------------------- #
# Read + normalize headers
# ----------------------------- #
raw = pd.read_csv(up)
raw.columns = raw.columns.str.strip().str.lower()

missing = [c for c in required_rank_cols if c not in raw.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Detected headers:", list(raw.columns))
    st.stop()

# Keep only what we need
df = raw[required_rank_cols].copy()

# Clean types
def to_num(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return np.nan

df["year"] = df["year"].apply(lambda x: int(float(str(x).strip())) if str(x).strip() != "" else np.nan)
for col in [
    "differentiation_rank", "relevance_rank", "esteem_rank",
    "knowledge_rank", "mib_innovation_rank"
]:
    df[col] = df[col].apply(to_num)

# Canonical names (rank-only)
df = df.rename(columns={
    "brand": "Brand",
    "category": "Category",
    "market": "Market",
    "year": "Year",
    "differentiation_rank": "Differentiation",
    "relevance_rank": "Relevance",
    "esteem_rank": "Esteem",
    "knowledge_rank": "Knowledge",
    "mib_innovation_rank": "Innovation",
})

# ----------------------------- #
# DNA calculations (rank scale, 0–100)
# ----------------------------- #
view = df.copy()
view["D"] = view["Differentiation"]
view["N"] = (view["Relevance"] + view["Esteem"]) / 2
view["A"] = view["Knowledge"] * k_knowledge + view["Innovation"] * (1 - k_knowledge)
view["DNA"] = view[["D", "N", "A"]].mean(axis=1)

def band(x, mid, high):
    if pd.isna(x): return "Low"
    if x >= high: return "High"
    if x >= mid: return "Mid"
    return "Low"

view["DLevel"] = view["D"].apply(lambda x: band(x, mid_threshold, high_threshold))
view["NLevel"] = view["N"].apply(lambda x: band(x, mid_threshold, high_threshold))
view["ALevel"] = view["A"].apply(lambda x: band(x, mid_threshold, high_threshold))
view["Pattern"] = view["DLevel"] + " / " + view["NLevel"] + " / " + view["ALevel"]

def auto_insight(dl, nl, al):
    d_txt = {"High":"Distinctive","Mid":"Some distinctiveness","Low":"Blends in"}.get(dl,"Blends in")
    n_txt = {"High":"Emotionally resonant","Mid":"Some resonance","Low":"Emotionally flat"}.get(nl,"Emotionally flat")
    a_txt = {"High":"Progressive","Mid":"Some novelty","Low":"Conservative"}.get(al,"Conservative")
    return f"{d_txt} | {n_txt} | {a_txt}"

def pattern_to_action(p):
    mapping = {
        "High / High / High": "Maintain momentum; codify codes; scale fame-driving acts.",
        "High / Mid / High": "Add emotional breadth without blunting edge.",
        "High / Low / High": "Build everyday relevance via service & community.",
        "High / High / Mid": "Inject fresh provocations to avoid drift.",
        "Mid / High / High": "Crystallise distinct codes; clarify POV.",
        "High / Mid / Mid": "Distinct but stagnant; pilot bolder innovation.",
        "Mid / High / Mid": "Sharpen codes; anchor in a felt need.",
        "Mid / Mid / High": "Clarify proposition & proof; keep momentum.",
        "Low / Low / Low": "Back to basics: define the brand world.",
    }
    return mapping.get(p, "Set foundational codes; then build warmth & momentum.")

view["AutoInsight"] = view.apply(lambda r: auto_insight(r["DLevel"], r["NLevel"], r["ALevel"]), axis=1)
view["AutoAction"]  = view["Pattern"].apply(pattern_to_action)

# ----------------------------- #
# YoY deltas (percentage) per Brand+Category
# ----------------------------- #
view = view.sort_values(["Brand", "Category", "Year"])
for col in ["D", "N", "A", "DNA"]:
    view[f"Δ YoY {col} (%)"] = np.nan

for (b, c), grp in view.groupby(["Brand", "Category"], dropna=False):
    idx = grp.index.tolist()
    prev = None
    for i in idx:
        if prev is not None:
            curr = view.loc[i, ["D","N","A","DNA"]]
            prevv = view.loc[prev, ["D","N","A","DNA"]]
            pct = (curr - prevv) / prevv.replace(0, np.nan) * 100.0
            for k in ["D","N","A","DNA"]:
                view.loc[i, f"Δ YoY {k} (%)"] = pct[k]
        prev = i

# ----------------------------- #
# Sidebar “view” controls (after data exists)
# ----------------------------- #
st.sidebar.markdown("### View controls")

brands_all = sorted(view["Brand"].dropna().unique().tolist())
focus_brand = st.sidebar.selectbox("Focus brand", brands_all if brands_all else ["—"])

# peers to use for Category Average line & vsCategory columns
peer_brands = st.sidebar.multiselect(
    "Peer brands for 'Category average' (choose any set)",
    options=brands_all,
    default=[b for b in brands_all if b != focus_brand][:3],
    help="Used for the trend overlay & 'vs Category' columns. If empty, uses ALL brands in the same Category."
)

metrics_opts = ["D", "N", "A", "DNA"]
metrics_to_show = st.sidebar.multiselect(
    "Metrics to show",
    options=metrics_opts,
    default=metrics_opts,
)

# Year range
min_year = int(view["Year"].min()) if len(view) else 2020
max_year = int(view["Year"].max()) if len(view) else 2025
year_min, year_max = st.sidebar.slider(
    "Year range",
    min_value=min_year, max_value=max_year,
    value=(min_year, max_year),
)

# Filtered table for display
table_df = view[(view["Year"] >= year_min) & (view["Year"] <= year_max)].copy()

# ----------------------------- #
# Category averages (based on peer set if provided)
# ----------------------------- #
def calc_category_avg(frame: pd.DataFrame) -> pd.DataFrame:
    """Returns Category avg per (Category, Market, Year) using chosen peers if available."""
    # Build mask of peers within same Category for each row:
    if peer_brands:
        peers = frame["Brand"].isin(peer_brands)
        base = frame.loc[peers].copy()
        # If peer set empties a category/year, fallback to all
        if base.empty:
            base = frame.copy()
    else:
        base = frame.copy()

    cat = (
        base.groupby(["Category", "Market", "Year"], dropna=False)[["D", "N", "A", "DNA"]]
        .mean()
        .rename(columns={
            "D": "CategoryAvg_D", "N": "CategoryAvg_N",
            "A": "CategoryAvg_A", "DNA": "CategoryAvg_DNA"
        })
        .reset_index()
    )
    return cat

cat_avg = calc_category_avg(table_df)
table_df = table_df.merge(cat_avg, on=["Category", "Market", "Year"], how="left")

# vs Category deltas (percentage)
for col in ["D", "N", "A", "DNA"]:
    table_df[f"vsCategory_{col} (%)"] = (
        (table_df[col] - table_df[f"CategoryAvg_{col}"]) /
        table_df[f"CategoryAvg_{col}"].replace(0, np.nan)
    ) * 100.0

# ----------------------------- #
# Interactive table filters (Brands)
# ----------------------------- #
table_brands = st.sidebar.multiselect(
    "Filter table to brands",
    sorted(view["Brand"].dropna().unique().tolist()),
    default=sorted(view["Brand"].dropna().unique().tolist())
)
table_df = table_df[table_df["Brand"].isin(table_brands)].copy()

# ----------------------------- #
# Column presentation / grouping
# (We drop BrandAvg_* and vsBrand_*; keep CategoryAvg_* and vsCategory_*)
# ----------------------------- #
# Build a MultiIndex for nicer grouping
id_cols   = ["Brand", "Category", "Market", "Year"]
raw_cols  = ["Differentiation", "Relevance", "Esteem", "Knowledge", "Innovation"]
dna_cols  = ["D", "N", "A", "DNA"]
lvl_cols  = ["DLevel", "NLevel", "ALevel", "Pattern", "AutoInsight", "AutoAction"]
cat_cols  = ["CategoryAvg_D", "CategoryAvg_N", "CategoryAvg_A", "CategoryAvg_DNA"]
vsc_cols  = ["vsCategory_D (%)", "vsCategory_N (%)", "vsCategory_A (%)", "vsCategory_DNA (%)"]
yoy_cols  = ["Δ YoY D (%)", "Δ YoY N (%)", "Δ YoY A (%)", "Δ YoY DNA (%)"]

# Reformat Year as int (ensure no decimals in table)
table_df["Year"] = table_df["Year"].astype("Int64")

# Prepare a tidy output with desired order
out_cols = id_cols + raw_cols + dna_cols + lvl_cols + cat_cols + vsc_cols + yoy_cols

# Ensure all exist (some could be missing if earlier steps changed)
out_cols = [c for c in out_cols if c in table_df.columns]
tidy = table_df[out_cols].copy()

# Attach a hierarchical header
def make_cols(cols):
    groups = {}
    for c in cols:
        if c in id_cols:   groups[c] = ("Identifiers", c)
        elif c in raw_cols: groups[c] = ("Raw (ranks)", c)
        elif c in dna_cols: groups[c] = ("DNA (calc)", c)
        elif c in lvl_cols: groups[c] = ("Interpretation", c)
        elif c in cat_cols: groups[c] = ("Category average", c)
        elif c in vsc_cols: groups[c] = ("Δ vs Category (%)", c)
        elif c in yoy_cols: groups[c] = ("Δ vs last year (%)", c)
        else:               groups[c] = ("Other", c)
    tuples = [groups[c] for c in cols]
    return pd.MultiIndex.from_tuples(tuples)

tidy.columns = make_cols(tidy.columns)

# ----------------------------- #
# Results Table
# ----------------------------- #
st.subheader("Results Table")
st.caption(
    "_Legend_: **Δ vs Category (%)** = (Brand – Category average) / Category average · 100. "
    "**Δ vs last year (%)** compares each brand’s score to its own prior year."
)
st.dataframe(tidy, use_container_width=True)

# ----------------------------- #
# Charts
# ----------------------------- #
st.subheader("Charts")

# Latest year row for focus brand (within the chosen range)
fb = table_df[table_df["Brand"] == focus_brand].copy()
fb = fb.sort_values("Year")
if not fb.empty:
    latest = fb.dropna(subset=["Year"]).iloc[-1]
    latest_year = int(latest["Year"])
    show_metrics = [m for m in metrics_to_show if m in ["D","N","A","DNA"]]

    # Bar: D/N/A (or selected metrics) for latest year
    bar_df = pd.DataFrame({
        "Pillar": show_metrics,
        "Score": [float(latest[m]) for m in show_metrics]
    })

    # Annotate with YoY% on bars when available
    yoy_labels = []
    for m in show_metrics:
        col = f"Δ YoY {m} (%)"
        val = fb.iloc[-1][col] if col in fb.columns else np.nan
        yoy_labels.append(val)

    fig_bar = px.bar(bar_df, x="Pillar", y="Score", title=f"{focus_brand} — latest year {latest_year} (YoY change shown)")
    fig_bar.update_layout(yaxis=dict(range=[0, 100]))

    # Add annotations for YoY%
    for i, (pillar, score, yoy) in enumerate(zip(bar_df["Pillar"], bar_df["Score"], yoy_labels)):
        if pd.notna(yoy):
            fig_bar.add_annotation(
                x=pillar, y=score, text=f"{yoy:+.1f}%", showarrow=False, yshift=12
            )

    st.plotly_chart(fig_bar, use_container_width=True)

    # Trend: brand vs category average line
    trend = fb.sort_values("Year")
    trend_x = trend["Year"].astype(int)

    fig_line = go.Figure()
    # Brand line
    for m in show_metrics:
        fig_line.add_trace(go.Scatter(
            x=trend_x, y=trend[m].astype(float),
            mode="lines+markers", name=f"{focus_brand} — {m}"
        ))
        # Category average overlay for same rows
        catcol = f"CategoryAvg_{m}"
        if catcol in trend.columns:
            fig_line.add_trace(go.Scatter(
                x=trend_x, y=trend[catcol].astype(float),
                mode="lines+markers", name=f"Category avg — {m}", line=dict(dash="dash")
            ))

    fig_line.update_layout(
        title=f"{focus_brand} — trend vs Category average",
        yaxis=dict(range=[0, 100]),
        xaxis=dict(tickmode="array", tickvals=sorted(trend_x.unique().tolist()))
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("Pick a **Focus brand** that exists within the selected year range to see charts.")

# ----------------------------- #
# Export
# ----------------------------- #
csv_bytes = table_df.sort_values(["Brand","Category","Year"]).to_csv(index=False).encode("utf-8")
st.download_button("Export CSV (all rows)", data=csv_bytes, file_name="dna_from_bav_output.csv")
