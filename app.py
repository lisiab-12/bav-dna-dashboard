import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Page setup ----------------
st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")
st.title("DNA from BAV — Converter")
st.caption("Upload WPP BAV raw export → get Distinction, Nuance, Adventure, YoY deltas, and auto-insights.")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")

with st.sidebar.expander("What do the thresholds mean?"):
    st.write(
        "- **High threshold (≥)**: score at or above this is tagged **High**.\n"
        "- **Mid threshold (≥, < High)**: score at or above this is **Mid**; below it is **Low**.\n"
        "These labels feed the Pattern/Insight/Action columns."
    )

high_threshold = st.sidebar.number_input("High threshold (≥)", min_value=0, max_value=100, value=65, step=1)
mid_threshold = st.sidebar.number_input("Mid threshold (≥, < High)", min_value=0, max_value=100, value=50, step=1)

k_knowledge = st.sidebar.slider("Adventure: Knowledge weight (k)", 0.0, 1.0, 0.40, 0.01)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = {k_knowledge:.2f}×Knowledge + {k_innovation:.2f}×Innovation")

st.sidebar.markdown("---")
st.sidebar.write("**Download:** use the 'Export CSV' button at the bottom.")

# ---------------- Helpers ----------------
def to_num(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return np.nan

def compute_levels(x, mid, high):
    if pd.isna(x):
        return "Low"
    if x >= high:
        return "High"
    if x >= mid:
        return "Mid"
    return "Low"

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

def auto_insight(d_level, n_level, a_level):
    d_txt = {"High":"Distinctive","Mid":"Some distinctiveness","Low":"Blends in"}.get(d_level,"Blends in")
    n_txt = {"High":"Emotionally resonant","Mid":"Some resonance","Low":"Emotionally flat"}.get(n_level,"Emotionally flat")
    a_txt = {"High":"Progressive","Mid":"Some novelty","Low":"Conservative"}.get(a_level,"Conservative")
    return f"{d_txt} | {n_txt} | {a_txt}"

# ---------------- Upload ----------------
template_cols_rank = [
    "Brand","Category","Market","Year",
    "Differentiation","Relevance","Esteem","Knowledge","Innovation"
]

up = st.file_uploader("Upload BAV CSV", type=["csv"])

if up is None:
    st.info(
        "Upload a CSV to begin. Required rank fields in raw pull: "
        "`brand, category, market, year, differentiation_rank, relevance_rank, esteem_rank, knowledge_rank, mib_innovation_rank`."
    )
    st.stop()

# ---------------- Load & normalize headers ----------------
raw = pd.read_csv(up)

# Lowercase, strip
raw.columns = raw.columns.str.strip().str.lower()

# Map many common names -> canonical
# (we only use *_rank columns for the five pillar inputs)
colmap = {
    # id
    "brand": "brand",
    "brand name": "brand",
    "category": "category",
    "sector": "category",
    "market": "market",
    "country": "market",
    "geography": "market",
    # year variants
    "year": "year",
    "study_year": "year",
    "study year": "year",
    "fieldwork year": "year",
    "wave": "year",
    # ranks
    "differentiation_rank": "differentiation_rank",
    "relevance_rank": "relevance_rank",
    "esteem_rank": "esteem_rank",
    "knowledge_rank": "knowledge_rank",
    "mib_innovation_rank": "mib_innovation_rank",
    # sometimes shorter
    "diff_rank": "differentiation_rank",
    "rel_rank": "relevance_rank",
    "innov_rank": "mib_innovation_rank",
}

# Build a working frame with only fields we need (tolerant to extra columns)
df = raw.rename(columns={c: colmap.get(c, c) for c in raw.columns}).copy()

required = [
    "brand","category","market","year",
    "differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"
]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {missing}")
    st.write("Detected headers:", list(df.columns))
    st.stop()

# Clean & coerce Year → int
def clean_year(x):
    s = str(x).strip()
    s = s.replace(",", "")
    try:
        val = int(float(s))
        # Ignore implausible tiny years
        return val if 1900 <= val <= 2100 else np.nan
    except:
        return np.nan

df["year"] = df["year"].apply(clean_year)

# Coerce ranks to numeric
for c in ["differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]:
    df[c] = df[c].apply(to_num)

# ---- Aggregate duplicates to Brand×Category×Market×Year (mean) ----
agg = (
    df.groupby(["brand","category","market","year"], dropna=False)[
        ["differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]
    ].mean()
    .reset_index()
)

# Drop rows without Year after cleaning
agg = agg.dropna(subset=["year"]).copy()
agg["year"] = agg["year"].astype(int)

# Canonical working frame
work = pd.DataFrame({
    "Brand":           agg["brand"],
    "Category":        agg["category"],
    "Market":          agg["market"],
    "Year":            agg["year"],
    "Differentiation": agg["differentiation_rank"],
    "Relevance":       agg["relevance_rank"],
    "Esteem":          agg["esteem_rank"],
    "Knowledge":       agg["knowledge_rank"],
    "Innovation":      agg["mib_innovation_rank"],
})

# ---------------- DNA metrics ----------------
out = work.copy()
out["D"]   = out["Differentiation"]
out["N"]   = (out["Relevance"] + out["Esteem"]) / 2
out["A"]   = out["Knowledge"] * k_knowledge + out["Innovation"] * (1 - k_knowledge)
out["DNA"] = out[["D","N","A"]].mean(axis=1)

# Levels, pattern, insights
out["DLevel"] = out["D"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["NLevel"] = out["N"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["ALevel"] = out["A"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["Pattern"] = out["DLevel"] + " / " + out["NLevel"] + " / " + out["ALevel"]
out["AutoInsight"] = out.apply(lambda r: auto_insight(r["DLevel"], r["NLevel"], r["ALevel"]), axis=1)
out["AutoAction"] = out["Pattern"].apply(pattern_to_action)

# ---------------- User view filters ----------------
st.sidebar.subheader("View controls")

brands_sorted = sorted(out["Brand"].dropna().unique().tolist())
default_brand = brands_sorted[0] if brands_sorted else None
focus_brand = st.sidebar.selectbox("Focus brand", brands_sorted, index=brands_sorted.index(default_brand) if default_brand in brands_sorted else 0)

peer_candidates = sorted(out.loc[out["Category"]==out[out["Brand"]==focus_brand]["Category"].iloc[0], "Brand"].unique().tolist()) \
    if focus_brand and not out[out["Brand"]==focus_brand].empty else brands_sorted
peer_candidates = [b for b in peer_candidates if b != focus_brand]

peer_brands = st.sidebar.multiselect(
    "Peer brands for 'Category average' (choose any set)",
    options=peer_candidates,
    default=peer_candidates[:3] if len(peer_candidates) >= 3 else peer_candidates
)

metric_choices = ["D","N","A","DNA"]
metrics_to_show = st.sidebar.multiselect("Metrics to show", metric_choices, default=metric_choices)

# ---------------- Category averages (by chosen peers) ----------------
# Compute Category average using only the selected peer brands (per Year)
if peer_brands:
    peers_df = out[out["Brand"].isin(peer_brands)].copy()
else:
    # if no peers chosen, use whole category (minus focus brand) to avoid empty
    cat = out[out["Brand"]==focus_brand]["Category"].iloc[0] if focus_brand in out["Brand"].values else None
    peers_df = out[(out["Category"]==cat) & (out["Brand"]!=focus_brand)].copy()

cat_avg = (
    peers_df.groupby(["Category","Year"], dropna=False)[["D","N","A","DNA"]].mean().reset_index()
    .rename(columns={"D":"CategoryAvg_D","N":"CategoryAvg_N","A":"CategoryAvg_A","DNA":"CategoryAvg_DNA"})
)

view = out.merge(cat_avg, on=["Category","Year"], how="left")

# ---------------- vs Category (%) ----------------
for m in ["D","N","A","DNA"]:
    view[f"Δ vsCategory_{m} [%]"] = (view[m] - view[f"CategoryAvg_{m}"]) / view[f"CategoryAvg_{m}"].replace(0, np.nan) * 100

# ---------------- YoY % (within Brand×Category×Market) ----------------
view = view.sort_values(["Brand","Category","Market","Year"]).reset_index(drop=True)
for m in ["D","N","A","DNA"]:
    col = f"Δ YoY {m} [%]"
    view[col] = np.nan

for keys, grp in view.groupby(["Brand","Category","Market"], dropna=False):
    idx = grp.index.tolist()
    for prev_i, curr_i in zip(idx[:-1], idx[1:]):
        prev_val = view.loc[prev_i, ["D","N","A","DNA"]]
        curr_val = view.loc[curr_i, ["D","N","A","DNA"]]
        for m in ["D","N","A","DNA"]:
            p, c = prev_val[m], curr_val[m]
            if pd.notna(p) and p != 0 and pd.notna(c):
                view.loc[curr_i, f"Δ YoY {m} [%]"] = (c - p) / abs(p) * 100

# ---------------- Display table ----------------
st.markdown("### Results Table")
st.caption(
    "Legend: **Δ vs Category (%)** = (Brand − Category average) / Category average · 100. "
    "**Δ vs last year (%)** compares each brand’s score to its own prior year within the same Category & Market."
)

# Clean display-only “None” → blank for delta/avg cols
display = view.copy()
for c in display.columns:
    if c.startswith("CategoryAvg_") or c.startswith("Δ vsCategory_") or c.startswith("Δ YoY"):
        display[c] = display[c].round(4)

# Group labels row (simple header hint)
group_hint = pd.DataFrame([{
    "Brand":"— Identifiers —", "Category":"— Identifiers —", "Market":"— Identifiers —", "Year":"— Identifiers —",
    "Differentiation":"— Raw ranks —", "Relevance":"— Raw ranks —", "Esteem":"— Raw ranks —", "Knowledge":"— Raw ranks —", "Innovation":"— Raw ranks —",
    "D":"— DNA calc —", "N":"— DNA calc —", "A":"— DNA calc —", "DNA":"— DNA calc —",
    "DLevel":"— Interpretation —", "NLevel":"— Interpretation —", "ALevel":"— Interpretation —", "Pattern":"— Interpretation —",
    "AutoInsight":"— Interpretation —", "AutoAction":"— Interpretation —"
}])
# Align columns
group_hint = group_hint.reindex(columns=display.columns, fill_value="")

show_df = pd.concat([group_hint, display], ignore_index=True)

st.dataframe(show_df, use_container_width=True)

# ---------------- Charts ----------------
st.markdown("### Charts")

# 1) Bar — latest year for focus brand + category averages
focus_df = view[view["Brand"]==focus_brand].copy()
if not focus_df.empty:
    latest_year = int(focus_df["Year"].max())
    f_latest = focus_df[focus_df["Year"]==latest_year].copy()

    # Single row per Market possible; aggregate across Market for the bar
    f_bar = f_latest.groupby("Year")[["D","N","A","DNA"]].mean().reset_index()
    f_bar_melt = f_bar.melt(id_vars=["Year"], value_vars=["D","N","A","DNA"], var_name="Pillar", value_name="Score")

    cat_bar = cat_avg[cat_avg["Year"]==latest_year][["Year","CategoryAvg_D","CategoryAvg_N","CategoryAvg_A","CategoryAvg_DNA"]] \
                .mean(numeric_only=True).to_frame().T
    cat_bar["Year"] = latest_year
    cat_bar = cat_bar.melt(id_vars=["Year"], value_vars=["CategoryAvg_D","CategoryAvg_N","CategoryAvg_A","CategoryAvg_DNA"],
                           var_name="Pillar", value_name="CategoryAvg")
    cat_bar["Pillar"] = cat_bar["Pillar"].map({
        "CategoryAvg_D":"D","CategoryAvg_N":"N","CategoryAvg_A":"A","CategoryAvg_DNA":"DNA"
    })

    bar_merge = f_bar_melt.merge(cat_bar[["Pillar","CategoryAvg"]], on="Pillar", how="left")

    fig_bar = px.bar(bar_merge, x="Pillar", y="Score", title=f"{focus_brand} — latest year {latest_year} (YoY change shown)")
    # Add category average as markers
    fig_bar.add_trace(go.Scatter(x=bar_merge["Pillar"], y=bar_merge["CategoryAvg"],
                                 mode="markers", name="Category avg"))
    fig_bar.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_bar, use_container_width=True)

# 2) Trend — brand vs category average (years as integers only)
trend_brand = view[view["Brand"]==focus_brand].groupby("Year")[["D","N","A","DNA"]].mean().reset_index()
trend_cat   = cat_avg.groupby("Year")[["CategoryAvg_D","CategoryAvg_N","CategoryAvg_A","CategoryAvg_DNA"]].mean().reset_index()

if not trend_brand.empty:
    # Choose which metric to show in trend: default DNA
    metric_for_trend = "DNA"
    y1 = trend_brand[["Year",metric_for_trend]].rename(columns={metric_for_trend:f"{focus_brand} — {metric_for_trend}"})
    y2 = trend_cat[["Year",f"CategoryAvg_{metric_for_trend}"]].rename(
        columns={f"CategoryAvg_{metric_for_trend}":"Category avg — "+metric_for_trend}
    )
    tmerge = y1.merge(y2, on="Year", how="outer").sort_values("Year")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=tmerge["Year"], y=tmerge[f"{focus_brand} — {metric_for_trend}"],
                                   mode="lines+markers", name=f"{focus_brand} — {metric_for_trend}"))
    fig_trend.add_trace(go.Scatter(x=tmerge["Year"], y=tmerge["Category avg — "+metric_for_trend],
                                   mode="lines+markers", name="Category avg — "+metric_for_trend))
    fig_trend.update_layout(title=f"{focus_brand} — trend vs Category average", xaxis=dict(tickmode="linear"), yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------------- Export ----------------
csv = view.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV (all rows)", data=csv, file_name="dna_from_bav_output.csv")
