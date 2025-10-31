import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")
st.title("DNA from BAV — Converter")
st.caption("Upload WPP BAV raw export → Distinction, Nuance, Adventure, YoY deltas, and auto-insights.")

# ---------------- Sidebar controls ----------------
st.sidebar.header("Controls")
high_threshold = st.sidebar.number_input("High threshold (≥)", 0, 100, 65)
mid_threshold = st.sidebar.number_input("Mid threshold (≥, < High)", 0, 100, 50)
k_knowledge = st.sidebar.slider("Adventure: Knowledge weight (k)", 0.0, 1.0, 0.40, 0.01)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = {k_knowledge:.2f}×Knowledge + {k_innovation:.2f}×Innovation")

# ---------------- Helper functions ----------------
def to_num(x):
    try:
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return float(x)
    except Exception:
        return np.nan

def clean_year(x):
    try:
        s = str(x).strip().replace(",", "")
        val = int(float(s))
        return val if 1900 <= val <= 2100 else np.nan
    except Exception:
        return np.nan

def compute_levels(x, mid, high):
    if pd.isna(x): return "Low"
    if x >= high: return "High"
    if x >= mid: return "Mid"
    return "Low"

def pattern_to_action(p):
    d = {
        "High / High / High": "Maintain momentum; codify codes; scale fame-driving acts.",
        "High / Mid / High": "Dial up emotional resonance without blunting edge.",
        "High / Low / High": "Build warmth and everyday relevance.",
        "High / High / Mid": "Inject freshness to avoid drift into heritage.",
        "Mid / High / High": "Clarify POV; crystallise Distinction.",
        "High / Mid / Mid": "Distinct but stagnant; pilot bolder innovation.",
        "Low / Low / Low": "Back to basics; define who you are.",
    }
    return d.get(p, "Set foundational codes; then build warmth and momentum.")

def auto_insight(d, n, a):
    d_txt = {"High":"Distinctive","Mid":"Some distinctiveness","Low":"Blends in"}.get(d,"Blends in")
    n_txt = {"High":"Emotionally resonant","Mid":"Some resonance","Low":"Flat"}.get(n,"Flat")
    a_txt = {"High":"Progressive","Mid":"Some novelty","Low":"Conservative"}.get(a,"Conservative")
    return f"{d_txt} | {n_txt} | {a_txt}"

# ---------------- Upload ----------------
up = st.file_uploader("Upload BAV CSV", type=["csv"])
if not up:
    st.info("Upload a CSV containing rank fields: differentiation_rank, relevance_rank, esteem_rank, knowledge_rank, mib_innovation_rank.")
    st.stop()

# ---------------- Load & clean headers ----------------
raw = pd.read_csv(up)

# Flatten multi-index headers (in case the export had two header rows)
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = ['_'.join([str(lvl) for lvl in tup if str(lvl) != 'nan']).strip() for tup in raw.columns]
raw.columns = raw.columns.astype(str).str.strip().str.lower()

# Remove duplicated columns (keep first)
raw = raw.loc[:, ~raw.columns.duplicated()]

# Map synonyms
colmap = {
    "brand":"brand","brand name":"brand",
    "category":"category","sector":"category",
    "market":"market","country":"market","geography":"market",
    "year":"year","study_year":"year","wave":"year",
    "differentiation_rank":"differentiation_rank","diff_rank":"differentiation_rank",
    "relevance_rank":"relevance_rank","esteem_rank":"esteem_rank",
    "knowledge_rank":"knowledge_rank",
    "mib_innovation_rank":"mib_innovation_rank","innovation_rank":"mib_innovation_rank"
}
df = raw.rename(columns={c: colmap.get(c, c) for c in raw.columns}).copy()

req = ["brand","category","market","year",
        "differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]
missing = [c for c in req if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# Clean values
df["year"] = df["year"].apply(clean_year)
for c in ["differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]:
    df[c] = df[c].apply(to_num)

# Drop empty year rows
df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

# Aggregate to single brand×category×market×year
agg = (
    df.groupby(["brand","category","market","year"], dropna=False)[
        ["differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]
    ].mean(numeric_only=True)
    .reset_index()
)

# ---------------- DNA Metrics ----------------
out = agg.rename(columns=str.title)
out["D"] = out["Differentiation_Rank"]
out["N"] = (out["Relevance_Rank"] + out["Esteem_Rank"]) / 2
out["A"] = out["Knowledge_Rank"] * k_knowledge + out["Mib_Innovation_Rank"] * (1 - k_knowledge)
out["DNA"] = out[["D","N","A"]].mean(axis=1)

out["DLevel"] = out["D"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["NLevel"] = out["N"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["ALevel"] = out["A"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["Pattern"] = out["DLevel"] + " / " + out["NLevel"] + " / " + out["ALevel"]
out["AutoInsight"] = out.apply(lambda r: auto_insight(r["DLevel"], r["NLevel"], r["ALevel"]), axis=1)
out["AutoAction"] = out["Pattern"].apply(pattern_to_action)

# ---------------- Sidebar Filters ----------------
brands = sorted(out["Brand"].dropna().unique())
focus = st.sidebar.selectbox("Focus brand", brands, index=0)
category = out[out["Brand"]==focus]["Category"].iloc[0]
peers_all = sorted(out[out["Category"]==category]["Brand"].unique())
peers = st.sidebar.multiselect("Peer brands for Category average", [b for b in peers_all if b!=focus], default=peers_all[:3])

# ---------------- Category averages ----------------
if peers:
    peers_df = out[out["Brand"].isin(peers)]
else:
    peers_df = out[(out["Category"]==category) & (out["Brand"]!=focus)]

cat_avg = (
    peers_df.groupby(["Category","Year"], dropna=False)[["D","N","A","DNA"]]
    .mean(numeric_only=True)
    .reset_index()
    .rename(columns={"D":"CategoryAvg_D","N":"CategoryAvg_N","A":"CategoryAvg_A","DNA":"CategoryAvg_DNA"})
)

view = out.merge(cat_avg, on=["Category","Year"], how="left")

# Δ vs Category (%)
for m in ["D","N","A","DNA"]:
    view[f"Δ vsCategory_{m} [%]"] = (view[m] - view[f"CategoryAvg_{m}"]) / view[f"CategoryAvg_{m}"].replace(0, np.nan) * 100

# YoY %
view = view.sort_values(["Brand","Category","Market","Year"])
for m in ["D","N","A","DNA"]:
    view[f"Δ YoY {m} [%]"] = view.groupby(["Brand","Category","Market"], dropna=False)[m].pct_change() * 100

# ---------------- Results Table ----------------
st.subheader("Results Table")
view_disp = view.copy()
for c in view_disp.columns:
    if c.startswith("Δ") or "Avg" in c:
        view_disp[c] = view_disp[c].round(2)
st.dataframe(view_disp, use_container_width=True)

# ---------------- Charts ----------------
focus_df = view[view["Brand"]==focus].copy()
if not focus_df.empty:
    latest = focus_df["Year"].max()
    cat_row = cat_avg[cat_avg["Year"]==latest].mean(numeric_only=True)
    bars = focus_df[focus_df["Year"]==latest][["D","N","A","DNA"]].mean().reset_index()
    bars.columns = ["Pillar","Score"]
    bars["CategoryAvg"] = [
        cat_row["CategoryAvg_D"],cat_row["CategoryAvg_N"],cat_row["CategoryAvg_A"],cat_row["CategoryAvg_DNA"]
    ]
    fig_bar = px.bar(bars, x="Pillar", y="Score", title=f"{focus} — {latest}")
    fig_bar.add_trace(go.Scatter(x=bars["Pillar"], y=bars["CategoryAvg"], mode="markers", name="Category avg"))
    st.plotly_chart(fig_bar, use_container_width=True)

    trend = focus_df.groupby("Year")[["D","N","A","DNA"]].mean().reset_index()
    trend_cat = cat_avg.groupby("Year")[["CategoryAvg_D","CategoryAvg_N","CategoryAvg_A","CategoryAvg_DNA"]].mean().reset_index()
    fig_trend = go.Figure()
    for m in ["D","N","A","DNA"]:
        fig_trend.add_trace(go.Scatter(x=trend["Year"], y=trend[m], mode="lines+markers", name=f"{focus} {m}"))
        fig_trend.add_trace(go.Scatter(x=trend_cat["Year"], y=trend_cat[f"CategoryAvg_{m}"], mode="lines+markers", name=f"Category {m}"))
    fig_trend.update_layout(title=f"{focus} vs Category — Trend", xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig_trend, use_container_width=True)

# ---------------- Export ----------------
st.download_button("Export CSV", view.to_csv(index=False).encode("utf-8"), "dna_from_bav_output.csv")
