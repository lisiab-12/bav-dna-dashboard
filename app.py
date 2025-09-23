import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ───────────────────────────── Page config ─────────────────────────────
st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")
st.title("DNA from BAV — Converter")
st.caption("Upload WPP BAV raw export → get Distinction, Nuance, Adventure, YoY deltas, and auto-insights.")

# ───────────────────────────── Sidebar controls ────────────────────────
st.sidebar.header("Controls")
high_threshold = st.sidebar.number_input("High threshold (≥)", min_value=0, max_value=100, value=65, step=1)
mid_threshold  = st.sidebar.number_input("Mid threshold (≥, < High)", min_value=0, max_value=100, value=50, step=1)

k_knowledge = st.sidebar.slider("Adventure: Knowledge weight (k)", 0.0, 1.0, 0.40, 0.01)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = {k_knowledge:.2f}×Knowledge + {k_innovation:.2f}×Innovation")

st.sidebar.markdown("---")
st.sidebar.write("**Download:** use the 'Export CSV' button at the bottom.")

# ───────────────────────────── Helpers ─────────────────────────────────
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
        "High / Mid / High": "Dial up emotional resonance & mainstream cues without blunting edge.",
        "High / Low / High": "Build warmth and everyday relevance via service & community moments.",
        "High / High / Mid": "Inject fresh provocations to avoid drift into safe heritage-only.",
        "Mid / High / High": "Crystallise Distinction (codes) to cut through; clarify POV.",
        "High / Mid / Mid": "Distinct but stagnant; pilot bolder innovation & experiences.",
        "Mid / High / Mid": "Sharpen codes and simplify message; anchor in a felt need.",
        "Mid / Mid / High": "Anchor proposition and proof; reduce vagueness while keeping momentum.",
        "Low / Low / Low":  "Back to basics: define who you are; craft minimum viable brand world.",
    }
    return mapping.get(pattern, "Set foundational codes; then build warmth and momentum.")

def auto_insight(d_level, n_level, a_level):
    d_txt = {"High":"Distinctive","Mid":"Some distinctiveness","Low":"Blends in"}.get(d_level,"Blends in")
    n_txt = {"High":"Emotionally resonant","Mid":"Some resonance","Low":"Emotionally flat"}.get(n_level,"Emotionally flat")
    a_txt = {"High":"Progressive","Mid":"Some novelty","Low":"Conservative"}.get(a_level,"Conservative")
    return f"{d_txt} | {n_txt} | {a_txt}"

# ───────────────────────────── Upload ──────────────────────────────────
required_rank_note = (
    "Required rank fields in raw pull: "
    "brand, category, market, year, differentiation_rank, relevance_rank, "
    "esteem_rank, knowledge_rank, mib_innovation_rank"
)
up = st.file_uploader("Upload BAV CSV", type=["csv"])
if up is None:
    st.info("Upload a CSV to begin. " + required_rank_note)
    st.stop()

# Read & normalize headers
df = pd.read_csv(up)
df.columns = df.columns.str.strip().str.lower()

# Require ONLY the rank fields
def need(name):
    if name not in df.columns:
        st.error(f"Missing required column: **{name}**")
        st.write("Detected headers:", list(df.columns))
        st.stop()

for col in [
    "brand","category","market","year",
    "differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"
]:
    need(col)

# Build canonical frame (rank-only)
df = pd.DataFrame({
    "Brand":            df["brand"],
    "Category":         df["category"],
    "Market":           df["market"],
    "Year":             df["year"],
    "Differentiation":  df["differentiation_rank"],
    "Relevance":        df["relevance_rank"],
    "Esteem":           df["esteem_rank"],
    "Knowledge":        df["knowledge_rank"],
    "Innovation":       df["mib_innovation_rank"],
})

# Types / cleaning
df["Year"] = df["Year"].apply(lambda x: int(float(str(x).strip())) if str(x).strip() != "" else np.nan)
for col in ["Differentiation","Relevance","Esteem","Knowledge","Innovation"]:
    df[col] = df[col].apply(to_num)

# ───────────────────────────── DNA computation ─────────────────────────
out = df.copy()
out["D"]   = out["Differentiation"]
out["N"]   = (out["Relevance"] + out["Esteem"]) / 2
out["A"]   = out["Knowledge"] * k_knowledge + out["Innovation"] * (1 - k_knowledge)
out["DNA"] = out[["D","N","A"]].mean(axis=1)

out["DLevel"] = out["D"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["NLevel"] = out["N"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["ALevel"] = out["A"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
out["Pattern"] = out["DLevel"] + " / " + out["NLevel"] + " / " + out["ALevel"]
out["AutoInsight"] = out.apply(lambda r: auto_insight(r["DLevel"], r["NLevel"], r["ALevel"]), axis=1)
out["AutoAction"]  = out["Pattern"].apply(pattern_to_action)

# Averages & deltas vs averages (Brand/Year & Category/Year)
brand_avg = (out.groupby(["Brand","Year"], dropna=False)[["D","N","A","DNA"]]
               .mean().rename(columns=lambda c: f"BrandAvg_{c}"))
cat_avg   = (out.groupby(["Category","Year"], dropna=False)[["D","N","A","DNA"]]
               .mean().rename(columns=lambda c: f"CategoryAvg_{c}"))

out = (out.merge(brand_avg, on=["Brand","Year"], how="left")
          .merge(cat_avg,   on=["Category","Year"], how="left"))

for m in ["D","N","A","DNA"]:
    out[f"vsBrand_{m}"]    = out[m] - out[f"BrandAvg_{m}"]
    out[f"vsCategory_{m}"] = out[m] - out[f"CategoryAvg_{m}"]

# YoY deltas (within Brand+Category)
out = out.sort_values(["Brand","Category","Year"])
out[["dD","dN","dA","dDNA"]] = np.nan
for (b,c), grp in out.groupby(["Brand","Category"], dropna=False):
    idx = grp.index.tolist()
    for prev, curr in zip(idx, idx[1:]):
        out.loc[curr, "dD"]   = out.loc[curr, "D"]   - out.loc[prev, "D"]
        out.loc[curr, "dN"]   = out.loc[curr, "N"]   - out.loc[prev, "N"]
        out.loc[curr, "dA"]   = out.loc[curr, "A"]   - out.loc[prev, "A"]
        out.loc[curr, "dDNA"] = out.loc[curr, "DNA"] - out.loc[prev, "DNA"]

# ───────────────────────────── View controls (focus/peers) ─────────────
all_years   = out["Year"].dropna().astype(int)
year_min, year_max = int(all_years.min()), int(all_years.max())
all_brands  = sorted(out["Brand"].dropna().unique())

st.sidebar.markdown("### View controls")
focus_brand = st.sidebar.selectbox("Focus brand", all_brands)
peer_brands = st.sidebar.multiselect(
    "Peer brands for 'Category average' (choose any set)",
    [b for b in all_brands if b != focus_brand],
    default=[]
)
metrics_to_show = st.sidebar.multiselect(
    "Metrics to show",
    ["D","N","A","DNA"],
    default=["D","N","A"]
)
yr_from, yr_to = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

# Subset by year range
view = out[(out["Year"] >= yr_from) & (out["Year"] <= yr_to)].copy()

# Focus brand and peers in the chosen window
focus_df = view[view["Brand"] == focus_brand].copy()
peers_df = view[view["Brand"].isin(peer_brands)].copy()

# Custom peer average per year based on selected peers
if not peers_df.empty:
    peer_avg = (peers_df.groupby("Year")[["D","N","A","DNA"]]
                    .mean()
                    .rename(columns=lambda c: f"PeerAvg_{c}")
                    .reset_index())
    focus_df = focus_df.merge(peer_avg, on="Year", how="left")

    # deltas vs selected peer average
    for m in ["D","N","A","DNA"]:
        col = f"PeerAvg_{m}"
        if col in focus_df:
            focus_df[f"vsPeer_{m}"] = focus_df[m] - focus_df[col]
else:
    peer_avg = pd.DataFrame({"Year": [], "PeerAvg_D": [], "PeerAvg_N": [], "PeerAvg_A": [], "PeerAvg_DNA": []})

# ───────────────────────────── Results table ───────────────────────────
st.markdown("### Results Table")
st.dataframe(view, use_container_width=True)

# ───────────────────────────── Charts ──────────────────────────────────
st.markdown("### Charts")

# 1) Latest-year bar for focus brand with YoY % change callouts
latest_two = (focus_df.dropna(subset=["Year"])
                        .sort_values("Year")
                        .tail(2))
if len(latest_two) >= 1:
    cur = latest_two.iloc[-1]
    pct_change = []
    vals = []
    for m in metrics_to_show:
        vals.append(cur[m] if m in cur else np.nan)
        if len(latest_two) == 2 and m in latest_two.columns and pd.notna(latest_two.iloc[-2][m]) and latest_two.iloc[-2][m] != 0:
            prev = latest_two.iloc[-2][m]
            pct_change.append((cur[m] - prev) / abs(prev) * 100.0)
        else:
            pct_change.append(np.nan)

    fig_bar = px.bar(
        x=metrics_to_show,
        y=vals,
        labels={"x":"Metric","y":"Score"},
        title=f"{focus_brand} — latest year {int(cur['Year'])} (YoY change shown)"
    )
    fig_bar.update_traces(
        text=[f"{p:+.1f}%" if pd.notna(p) else "" for p in pct_change],
        textposition="outside"
    )
    fig_bar.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_bar, use_container_width=True)

# 2) Trend lines for focus brand, optional peers, selected metrics
brands_to_plot = [focus_brand] + (peer_brands if peer_brands else [])
trend = view[view["Brand"].isin(brands_to_plot)].copy()
tlong = trend.melt(id_vars=["Brand","Year"], value_vars=metrics_to_show,
                   var_name="Metric", value_name="Score")
if not tlong.empty:
    fig_line = px.line(
        tlong, x="Year", y="Score", color="Brand", line_dash="Metric",
        markers=True, title="Trend (selected brands & metrics)"
    )
    fig_line.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_line, use_container_width=True)

# 3) Optional: overlay the Peer Average line (computed from selected peers)
if not peers_df.empty:
    peer_long = (peer_avg.melt(id_vars=["Year"],
                               value_vars=[f"PeerAvg_{m}" for m in metrics_to_show if f"PeerAvg_{m}" in peer_avg.columns],
                               var_name="Metric", value_name="Score"))
    if not peer_long.empty:
        peer_long["Brand"]  = "Peer Avg"
        peer_long["Metric"] = peer_long["Metric"].str.replace("PeerAvg_", "", regex=False)

        combo = pd.concat([tlong, peer_long], ignore_index=True) if not tlong.empty else peer_long
        fig_line_peer = px.line(
            combo, x="Year", y="Score", color="Brand", line_dash="Metric",
            markers=True, title="Trend with Peer Average"
        )
        fig_line_peer.update_layout(yaxis=dict(range=[0,100]))
        st.plotly_chart(fig_line_peer, use_container_width=True)

# ───────────────────────────── Export ──────────────────────────────────
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV (all rows)", data=csv, file_name="dna_from_bav_output.csv")
