import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")

st.title("DNA from BAV — Converter")
st.caption("Upload WPP BAV raw export → get Distinction, Nuance, Adventure, YoY deltas, and auto-insights.")

# ── Sidebar controls ───────────────────────────────────────────────────────────────
st.sidebar.header("Controls")
high_threshold = st.sidebar.number_input("High threshold (≥)", min_value=0, max_value=100, value=65, step=1)
mid_threshold  = st.sidebar.number_input("Mid threshold (≥, < High)", min_value=0, max_value=100, value=50, step=1)

k_knowledge = st.sidebar.slider("Adventure: Knowledge weight (k)", 0.0, 1.0, 0.40, 0.01)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = {k_knowledge:.2f}×Knowledge + {k_innovation:.2f}×Innovation")

st.sidebar.markdown("---")
st.sidebar.write("**Download:** use the 'Export CSV' button at the bottom.")

# ── Helpers ───────────────────────────────────────────────────────────────────────
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
        "Low / Low / Low":  "Back to basics: define who you are; craft minimum viable brand world.",
    }
    return mapping.get(pattern, "Set foundational codes; then build warmth and momentum.")

def auto_insight(d_level, n_level, a_level):
    d_txt = {"High":"Distinctive","Mid":"Some distinctiveness","Low":"Blends in"}.get(d_level,"Blends in")
    n_txt = {"High":"Emotionally resonant","Mid":"Some resonance","Low":"Emotionally flat"}.get(n_level,"Emotionally flat")
    a_txt = {"High":"Progressive","Mid":"Some novelty","Low":"Conservative"}.get(a_level,"Conservative")
    return f"{d_txt} | {n_txt} | {a_txt}"

# ── File upload ───────────────────────────────────────────────────────────────────
required = ["Brand","Category","Market","Year","Differentiation","Relevance","Esteem","Knowledge","Innovation"]
up = st.file_uploader("Upload BAV CSV", type=["csv"])

if up is None:
    st.info("Upload a CSV to begin. Required rank fields in your raw pull: "
            "brand, category, market, year, differentiation_rank, relevance_rank, esteem_rank, knowledge_rank, mib_innovation_rank")
    st.stop()

# Read & normalize headers
df = pd.read_csv(up)
df.columns = df.columns.str.strip().str.lower()

# Require ONLY the rank fields you use
def need(name):
    if name not in df.columns:
        st.error(f"Missing required column: **{name}**")
        st.write("Detected headers:", list(df.columns))
        st.stop()

need("brand")
need("category")
need("market")
need("year")
need("differentiation_rank")
need("relevance_rank")
need("esteem_rank")
need("knowledge_rank")
need("mib_innovation_rank")

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

# ── DNA computation ───────────────────────────────────────────────────────────────
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
out["AutoAction"] = out["Pattern"].apply(pattern_to_action)

# ── Averages & deltas vs averages (Brand/Year & Category/Year) ───────────────────
brand_avg = (out.groupby(["Brand","Year"], dropna=False)[["D","N","A","DNA"]]
               .mean().rename(columns=lambda c: f"BrandAvg_{c}"))
cat_avg   = (out.groupby(["Category","Year"], dropna=False)[["D","N","A","DNA"]]
               .mean().rename(columns=lambda c: f"CategoryAvg_{c}"))

out = (out.merge(brand_avg, on=["Brand","Year"], how="left")
          .merge(cat_avg,   on=["Category","Year"], how="left"))

for m in ["D","N","A","DNA"]:
    out[f"vsBrand_{m}"]    = out[m] - out[f"BrandAvg_{m}"]
    out[f"vsCategory_{m}"] = out[m] - out[f"CategoryAvg_{m}"]

# ── YoY deltas (within Brand+Category) ───────────────────────────────────────────
out = out.sort_values(["Brand","Category","Year"])
out[["dD","dN","dA","dDNA"]] = np.nan
for (b,c), grp in out.groupby(["Brand","Category"], dropna=False):
    idx = grp.index.tolist()
    for prev, curr in zip(idx, idx[1:]):
        out.loc[curr, "dD"]   = out.loc[curr, "D"]   - out.loc[prev, "D"]
        out.loc[curr, "dN"]   = out.loc[curr, "N"]   - out.loc[prev, "N"]
        out.loc[curr, "dA"]   = out.loc[curr, "A"]   - out.loc[prev, "A"]
        out.loc[curr, "dDNA"] = out.loc[curr, "DNA"] - out.loc[prev, "DNA"]

# ── Output table ─────────────────────────────────────────────────────────────────
st.markdown("### Results Table")
st.dataframe(out, use_container_width=True)

# ── Charts ───────────────────────────────────────────────────────────────────────
st.markdown("### Charts")
agg  = out.groupby(["Brand","Category","Year"], dropna=False)[["D","N","A"]].mean().reset_index()
melt = agg.melt(id_vars=["Brand","Category","Year"], value_vars=["D","N","A"], var_name="Pillar", value_name="Score")
fig_bar = px.bar(melt, x="Pillar", y="Score", color="Pillar", barmode="group", title="D / N / A Breakdown")
fig_bar.update_layout(yaxis=dict(range=[0,100]))
st.plotly_chart(fig_bar, use_container_width=True)

if len(out["Brand"].unique()) == 1 and len(out["Category"].unique()) == 1:
    trend = out.sort_values("Year")
    tlong = trend.melt(id_vars=["Year"], value_vars=["D","N","A","DNA"], var_name="Metric", value_name="Score")
    fig_line = px.line(tlong, x="Year", y="Score", color="Metric", markers=True, title="YoY Trends")
    fig_line.update_layout(yaxis=dict(range=[0,100]))
    st.plotly_chart(fig_line, use_container_width=True)

    latest = trend.dropna(subset=["Year"]).sort_values("Year").tail(1)
    if not latest.empty:
        r = [float(latest["D"]), float(latest["N"]), float(latest["A"])]
        theta = ["D","N","A"]
        fig_radar = go.Figure(data=go.Scatterpolar(r=r+[r[0]], theta=theta+[theta[0]], fill='toself'))
        fig_radar.update_layout(title=f"Profile — {int(latest['Year'].iloc[0])}",
                                polar=dict(radialaxis=dict(visible=True, range=[0,100])))
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Export ───────────────────────────────────────────────────────────────────────
csv = out.to_csv(index=False).encode("utf-8")
st.download_button("Export CSV (all rows)", data=csv, file_name="dna_from_bav_output.csv")
