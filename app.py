
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="DNA from BAV — Converter", layout="wide")

st.title("DNA from BAV — Converter")
st.caption("Upload WPP BAV raw export → get Distinction, Nuance, Adventure, YoY deltas, and auto-insights.")

# Sidebar controls
st.sidebar.header("Controls")
high_threshold = st.sidebar.number_input("High threshold (≥)", min_value=0, max_value=100, value=65, step=1)
mid_threshold = st.sidebar.number_input("Mid threshold (≥, < High)", min_value=0, max_value=100, value=50, step=1)

k_knowledge = st.sidebar.slider("Adventure: Knowledge weight (k)", 0.0, 1.0, 0.40, 0.01)
k_innovation = 1 - k_knowledge
st.sidebar.caption(f"Adventure = {k_knowledge:.2f}×Knowledge + {k_innovation:.2f}×Innovation")

st.sidebar.markdown("---")
st.sidebar.write("**Download:** use the 'Export CSV' button at the bottom.")

# File upload
template_cols = ["Brand","Category","Market","Year","Differentiation","Relevance","Esteem","Knowledge","Innovation"]
up = st.file_uploader("Upload BAV CSV", type=["csv"])
# --- Fix column names so they match the app requirements ---
required_columns = {
    "brand": "Brand",
    "category": "Category",
    "market": "Market",
    "year": "Year",
    "differentiation": "Differentiation",
    "relevance": "Relevance",
    "esteem": "Esteem",
    "knowledge": "Knowledge",
    "innovation": "Innovation"
}

# Normalize: lowercase, strip spaces
df.columns = df.columns.str.strip().str.lower()

# Rename if matches dictionary
df.rename(columns=required_columns, inplace=True)

# Check if all required columns exist
missing = [col for col in required_columns.values() if col not in df.columns]
if missing:
    st.error(f"Missing required columns even after renaming: {missing}")



# If no file yet, show a hint and stop cleanly (prevents NameError)
if up is None:
    st.info("Upload a CSV to begin. Required columns: " + ", ".join(template_cols))
    st.stop()

# Read the uploaded CSV
df = pd.read_csv(up)

# --- Fix column names so they match template_cols ---
# Map common variants -> canonical names
rename_map = {
    "brand": "Brand",
    "brand name": "Brand",

    "category": "Category",
    "sector": "Category",

    "market": "Market",
    "country": "Market",
    "geography": "Market",

    "year": "Year",
    "wave": "Year",
    "fieldwork year": "Year",
    "study year": "Year",

    "differentiation": "Differentiation",
    "diff": "Differentiation",

    "relevance": "Relevance",
    "rel": "Relevance",

    "esteem": "Esteem",

    "knowledge": "Knowledge",
    "know": "Knowledge",

    "innovation": "Innovation",
    "innov": "Innovation",
}

# Normalize headers then rename
df.columns = df.columns.str.strip().str.lower()
df.rename(columns=rename_map, inplace=True)

# Check required columns and keep only what we need
missing = [c for c in template_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns even after renaming: {missing}")
    st.write("Detected headers:", list(df.columns))
    st.stop()

df = df[template_cols]  # drop extra columns

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

# Interpretation mapping
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

if up is not None:
    df = pd.read_csv(up)
    missing = [c for c in template_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["Year"] = df["Year"].apply(lambda x: int(float(str(x).strip())) if str(x).strip() != "" else np.nan)
    for col in ["Differentiation","Relevance","Esteem","Knowledge","Innovation"]:
        df[col] = df[col].apply(to_num)

    out = df.copy()
    out["D"] = out["Differentiation"]
    out["N"] = (out["Relevance"] + out["Esteem"]) / 2
    out["A"] = out["Knowledge"] * k_knowledge + out["Innovation"] * (1 - k_knowledge)
    out["DNA"] = out[["D","N","A"]].mean(axis=1)

    out["DLevel"] = out["D"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
    out["NLevel"] = out["N"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
    out["ALevel"] = out["A"].apply(lambda x: compute_levels(x, mid_threshold, high_threshold))
    out["Pattern"] = out["DLevel"] + " / " + out["NLevel"] + " / " + out["ALevel"]
    out["AutoInsight"] = out.apply(lambda r: auto_insight(r["DLevel"], r["NLevel"], r["ALevel"]), axis=1)
    out["AutoAction"] = out["Pattern"].apply(pattern_to_action)

    # YoY deltas
    out = out.sort_values(["Brand","Category","Year"])
    out[["dD","dN","dA","dDNA"]] = np.nan
    for key, grp in out.groupby(["Brand","Category"], dropna=False):
        idx = grp.index.tolist()
        prev = None
        for i in idx:
            if prev is not None:
                out.loc[i, "dD"] = out.loc[i, "D"] - out.loc[prev, "D"]
                out.loc[i, "dN"] = out.loc[i, "N"] - out.loc[prev, "N"]
                out.loc[i, "dA"] = out.loc[i, "A"] - out.loc[prev, "A"]
                out.loc[i, "dDNA"] = out.loc[i, "DNA"] - out.loc[prev, "DNA"]
            prev = i

    st.markdown("### Results Table")
    st.dataframe(out, use_container_width=True)

    # Charts
    st.markdown("### Charts")
    agg = out.groupby(["Brand","Category","Year"], dropna=False)[["D","N","A"]].mean().reset_index()
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
            fig_radar.update_layout(title=f"Profile — {int(latest['Year'].iloc[0])}", polar=dict(radialaxis=dict(visible=True, range=[0,100])))
            st.plotly_chart(fig_radar, use_container_width=True)

    # Export
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV (all rows)", data=csv, file_name="dna_from_bav_output.csv")

else:
    st.info("Upload a CSV to begin. Required columns: Brand, Category, Market, Year, Differentiation, Relevance, Esteem, Knowledge, Innovation")

