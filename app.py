import pandas as pd
import numpy as np
import streamlit as st

# ---------- File upload ----------
st.subheader("Upload BAV Raw CSV")
up = st.file_uploader("Choose CSV", type=["csv"], accept_multiple_files=False, key="bav_csv")

if up is None:
    st.info("👆 Upload your BAV CSV to begin.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_csv(file):
    # Try pyarrow for speed if available, else fallback
    try:
        df = pd.read_csv(file, engine="pyarrow")
    except Exception:
        file.seek(0)  # reset pointer after failed parse
        df = pd.read_csv(file, low_memory=False)
    return df

raw = load_csv(up)

# ---------- Normalise columns ----------
# Flatten multi-index headers (if present), lowercase, strip
if isinstance(raw.columns, pd.MultiIndex):
    raw.columns = [
        "_".join([str(x) for x in tup if str(x) != "nan"]).strip()
        for tup in raw.columns
    ]
raw.columns = raw.columns.astype(str).str.strip().str.lower()

# If there are exact duplicate column names, keep the first occurrence
raw = raw.loc[:, ~raw.columns.duplicated()]

def pick_first(df, candidates):
    """Return first matching column as a 1-D Series or None."""
    found = [c for c in candidates if c in df.columns]
    if not found:
        return None
    ser = df[found[0]]
    if isinstance(ser, pd.DataFrame):  # guard if duplicated names created a 2-D slice
        ser = ser.iloc[:, 0]
    return ser

# Map your needed columns (accept a few common variants)
brand_ser  = pick_first(raw, ["brand", "brand name"])
cat_ser    = pick_first(raw, ["category", "sector"])
mkt_ser    = pick_first(raw, ["market", "country", "geography"])
year_ser   = pick_first(raw, ["year", "study_year", "fieldwork_year", "wave"])
diff_ser   = pick_first(raw, ["differentiation_rank", "diff_rank", "differentiation rank"])
rel_ser    = pick_first(raw, ["relevance_rank", "relevance rank"])
est_ser    = pick_first(raw, ["esteem_rank", "esteem rank"])
know_ser   = pick_first(raw, ["knowledge_rank", "knowledge rank"])
innov_ser  = pick_first(raw, ["mib_innovation_rank", "innovation_rank", "innovation rank"])

missing_bits = []
if brand_ser is None:  missing_bits.append("brand")
if cat_ser   is None:  missing_bits.append("category/sector")
if mkt_ser   is None:  missing_bits.append("market/country")
if year_ser  is None:  missing_bits.append("year/study_year/fieldwork_year/wave")
if diff_ser  is None:  missing_bits.append("differentiation_rank")
if rel_ser   is None:  missing_bits.append("relevance_rank")
if est_ser   is None:  missing_bits.append("esteem_rank")
if know_ser  is None:  missing_bits.append("knowledge_rank")
if innov_ser is None:  missing_bits.append("mib_innovation_rank/innovation_rank")

if missing_bits:
    st.error(f"Missing required columns (after normalizing): {missing_bits}")
    st.dataframe(raw.head(20))
    st.stop()

df = pd.DataFrame({
    "brand":  brand_ser,
    "category": cat_ser,
    "market": mkt_ser,
    "year":   year_ser,
    "differentiation_rank": diff_ser,
    "relevance_rank":       rel_ser,
    "esteem_rank":          est_ser,
    "knowledge_rank":       know_ser,
    "mib_innovation_rank":  innov_ser,
})

# ---------- Cleaning ----------
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
        v = int(float(s))
        return v if 1900 <= v <= 2100 else np.nan
    except Exception:
        return np.nan

df["year"] = df["year"].apply(clean_year)
for c in ["differentiation_rank","relevance_rank","esteem_rank","knowledge_rank","mib_innovation_rank"]:
    df[c] = df[c].apply(to_num)

df = df.dropna(subset=["year"]).copy()
df["year"] = df["year"].astype(int)

st.success("✅ Data loaded and normalized.")
st.write("Preview:")
st.dataframe(df.head(20), use_container_width=True)
