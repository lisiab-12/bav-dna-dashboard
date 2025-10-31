# ---- Interactive table filters (Brands) ----
table_brands = st.sidebar.multiselect(
    "Filter table to brands",
    sorted(view["Brand"].dropna().unique().tolist()),
    default=sorted(view["Brand"].dropna().unique().tolist())
)
table_df = view[view["Brand"].isin(table_brands)].copy()

# ---- Drop Brand Avg & vsBrand_* from display (keep Category averages and vsCategory_*) ----
cols_to_drop = [c for c in table_df.columns if c.startswith("BrandAvg_") or c.startswith("vsBrand_")]
table_df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

# ---- Replace absolute YoY deltas with percentage YoY & rename headings ----
# (per Brand+Category within selected rows)
table_df = table_df.sort_values(["Brand","Category","Year"])
for (b,c), grp in table_df.groupby(["Brand","Category"], dropna=False):
    idx = grp.index.tolist()
    for prev, curr in zip(idx, idx[1:]):
        for m in ["D","N","A","DNA"]:
            prev_val, curr_val = table_df.loc[prev, m], table_df.loc[curr, m]
            if pd.notna(prev_val) and prev_val != 0 and pd.notna(curr_val):
                pct = (curr_val - prev_val) / abs(prev_val) * 100.0
            else:
                pct = np.nan
            table_df.loc[curr, f"YoY%_{m}"] = pct

# Now drop the old absolute deltas (dD, dN, dA, dDNA) so we only show %
table_df.drop(columns=["dD","dN","dA","dDNA"], inplace=True, errors="ignore")

# ---- Build grouped (multi-index) headers to make the table self-explanatory ----
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

    # Levels & patterning
    ("Levels & pattern","DLevel"),
    ("Levels & pattern","NLevel"),
    ("Levels & pattern","ALevel"),
    ("Levels & pattern","Pattern"),
    ("Levels & pattern","AutoInsight"),
    ("Levels & pattern","AutoAction"),

    # Category averages (kept)
    ("Category avg","CategoryAvg_D"),
    ("Category avg","CategoryAvg_N"),
    ("Category avg","CategoryAvg_A"),
    ("Category avg","CategoryAvg_DNA"),

    # Delta vs category average (points)
    ("Δ vs category (pts)","vsCategory_D"),
    ("Δ vs category (pts)","vsCategory_N"),
    ("Δ vs category (pts)","vsCategory_A"),
    ("Δ vs category (pts)","vsCategory_DNA"),

    # YoY % change
    ("YoY change %","YoY%_D"),
    ("YoY change %","YoY%_N"),
    ("YoY change %","YoY%_A"),
    ("YoY change %","YoY%_DNA"),
]

# Filter to existing columns in case some are missing
display_cols = [(g,c) for (g,c) in display_cols if c in table_df.columns]

multi_cols = pd.MultiIndex.from_tuples(display_cols, names=["",""])
table_to_show = table_df[[c for (_,c) in display_cols]].copy()
table_to_show.columns = multi_cols

st.markdown("### Results Table")
st.dataframe(
    table_to_show,
    use_container_width=True,
    column_config={
        # Ensure Year prints as 2023 not 2023.0
        ("Identifiers","Year"): st.column_config.NumberColumn(format="%d"),
        # Format YoY % columns nicely
        ("YoY change %","YoY%_D")  : st.column_config.NumberColumn(format="%.1f%%"),
        ("YoY change %","YoY%_N")  : st.column_config.NumberColumn(format="%.1f%%"),
        ("YoY change %","YoY%_A")  : st.column_config.NumberColumn(format="%.1f%%"),
        ("YoY change %","YoY%_DNA"): st.column_config.NumberColumn(format="%.1f%%"),
    }
)
