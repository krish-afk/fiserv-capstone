"""
Create hypothesis-ready CSV splits from your FSBI dataset.

Expected columns (at minimum):
- Period
- Geo
- Sector Name
- Sub-Sector Name
- plus your FSBI + macro feature columns (including lags/MAs)

Outputs (CSVs) written to:
../../../data/processed_data/splits/

Splits created:
1) Single MSAs (one CSV per MSA)
2) Region groupings (Northeast/Midwest/South/West) for STATE rows
3) Energy states vs Non-energy states (STATE rows)
4) Sector level datasets:
   - all sectors together (everything)
   - one CSV per Sector Name (excluding ALL unless you want it)
5) Leading-only subset (lag/MA features only + ID cols + target)
6) Entertainment & Recreation subset (Sector Name contains "Arts" / "Entertainment" / "Recreation")
"""

import re
import os
from pathlib import Path

import pandas as pd


# -----------------------------
# CONFIG (edit if needed)
# -----------------------------
import os
INPUT_CSV = "../../../data/processed_data/processed_fiserv_macro_data.csv"  # <-- change this
OUT_DIR = Path("../../../data/processed_data/splits/").resolve()

# Common region mapping (Census-style). DC included in South bucket.
REGIONS = {
    "Northeast": ["CT","ME","MA","NH","RI","VT","NJ","NY","PA"],
    "Midwest":   ["IL","IN","MI","OH","WI","IA","KS","MN","MO","NE","ND","SD"],
    "South":     ["DE","FL","GA","MD","NC","SC","VA","DC","WV","AL","KY","MS","TN","AR","LA","OK","TX"],
    "West":      ["AZ","CO","ID","MT","NV","NM","UT","WY","AK","CA","HI","OR","WA"],
}

# Energy-heavy states (adjust if your capstone defines differently)
ENERGY_STATES = {"TX","LA","OK","ND","NM","WY","AK","CO","WV"}


# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    """Make filesystem-friendly filenames."""
    s = str(s).strip()
    s = re.sub(r"[^\w\s\-\.\&]", "", s)   # drop weird punctuation
    s = re.sub(r"\s+", "_", s)           # spaces -> underscores
    s = s.replace("&", "and")
    return s[:160]  # avoid ultra-long filenames

def is_state_geo(geo: str) -> bool:
    """Your Geo column sometimes looks like state codes (AK, CA...) and sometimes MSA names."""
    if pd.isna(geo):
        return False
    geo = str(geo).strip()
    return bool(re.fullmatch(r"[A-Z]{2}", geo))

def is_msa_geo(geo: str) -> bool:
    """Heuristic: if not a 2-letter state and not obvious national label, treat as MSA-like."""
    if pd.isna(geo):
        return False
    geo = str(geo).strip()
    if is_state_geo(geo):
        return False
    # Exclude national-ish labels if they appear
    if geo.upper() in {"US", "USA", "UNITED_STATES", "UNITED STATES"}:
        return False
    # Many MSAs contain hyphens and spaces and state abbreviations; but keep heuristic broad
    return True

def ensure_period_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Period if it is like YYYYMMDD as int/string."""
    if "Period" not in df.columns:
        raise ValueError("Missing required column: Period")
    # If already datetime-like, keep it; else parse YYYYMMDD
    if not pd.api.types.is_datetime64_any_dtype(df["Period"]):
        df["Period"] = pd.to_datetime(df["Period"].astype(str), format="%Y%m%d", errors="coerce")
    if df["Period"].isna().any():
        bad = df.loc[df["Period"].isna(), "Period"].head(5)
        raise ValueError(f"Some Period values could not be parsed. Examples:\n{bad}")
    return df

def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# -----------------------------
# Main
# -----------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = ensure_period_datetime(df)

    required = {"Geo", "Sector Name", "Sub-Sector Name"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- (0) Save a cleaned copy (optional but handy)
    write_csv(df, OUT_DIR / "00_full_dataset.csv")

    # --- 1) Single MSAs (one CSV per MSA)
    msa_df = df[df["Geo"].apply(is_msa_geo)].copy()
    if len(msa_df) > 0:
        msa_out = OUT_DIR / "msas"
        for msa, g in msa_df.groupby("Geo"):
            fname = f"msa__{safe_name(msa)}.csv"
            write_csv(g, msa_out / fname)
        # Also save a combined MSA-only file
        write_csv(msa_df, OUT_DIR / "01_all_msas_combined.csv")

    # --- 2) Region groupings (Northeast/Midwest/South/West) for state rows only
    state_df = df[df["Geo"].apply(is_state_geo)].copy()
    if len(state_df) > 0:
        # map Geo -> region
        state_to_region = {st: r for r, states in REGIONS.items() for st in states}
        state_df["Region"] = state_df["Geo"].map(state_to_region)
        # Drop unknowns if any
        state_df_known = state_df.dropna(subset=["Region"]).copy()
        region_out = OUT_DIR / "regions"
        for region, g in state_df_known.groupby("Region"):
            fname = f"region__{safe_name(region)}.csv"
            write_csv(g, region_out / fname)
        write_csv(state_df_known, OUT_DIR / "02_regions_combined.csv")

    # --- 3) Energy states vs non-energy states (state rows)
    if len(state_df) > 0:
        energy_df = state_df[state_df["Geo"].isin(ENERGY_STATES)].copy()
        non_energy_df = state_df[~state_df["Geo"].isin(ENERGY_STATES)].copy()
        write_csv(energy_df, OUT_DIR / "03_energy_states.csv")
        write_csv(non_energy_df, OUT_DIR / "03_non_energy_states.csv")

    # --- 4) Sector level datasets
    # 4a) All sectors together (everything)
    write_csv(df, OUT_DIR / "04_all_sectors_together.csv")

    # 4b) One CSV per sector (excluding 'ALL' by default)
    sector_out = OUT_DIR / "sectors"
    sector_df = df.copy()
    # If you want to include Sector Name == 'ALL' in the per-sector outputs, delete the next line:
    sector_df = sector_df[sector_df["Sector Name"].astype(str).str.upper() != "ALL"].copy()

    for sector, g in sector_df.groupby("Sector Name"):
        fname = f"sector__{safe_name(sector)}.csv"
        write_csv(g, sector_out / fname)

    # --- 5) Leading-only subset
    # Keep ID cols + target (PCE normalized if present) + ONLY lag/MA features (+ interactions if you want)
    id_cols = [c for c in ["Period", "Geo", "Sector Name", "Sub-Sector Name"] if c in df.columns]

    # Choose a target if present
    # (You can change target_priority if you want a different target.)
    target_priority = [
        "PersonalConsumptionExpenditures_normalized",
        "PersonalConsumptionExpenditures_normalized_MoM_normalized",
        "PersonalConsumptionExpenditures_normalized_YoY_normalized",
    ]
    target_cols = [c for c in target_priority if c in df.columns]

    # Define leading features: lag and moving average engineered features
    leading_feature_cols = [
        c for c in df.columns
        if (
            re.search(r"_lag(1|2|3|4|5|6|9|12)\b", c)  # lags you might have
            or re.search(r"_MA(3|6|12)\b", c)          # moving averages you have
        )
    ]

    # Optional: keep interaction terms (often still "leading" if built from leading features)
    interaction_cols = [c for c in df.columns if c.endswith("_interaction")]

    leading_cols = id_cols + target_cols + sorted(set(leading_feature_cols + interaction_cols))
    leading_only = df[leading_cols].copy()
    write_csv(leading_only, OUT_DIR / "05_leading_only_subset.csv")

    # --- 6) Entertainment & recreation sector subset
    # Your original one-hot had Sector Name_Arts, Entertainment, and Recreation.
    # In your new long-format, this likely appears in "Sector Name" values.
    ent_mask = df["Sector Name"].astype(str).str.contains(
        r"Arts|Entertainment|Recreation", case=False, regex=True
    )
    entertainment_df = df[ent_mask].copy()
    write_csv(entertainment_df, OUT_DIR / "06_entertainment_and_recreation.csv")

    print(f"Done. Wrote splits to: {OUT_DIR}")


if __name__ == "__main__":
    main()
