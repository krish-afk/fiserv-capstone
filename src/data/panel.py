# src/data/panel.py
import re
import pandas as pd
from typing import Optional
from src.data.store import read_master, read_fsbi
from src.data.features import build_all_features
from src.data.transform import transform_target
from src.utils.config import config

# FSBI dimension columns — present in every row of the long-format DataFrame.
# All other columns are treated as numeric metric columns.
_FSBI_DIM_COLS = ["date", "Geo", "Sector Name", "Sub-Sector Name"]


def _sanitize(s: str) -> str:
    """
    Convert an arbitrary string to a safe, lowercase column-name fragment.
    Runs of non-alphanumeric characters become a single underscore.
    """
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _impute_fsbi_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the FSBI long-format DataFrame within each
    (Geo, Sector Name, Sub-Sector Name) group, preventing cross-geography
    contamination.

    Strategy (in order):
      1. Linear interpolation within group
      2. Forward-fill, then backward-fill within group (handles edges)
    """
    group_cols  = ["Geo", "Sector Name", "Sub-Sector Name"]
    metric_cols = [c for c in df.columns if c not in _FSBI_DIM_COLS]

    df = df.copy()
    df[metric_cols] = (
        df.groupby(group_cols)[metric_cols]
        .transform(lambda x: x.interpolate(method="linear", limit_direction="both"))
    )
    df[metric_cols] = (
        df.groupby(group_cols)[metric_cols]
        .transform(lambda x: x.ffill().bfill())
    )
    return df


def _pivot_fsbi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a (pre-filtered) FSBI long DataFrame to wide format, one row per date.

    Output column naming convention:
        fsbi_{safe_geo}_{safe_sector}_{safe_subsector}_{safe_metric}

    Steps:
      1. Group-level imputation within (Geo, Sector, Sub-Sector)
      2. pivot_table with (Geo, Sector Name, Sub-Sector Name) as column index
      3. Flatten the resulting MultiIndex columns
    """
    metric_cols = [c for c in df.columns if c not in _FSBI_DIM_COLS]

    df = _impute_fsbi_groups(df)

    wide = df.pivot_table(
        index="date",
        columns=["Geo", "Sector Name", "Sub-Sector Name"],
        values=metric_cols,
        aggfunc="first",
    )
    # MultiIndex column structure: (metric, Geo, Sector Name, Sub-Sector Name)
    wide.columns = [
        f"fsbi_{_sanitize(geo)}_{_sanitize(sec)}_{_sanitize(sub)}_{_sanitize(metric)}"
        for metric, geo, sec, sub in wide.columns
    ]
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    wide = wide.sort_index()
    return wide


def _filter_and_pivot_fsbi(
    fsbi_long: pd.DataFrame,
    geography: Optional[list],
    sectors: Optional[list],
    subsectors: Optional[list],
) -> pd.DataFrame:
    """
    Filter the FSBI long DataFrame to the panel's desired geography, sectors,
    and sub-sectors, then pivot to wide format.

    Geography filter:
        - None → filter to the national geography defined in
          config["data"]["sources"]["fsbi"]["national_geo"] (default "US").
          This keeps the pivot narrow for national panels.
        - list → filter rows where Geo is in the list (e.g. ["CA"]).

    Sectors filter:
        - None → include all sectors.
        - list → case-insensitive substring match on the "Sector Name" column
          (e.g. ["retail"] matches "Retail Trade").

    Subsectors filter:
        - None → include all sub-sectors.
        - list → case-insensitive substring match on the "Sub-Sector Name" column.

    Returns:
        Wide DataFrame indexed by date, or an empty DataFrame if no rows match.
    """
    df = fsbi_long.copy()

    # Geography filter
    if geography is None:
        national_geo = config["data"]["sources"]["fsbi"].get("national_geo", "US")
        df = df[df["Geo"] == national_geo]
    else:
        df = df[df["Geo"].isin(geography)]

    # Sector filter (substring match on Sector Name)
    if sectors is not None:
        pattern = "|".join(re.escape(s) for s in sectors)
        df = df[df["Sector Name"].str.contains(pattern, case=False, na=False)]

    # Sub-sector filter (substring match on Sub-Sector Name)
    if subsectors is not None:
        pattern = "|".join(re.escape(s) for s in subsectors)
        df = df[df["Sub-Sector Name"].str.contains(pattern, case=False, na=False)]

    if df.empty:
        print(f"[WARN] FSBI filter returned no rows "
              f"(geography={geography}, sectors={sectors}, subsectors={subsectors})"
              f" — no FSBI features for this panel")
        return pd.DataFrame()

    wide = _pivot_fsbi(df)
    return wide


def build_panel(
    panel_name: str,
    master: pd.DataFrame = None,
    fsbi_long: pd.DataFrame = None,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """
    Build a (y, X, y_level) panel for a named experiment config.

    Steps:
      1. Load master (BEA + FRED + USCB) and FSBI long data from disk if not passed.
      2. Filter FSBI rows by geography/sectors defined in the panel config.
      3. Pivot filtered FSBI to wide and left-join onto master.
      4. Run build_all_features() on the combined panel-specific frame.
      5. Extract target (with optional MoM/YoY transform) and feature matrix X.

    Feature engineering is intentionally done here rather than on a shared master
    so that the FSBI columns included in features reflect each panel's geographic
    and sector scope — national panels get national FSBI features, state panels
    get state FSBI features, etc.

    Args:
        panel_name: Key in config["data"]["panels"]
        master:     Optional pre-loaded master DataFrame (BEA + FRED + USCB).
                    Loads from disk if not provided.
        fsbi_long:  Optional pre-loaded FSBI long DataFrame.
                    Loads from disk if not provided.
    Returns:
        Tuple of:
          y:       Target series, transformed per target_transform config.
          X:       Feature DataFrame aligned to y's index.
          y_level: Raw (untransformed) target levels, used for inverse_transform
                   and level-space error metrics.
    """
    if master is None:
        master = read_master()
    if fsbi_long is None:
        fsbi_long = read_fsbi()

    panels_config = config["data"]["panels"]
    if panel_name not in panels_config:
        raise ValueError(
            f"Panel '{panel_name}' not found in config. "
            f"Available panels: {list(panels_config.keys())}"
        )

    panel_cfg        = panels_config[panel_name]
    geography        = panel_cfg.get("geography")
    sectors          = panel_cfg.get("sectors")
    subsectors       = panel_cfg.get("subsectors")
    target_col       = panel_cfg["target"]
    target_transform = panel_cfg.get("target_transform")

    # Filter and pivot FSBI for this panel; join onto master
    fsbi_wide = _filter_and_pivot_fsbi(fsbi_long, geography, sectors, subsectors)
    if not fsbi_wide.empty:
        panel_df = master.join(fsbi_wide, how="left")
    else:
        panel_df = master.copy()

    # Build panel-specific feature matrix (lags, rolling, growth rates)
    panel_df = build_all_features(panel_df)

    # Extract raw target level before transform — needed for inverse transform
    y_raw = panel_df[target_col]
    X     = panel_df.drop(columns=[target_col])

    # Remove other prediction targets from X — targets are never features
    other_targets = [
        c for c in config["data"]["targets"]
        if c in X.columns and c != target_col
    ]
    if other_targets:
        X = X.drop(columns=other_targets)

    # Apply target transform; trim X and y_level to match (shorter after pct_change)
    y = transform_target(y_raw, method=target_transform)
    y.name = f"{target_col}_{target_transform}" if target_transform else target_col

    X = X.loc[y.index].copy()
    y_level = y_raw.loc[y.index]

    # Add AR lags of the transformed target, e.g. pce_mom_lag1, pce_mom_lag2, pce_mom_lag3.
    # This is time-safe because it only uses prior target values.
    for lag in config["features"].get("target_lags", []):
        X[f"{y.name}_lag{lag}"] = y.shift(lag)

    fsbi_cols = [c for c in X.columns if c.startswith("fsbi_")]
    macro_feature_cols = [c for c in X.columns if not c.startswith("fsbi_")]
    print(f"[INFO] Panel '{panel_name}' built — "
          f"{len(y)} observations, {X.shape[1]} features "
          f"({len(fsbi_cols)} FSBI, {len(macro_feature_cols)} macro), "
          f"target: {y.name}")
    return y, X, y_level


def build_all_panels(
    master: pd.DataFrame = None,
    fsbi_long: pd.DataFrame = None,
) -> dict[str, tuple[pd.Series, pd.DataFrame, pd.Series]]:
    """
    Build all panels defined in config and return as a dict.
    Useful for running the same experiment across multiple panels.

    Returns:
        Dict of {panel_name: (y, X, y_level)}
    """
    if master is None:
        master = read_master()
    if fsbi_long is None:
        fsbi_long = read_fsbi()

    return {
        name: build_panel(name, master, fsbi_long)
        for name in config["data"]["panels"]
    }
