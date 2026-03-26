# src/data/panel.py
import pandas as pd
from typing import Optional
from src.data.store import read_master
from src.utils.config import config


def _get_fsbi_columns(master: pd.DataFrame,
                      geography: Optional[list],
                      sectors: Optional[list]) -> list:
    """
    Identify FSBI columns in master that match the given
    geography and sector filters.

    FSBI columns in master follow the naming convention:
    fsbi_<geography>_<sector>_<metric>

    Args:
        master: Master DataFrame
        geography: List of geography codes to include (None = all)
        sectors: List of sector names to include (None = all)
    Returns:
        List of matching column names
    """
    fsbi_cols = [c for c in master.columns if c.startswith("fsbi_")]

    if geography:
        fsbi_cols = [
            c for c in fsbi_cols
            if any(geo.lower() in c.lower() for geo in geography)
        ]
    if sectors:
        fsbi_cols = [
            c for c in fsbi_cols
            if any(sec.lower() in c.lower() for sec in sectors)
        ]
    return fsbi_cols


def build_panel(panel_name: str,
                master: pd.DataFrame = None) -> tuple[pd.Series, pd.DataFrame]:
    """
    Build a (y, X) panel from the master dataset using filter rules
    defined in config.yaml under data.panels.<panel_name>.

    Args:
        panel_name: Key in config["data"]["panels"]
        master: Optional pre-loaded master DataFrame;
                loads from disk if not provided
    Returns:
        Tuple of (y: PCE target series, X: feature DataFrame)
    """
    if master is None:
        master = read_master()

    panels_config = config["data"]["panels"]
    if panel_name not in panels_config:
        raise ValueError(
            f"Panel '{panel_name}' not found in config. "
            f"Available panels: {list(panels_config.keys())}"
        )

    panel_cfg  = panels_config[panel_name]
    geography  = panel_cfg.get("geography")
    sectors    = panel_cfg.get("sectors")
    target_col = config["data"]["target"]

    # Always include PCE and non-FSBI macro columns
    macro_cols = [c for c in master.columns if not c.startswith("fsbi_")]
    fsbi_cols  = _get_fsbi_columns(master, geography, sectors)

    all_cols = macro_cols + fsbi_cols
    panel_df = master[all_cols].copy()

    y = panel_df[target_col]
    X = panel_df.drop(columns=[target_col])

    print(f"[INFO] Panel '{panel_name}' built — "
          f"{X.shape[1]} features "
          f"({len(fsbi_cols)} FSBI, {len(macro_cols)-1} macro)")
    return y, X


def build_all_panels(master: pd.DataFrame = None) -> dict[str, tuple]:
    """
    Build all panels defined in config and return as a dict.
    Useful for running the same experiment across multiple panels.

    Returns:
        Dict of {panel_name: (y, X)}
    """
    if master is None:
        master = read_master()

    return {
        name: build_panel(name, master)
        for name in config["data"]["panels"]
    }
