# src/data/__init__.py
# Public API for the data package.
# Downstream code can import from src.data directly:
#   from src.data import run_ingestion, run_cleaning, build_panel

from src.data.ingest import run_ingestion
from src.data.load import load_all_raw
from src.data.clean import run_cleaning
from src.data.store import read_master, write_master, read_fsbi, write_fsbi
from src.data.features import build_all_features, build_named_subset
from src.data.transform import (
    transform_target,
    inverse_transform_target,
    scale_features,
    difference_series,
)
from src.data.panel import build_panel, build_all_panels

__all__ = [
    # Ingestion
    "run_ingestion",
    # Loading
    "load_all_raw",
    # Cleaning
    "run_cleaning",
    # Storage
    "read_master",
    "write_master",
    "read_fsbi",
    "write_fsbi",
    # Feature engineering
    "build_all_features",
    "build_named_subset",
    # Transforms
    "transform_target",
    "inverse_transform_target",
    "scale_features",
    "difference_series",
    # Panels
    "build_panel",
    "build_all_panels",
]
