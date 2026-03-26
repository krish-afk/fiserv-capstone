# src/utils/config.py
import yaml
from pathlib import Path

# TODO: Consider making this an explicit argument rather than 
# auto-discovering, if you want to support multiple config files
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and return the project config as a dictionary."""
    # TODO: Add validation here (e.g. required keys, path existence checks)
    with open(path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
