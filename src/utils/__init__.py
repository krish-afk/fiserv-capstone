# src/utils/__init__.py
# Importing config here ensures load_dotenv() runs at package import time,
# before any other module in the project accesses os.environ for API keys.
from src.utils.config import config

__all__ = ["config"]
