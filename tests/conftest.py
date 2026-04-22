"""Shared fixtures for community module tests."""
import sys
from pathlib import Path

# Add app/ to the Python path so that `community`, `financial_brief_formatter`, etc.
# can be imported without installing a package.
APP_DIR = Path(__file__).resolve().parent.parent / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
