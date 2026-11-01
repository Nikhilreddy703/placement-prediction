"""Main application entry point for Streamlit.
Imports and executes app.app to maintain clean root-level structure.
"""
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Execute the application
import runpy

if __name__ == "__main__":
    app_path = BASE_DIR / "app" / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")
else:
    app_path = BASE_DIR / "app" / "app.py"
    runpy.run_path(str(app_path), run_name="__main__")

