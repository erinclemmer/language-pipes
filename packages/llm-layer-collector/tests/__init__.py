import os
import sys

# Make `llm_layer_collector` importable from src/ without requiring an install,
# so the suite runs the in-tree copy when this package is not pip-installed.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
