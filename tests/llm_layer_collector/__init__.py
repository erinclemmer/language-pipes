import os
import sys

# The project is not installed; make `language_pipes` importable from src/ for
# every module in this test package (matches the sys.path convention used by the
# other test suites in tests/).
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
