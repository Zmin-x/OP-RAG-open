from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from op_rag.herb_dashboard import run_server  # noqa: E402


if __name__ == "__main__":
    run_server()
