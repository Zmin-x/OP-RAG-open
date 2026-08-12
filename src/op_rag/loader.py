from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import DATA_DIR


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_kb(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    """Load the three-level demo knowledge base from JSON files."""
    return {
        "syndromes": load_json(data_dir / "syndromes.json"),
        "formulas": load_json(data_dir / "formulas.json"),
        "herbs": load_json(data_dir / "herbs.json"),
        "syndrome_formula_map": load_json(data_dir / "syndrome_formula_map.json"),
    }
