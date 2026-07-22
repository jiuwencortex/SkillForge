from __future__ import annotations

import json
from pathlib import Path


class PersistentMixin:
    """Mixin for atomic tmp-file JSON persistence."""

    def _save_json(self, data: dict, path: Path) -> None:
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(data))
        tmp.rename(path)

    def _load_json(self, path: Path) -> dict | None:
        path = Path(path)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
