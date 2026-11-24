"""事件流日志：按 stream-json 约定逐块写出 events.jsonl。"""

import json
from pathlib import Path
from typing import Any


class EventWriter:
    """向 events.jsonl 追加写事件；每条事件一行 JSON。"""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = []
