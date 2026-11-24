import argparse
import json
import re
from pathlib import Path
from typing import Any

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def normalized_tokens(value: Any) -> list[str]:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return _TOKEN.findall(value.casefold())


def ngrams(value: Any, size: int = 13) -> set[tuple[str, ...]]:
    tokens = normalized_tokens(value)
    return {tuple(tokens[index : index + size]) for index in range(max(0, len(tokens) - size + 1))}


def template_entity_keys(record: dict[str, Any]) -> set[str]:
    keys = set()
    template = record.get("template_id", record.get("task_type"))
    entities = record.get("entities", [])
    if template:
        keys.add(f"template:{template}")
    for entity in entities:
        keys.add(f"entity:{normalized_tokens(entity) and ' '.join(normalized_tokens(entity))}")
    if template and entities:
        keys.add(f"pair:{template}:{'|'.join(sorted(' '.join(normalized_tokens(value)) for value in entities))}")
    return keys


def decontaminate(records: list[dict], evaluation_records: list[dict], size: int = 13) -> tuple[list[dict], list[dict]]:
    blocked_ngrams = set()
    blocked_keys = set()
    for row in evaluation_records:
        blocked_ngrams |= ngrams(row, size)
        blocked_keys |= template_entity_keys(row)
    kept, removed = [], []
    for row in records:
        overlap = bool(ngrams(row, size) & blocked_ngrams or template_entity_keys(row) & blocked_keys)
        (removed if overlap else kept).append(row)
    return kept, removed


def _read(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        import pandas as pd

        pd.DataFrame(rows).to_parquet(path, index=False)
    else:
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removed", type=Path, required=True)
    args = parser.parse_args()
    evaluation = [row for path in args.evaluation for row in _read(path)]
    kept, removed = decontaminate(_read(args.train), evaluation)
    _write(args.output, kept)
    _write(args.removed, removed)


if __name__ == "__main__":
    main()
