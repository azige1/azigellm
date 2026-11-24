import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .metrics import token_gap


def summarize_initial_kl(rows: list[dict]) -> dict[str, float]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["teacher"])].append(token_gap(row["student_logprobs"], row["teacher_logprobs"], row["response_mask"]))
    if not grouped:
        raise ValueError("第 0 步 KL 输入不能为空")
    return {name: float(np.mean(values)) for name, values in grouped.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = {"step": 0, "mean_token_kl": summarize_initial_kl(rows)}
    result["mean"] = float(np.mean(list(result["mean_token_kl"].values())))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
