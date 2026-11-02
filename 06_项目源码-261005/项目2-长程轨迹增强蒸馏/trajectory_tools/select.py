"""筛选：剔除答案获取轨迹、列出环境类重投清单、选出成功轨迹。"""

import json
from pathlib import Path
from typing import Any


def select(
    index_path: Path,
    attribution: dict[str, list[str]],
    out_path: Path,
    pass_rate_threshold: float = 0.9,
    max_blind_threshold: int = 8,
) -> dict[str, Any]:
    """按说明口径筛选：返回 {selected, rerun, dropped_answer_access} 计数。"""
    entries = [
        json.loads(line)
        for line in Path(index_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = []
    rerun = []
    dropped = 0
    for entry in entries:
        tid = entry["id"]
        modes = attribution.get(tid, [])
        if "answer_access" in modes:
            dropped += 1
            continue
        rate = entry.get("hidden_pass_rate")
        # 没有判分结果（环境安装失败、判分超时）的轨迹与模型行为无关，列入重投清单；
        # 由模型改动导致的包结构失败仍按通过率处理，不重投
        if rate is None:
            rerun.append(tid)
            continue
        if (
            rate is not None
            and rate >= pass_rate_threshold
            and entry.get("max_blind", 0) <= max_blind_threshold
        ):
            selected.append(entry)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for entry in selected:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    rerun_path = out.parent / "rerun.txt"
    rerun_path.write_text("\n".join(rerun) + ("\n" if rerun else ""), encoding="utf-8")
    return {
        "selected": len(selected),
        "rerun": len(rerun),
        "dropped_answer_access": dropped,
        "selected_path": str(out),
        "rerun_path": str(rerun_path),
    }


__all__ = []
