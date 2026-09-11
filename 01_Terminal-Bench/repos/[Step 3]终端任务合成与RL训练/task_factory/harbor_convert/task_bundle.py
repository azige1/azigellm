"""把 SIF/Apptainer 任务目录转换为 Harbor 任务格式。

输入：包含 ``task.json`` / ``test_final_state.py`` / ``container.def`` 的任务目录。
输出：Harbor 目录（``instruction.md``、``environment/Dockerfile``、
``tests/test_final_state.py``、``tests/test.sh``）。
"""
from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

from task_factory.harbor_convert.sif_to_docker import def_to_dockerfile


def read_task_json(task_dir: Path) -> Optional[Dict[str, Any]]:
    """读取任务目录下的 task.json；失败返回 None。"""
    task_json = task_dir / "task.json"
    if not task_json.exists():
        return None
    try:
        return json.loads(task_json.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def render_instruction_md(task_data: Dict[str, Any]) -> str:
    """根据任务元数据生成 instruction.md 内容。"""
    description = (
        task_data.get("task_description")
        or task_data.get("description")
        or "No description available"
    )
    task_id = task_data.get("task_id")
    lines = ["# Task", "", description, ""]
    if task_id:
        lines.append(f"<!-- Task ID: {task_id} -->")
    return "\n".join(lines)


def find_task_dirs(tasks_dir: Path) -> List[Path]:
    """列出所有合法任务目录（含 task.json 且目录名含 'task'），按名称排序。"""
    if not tasks_dir.exists():
        return []
    return sorted(
        d for d in tasks_dir.iterdir()
        if d.is_dir() and "task" in d.name and (d / "task.json").exists()
    )


def convert_task(
    task_dir: Path,
    output_dir: Path,
    reuse_dockerfile: bool = False,
    model: str = "gpt-5.1",
    provider: str = "openai",
) -> Dict[str, Any]:
    """把单个 SIF 任务目录转换为 Harbor 格式。

    参数:
        task_dir: 源任务目录（须含 task.json、test_final_state.py、container.def）。
        output_dir: 输出根目录，结果写入 ``output_dir/task_dir.name/``。
        reuse_dockerfile: Dockerfile 已存在时跳过 LLM 转换。
        model: def→Dockerfile 转换所用模型。
        provider: LLM 提供方（``openai`` 或 ``anthropic``）。

    返回:
        ``{"success": bool, "error": str | None}``。
    """
    required = [
        task_dir / "task.json",
        task_dir / "test_final_state.py",
        task_dir / "container.def",
    ]
    missing = [str(f) for f in required if not f.exists()]
    if missing:
        return {"success": False, "error": f"Missing required files: {missing}"}

    task_data = read_task_json(task_dir)
    if task_data is None:
        return {"success": False, "error": "Failed to load task.json"}

    harbor_dir = output_dir / task_dir.name
    env_dir = harbor_dir / "environment"
    tests_dir = harbor_dir / "tests"

    harbor_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    (harbor_dir / "instruction.md").write_text(
        render_instruction_md(task_data), encoding="utf-8"
    )

    # Dockerfile：已有则复用，否则调 LLM 转换
    dockerfile_path = env_dir / "Dockerfile"
    if reuse_dockerfile and dockerfile_path.exists():
        pass
    else:
        def_content = (task_dir / "container.def").read_text(encoding="utf-8")
        dockerfile_content = def_to_dockerfile(
            def_content, model=model, provider=provider
        )
        if not dockerfile_content:
            return {"success": False, "error": "Dockerfile conversion failed"}
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

    test_src = (task_dir / "test_final_state.py").read_text(encoding="utf-8")
    (tests_dir / "test_final_state.py").write_text(test_src, encoding="utf-8")

    test_sh = tests_dir / "test.sh"
    test_sh.write_text(
        "#!/bin/bash\nset -e\npython3 -m pytest /tests/test_final_state.py -v\n",
        encoding="utf-8",
    )
    test_sh.chmod(test_sh.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return {"success": True, "error": None}
