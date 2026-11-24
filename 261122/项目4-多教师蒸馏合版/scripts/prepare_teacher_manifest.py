import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # GPU 4-7（Ray 本地映射 0-3），每卡常驻两个 Qwen3-8B 教师，共八个
    variables = {
        "code_gen": ("CODE_GEN_TEACHER_MODEL", 0),
        "repo_impl": ("REPO_IMPL_TEACHER_MODEL", 0),
        "issue_fix": ("ISSUE_FIX_TEACHER_MODEL", 1),
        "test_gen": ("TEST_GEN_TEACHER_MODEL", 1),
        "math_science": ("MATH_SCIENCE_TEACHER_MODEL", 2),
        "instruction_following": ("INSTRUCTION_FOLLOWING_TEACHER_MODEL", 2),
        "tool_calling": ("TOOL_CALLING_TEACHER_MODEL", 3),
        "general_chat": ("GENERAL_CHAT_TEACHER_MODEL", 3),
    }
    manifest = {}
    for teacher_id, (variable, gpu) in variables.items():
        path = os.environ.get(variable)
        if not path:
            raise RuntimeError(f"缺少环境变量: {variable}")
        manifest[teacher_id] = {"path": path, "version": path, "gpu": gpu}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
