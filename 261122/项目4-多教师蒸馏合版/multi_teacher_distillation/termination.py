import argparse
import json
from pathlib import Path
from typing import Any


def terminal_token_ids(tokenizer: Any) -> set[int]:
    values = getattr(tokenizer, "eos_token_id", None)
    if values is None:
        values = []
    elif not isinstance(values, (list, tuple)):
        values = [values]
    ids = {int(value) for value in values if value is not None}
    for token in ("<|im_end|>", "<|endoftext|>"):
        value = tokenizer.convert_tokens_to_ids(token)
        if value is not None and value != getattr(tokenizer, "unk_token_id", None):
            ids.add(int(value))
    return ids


def assistant_suffix_ids(tokenizer: Any) -> list[int]:
    marker = "terminal-check-7f3a"
    messages = [{"role": "user", "content": "检查"}, {"role": "assistant", "content": marker}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    prefix = rendered.split(marker, 1)[0] + marker
    full_ids = tokenizer.encode(rendered, add_special_tokens=False)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise RuntimeError("assistant 模板前后缀 token 化不一致")
    return full_ids[len(prefix_ids) :]


def compare_templates(student: Any, teachers: dict[str, Any]) -> dict[str, Any]:
    student_ids = assistant_suffix_ids(student)
    student_set = terminal_token_ids(student)
    rows = {}
    for name, tokenizer in teachers.items():
        teacher_ids = assistant_suffix_ids(tokenizer)
        rows[name] = {
            "student_suffix": student_ids,
            "teacher_suffix": teacher_ids,
            "terminal_ids_match": student_set == terminal_token_ids(tokenizer),
            "assistant_suffix_match": student_ids == teacher_ids,
        }
    return {
        "teachers": rows,
        "passed": all(
            value["terminal_ids_match"] and value["assistant_suffix_match"] for value in rows.values()
        ),
    }


def terminal_logprob(model: Any, tokenizer: Any, record: dict[str, Any], terminal_id: int) -> float:
    import torch
    import torch.nn.functional as F

    messages = [*record["prompt"], {"role": "assistant", "content": record["answer"]}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    ids = tokenizer.encode(rendered, add_special_tokens=False)
    positions = [index for index, value in enumerate(ids) if value == terminal_id]
    if not positions or positions[-1] == 0:
        raise RuntimeError("固定回答中找不到学生终止 token")
    position = positions[-1]
    tensors = torch.tensor([ids], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=tensors).logits[0, position - 1].float()
        return float(F.log_softmax(logits, dim=-1)[terminal_id].item())


def check_fixed_prompts(student_path: str, teacher_paths: dict[str, str], records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    terminal_id = assistant_suffix_ids(student_tokenizer)[0]
    student = AutoModelForCausalLM.from_pretrained(student_path, dtype=torch.bfloat16, device_map="auto").eval()
    student_scores = [terminal_logprob(student, student_tokenizer, row, terminal_id) for row in records]
    del student
    torch.cuda.empty_cache()
    checks = []
    for name, path in teacher_paths.items():
        tokenizer = AutoTokenizer.from_pretrained(path)
        teacher = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto").eval()
        for index, row in enumerate(records):
            teacher_score = terminal_logprob(teacher, tokenizer, row, terminal_id)
            checks.append({
                "teacher": name, "prompt_id": str(row.get("prompt_id", index)),
                "student_logprob": student_scores[index], "teacher_logprob": teacher_score,
                "gap": abs(student_scores[index] - teacher_score),
            })
        del teacher
        torch.cuda.empty_cache()
    return {"logprob_checks": checks, "passed": all(row["gap"] <= 2.0 for row in checks)}


def require_termination_alignment(report: dict[str, Any], maximum_gap: float = 2.0) -> None:
    if not report.get("templates", {}).get("passed", False):
        raise RuntimeError("学生与教师的 assistant 终止 token 或模板后缀不一致")
    for row in report.get("logprob_checks", []):
        if abs(float(row["gap"])) > maximum_gap:
            raise RuntimeError(f"教师 {row['teacher']} 在终止 token 上的 log-prob 差值超过 2")


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True)
    parser.add_argument("--teacher", action="append", required=True, help="domain=path")
    parser.add_argument("--fixed-prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    teacher_paths = dict(value.split("=", 1) for value in args.teacher)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    teachers = {name: AutoTokenizer.from_pretrained(path) for name, path in teacher_paths.items()}
    records = [json.loads(line) for line in args.fixed_prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("固定 prompt 集不能为空")
    report = {"templates": compare_templates(student_tokenizer, teachers)}
    require_termination_alignment(report)
    report.update(check_fixed_prompts(args.student, teacher_paths, records))
    report["passed"] = report["templates"]["passed"] and report["passed"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    require_termination_alignment(report)


if __name__ == "__main__":
    main()
