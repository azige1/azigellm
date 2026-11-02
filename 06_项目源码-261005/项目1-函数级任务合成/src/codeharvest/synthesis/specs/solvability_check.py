"""可解性检查：独立小模型只读「签名 + 规格」实现函数，采样多次跑隐藏测试。"""

import ast
import json
import random

from pydantic import Field

from codeharvest.llm.config import LLMConfig
from codeharvest.execution.config import ExecutionConfig
from codeharvest.execution.equivalence import check_code_equivalence
from codeharvest.llm.batch import CompletionEngine
from codeharvest.synthesis.specs.task import SpecTask
from codeharvest.synthesis.specs.parsing import collect_refined_specs, extract_code_reply
from codeharvest.utils.io import read_json_data, read_test_targets
from codeharvest.workspace import EXECUTION_DIR


class SolvabilityConfig(LLMConfig, ExecutionConfig):
    model_name: str = Field(
        "Qwen/Qwen3-8B",
        description="用于可解性检查的独立小模型",
    )
    base_url: str | None = Field(
        "http://127.0.0.1:8001/v1",
        description="可解性检查模型的服务地址",
    )
    temperature: float = Field(
        0.8,
        description="可解性检查采样的温度",
    )
    max_tokens: int = Field(
        2048,
        description="单次实现的最大生成长度",
    )
    rewrite_model_name: str = Field(
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        description="退回重写规格时使用的合成模型",
    )
    rewrite_base_url: str | None = Field(
        "http://127.0.0.1:8000/v1",
        description="合成模型的服务地址",
    )
    n: int = Field(
        4,
        description="每个规格的采样实现条数",
    )
    in_file: str = Field(
        None,
        description="规格任务输入文件，缺省 {exp_id}_specgen.json",
    )
    exp_id: str = Field(
        "temp",
        description="实验 ID",
    )


def _signature_with_spec(code: str, spec: str) -> str:
    """只暴露签名 + 规格：去掉函数体，docstring 换成规格文本。"""
    tree = ast.parse(code)
    node = tree.body[0]
    if isinstance(node, ast.ClassDef):
        methods = [
            m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if methods:
            node.body = [copy_signature_node(methods[0], spec)]
        return ast.unparse(node)
    return ast.unparse(copy_signature_node(node, spec))


def copy_signature_node(node, spec: str):
    import copy as _copy

    new_node = _copy.deepcopy(node)
    new_node.body = [ast.Expr(value=ast.Constant(value=spec))]
    return new_node


def _build_solvability_payload(fut, spec: str) -> list[dict]:
    from codeharvest.synthesis.specs.prompts import (
        SOLVABILITY_SYSTEM_MESSAGE,
        SOLVABILITY_TASK_MESSAGE,
    )

    signature_spec = _signature_with_spec(fut.code, spec)
    return [
        {"role": "system", "content": SOLVABILITY_SYSTEM_MESSAGE},
        {
            "role": "user",
            "content": SOLVABILITY_TASK_MESSAGE.format(
                code_snipppet=signature_spec, function_name=fut.name
            ),
        },
    ]


def _record_id(record: dict) -> str | None:
    raw = record.get("function_id") or record.get("method_id")
    if isinstance(raw, dict):
        return raw.get("identifier")
    return raw


def _collect_failure_summary(check_results: list) -> str:
    """汇总各次采样失败的报错摘要（不附测试源码）。"""
    summaries = []
    for idx, (_, error, run_logs) in enumerate(check_results):
        if isinstance(run_logs, dict) and "error" in run_logs:
            summaries.append(f"采样 {idx}: {run_logs['error'][:300]}")
        elif error:
            summaries.append(f"采样 {idx}: {str(error)[:300]}")
        else:
            for test_id, log in (run_logs or {}).items():
                if not log.get("valid", True):
                    failed = log.get("failed_names", []) + log.get("errored_names", [])
                    summaries.append(f"采样 {idx} {test_id}: 未通过 {failed}")
    return "\n".join(summaries)[:2000]


class SolvabilityChecker:

    @staticmethod
    def load_tasks(args: SolvabilityConfig):
        in_file = args.in_file or f"{args.exp_id}_specgen.json"
        file_path = EXECUTION_DIR / in_file
        records = read_json_data(file_path)
        futs = {f.id: f for f in read_test_targets(file_path)}
        return records, futs

    @staticmethod
    def check_candidates(args, fut, implementations: list[str]) -> tuple[int, list]:
        """逐候选用 codegen 模式跑隐藏测试，返回通过次数与逐次结果。"""
        passed = 0
        results = []
        for code in implementations:
            port = random.randint(3000, 10000)
            try:
                all_valid, error, run_logs = check_code_equivalence(
                    code, fut, port, local=args.local, image=args.image
                )
            except Exception as e:
                all_valid, error, run_logs = False, repr(e), {"error": repr(e)}
            if all_valid:
                passed += 1
            results.append((all_valid, error, run_logs))
        return passed, results

    @staticmethod
    def run(args: SolvabilityConfig) -> None:
        records, futs_by_id = SolvabilityChecker.load_tasks(args)

        output_records = []
        stats = {"total": 0, "ok": 0, "rewritten": 0, "dropped_unsolvable": 0}

        for task_record in records:
            record_id = _record_id(task_record)
            fut = futs_by_id.get(record_id)
            task_id = record_id or "unknown"
            spec = task_record.get("spec", "")
            stats["total"] += 1

            entry = {
                "id": task_id,
                "spec": spec,
                "solve_rate": 0.0,
                "status": "dropped_unsolvable",
            }

            if fut is None or not spec:
                output_records.append(entry)
                stats["dropped_unsolvable"] += 1
                continue

            payload = [_build_solvability_payload(fut, spec)]
            outputs = CompletionEngine.run_chat_completions(args, payload)
            implementations = [
                extract_code_reply(output[0]) for output in outputs
            ]
            implementations = [c for c in implementations if c.strip()]

            passed, results = SolvabilityChecker.check_candidates(args, fut, implementations)
            entry["solve_rate"] = passed / args.n if args.n else 0.0

            if passed > 0:
                entry["status"] = "ok"
                output_records.append(entry)
                stats["ok"] += 1
                continue

            # 全失败：附上失败摘要重写一次规格，再查一次
            failure_summary = _collect_failure_summary(results)
            rewritten_spec = SolvabilityChecker.rewrite_spec(args, fut, spec, failure_summary)
            if rewritten_spec:
                entry["spec"] = rewritten_spec
                payload = [_build_solvability_payload(fut, rewritten_spec)]
                outputs = CompletionEngine.run_chat_completions(args, payload)
                implementations = [
                    extract_code_reply(output[0]) for output in outputs
                ]
                implementations = [c for c in implementations if c.strip()]
                passed, _ = SolvabilityChecker.check_candidates(args, fut, implementations)
                entry["solve_rate"] = passed / args.n if args.n else 0.0

            if entry["solve_rate"] > 0:
                entry["status"] = "rewritten"
                stats["rewritten"] += 1
            else:
                entry["status"] = "dropped_unsolvable"
                stats["dropped_unsolvable"] += 1
            output_records.append(entry)

        out_file = EXECUTION_DIR / f"{args.exp_id}_solvability.json"
        with open(out_file, "w") as f:
            json.dump(
                {"stats": stats, "records": output_records},
                f,
                indent=2,
            )
        print(f"可解性检查完成: {out_file}")
        print(json.dumps(stats, indent=2))

    @staticmethod
    def rewrite_spec(args, fut, spec: str, failure_summary: str) -> str:
        """把失败摘要反馈给规格合成模型重写一次（不附测试源码）。"""
        task = SpecTask(func_meth=fut, generated_test="")
        messages = list(task.chat_messages)
        messages.append(
            {
                "role": "user",
                "content": (
                    "The specification above was not solvable: independent implementations failed the hidden tests.\n"
                    "Failure summary (no test source shown):\n"
                    f"{failure_summary}\n"
                    "Rewrite the specification to remove ambiguity. Return only the updated function with docstring enclosed in markdown."
                ),
            }
        )
        payload_args = args.model_copy(
            update={
                "n": 1,
                "temperature": 0.3,
                "model_name": args.rewrite_model_name,
                "base_url": args.rewrite_base_url,
            }
        )
        outputs = CompletionEngine.run_chat_completions(payload_args, [messages])
        rewritten = collect_refined_specs(outputs)
        return rewritten[0] if rewritten and rewritten[0] else ""
