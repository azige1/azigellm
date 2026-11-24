"""开训前 pass@8 筛选与阶段二题池定期清理脚本（3.1 节）。

使用 vLLM 离线推理对每道候选题目采样 8 次，再调用项目一（函数级）或项目二（仓库级）
判分接口，只保留通过次数在 1/8 至 6/8 之间的题目，降低开训时零方差组占比。
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

import typer
from transformers import AutoTokenizer

from long_horizon_rl.curriculum import (
    filter_tasks_by_pass_at_8,
    load_jsonl,
    prune_fully_solved_tasks,
    save_jsonl,
)
from long_horizon_rl.data_prep import build_function_prompt_messages, build_repo_prompt_messages
from long_horizon_rl.reward_eval import evaluate_function_task, evaluate_repo_workspace

app = typer.Typer(no_args_is_help=True, add_completion=False, help="pass@8 题目难度筛选与题池清理")


def _generate_n_vllm(
    llm: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> list[str]:
    from vllm import SamplingParams

    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outputs = llm.generate([prompt_text], sampling_params=params)
    return [cand.text for cand in outputs[0].outputs]


def _extract_code(reply: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", reply, flags=re.DOTALL)
    return blocks[-1].strip() if blocks else reply.strip()


def _evaluate_repo_candidates(
    task: dict[str, Any],
    *,
    model_path: str,
    n: int,
    max_turns: int,
    compact_at: int,
    base_url: str,
) -> list[float]:
    """对仓库级任务执行 n 次多轮 rollout 并用项目二判分返回各次通过率。"""
    from openai import OpenAI

    from agent_rollout.run_loop import run_rollout

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    rates: list[float] = []
    with tempfile.TemporaryDirectory(prefix="lhrl-pass8-") as tmp:
        for idx in range(n):
            res = run_rollout(
                task=task,
                model=model_path,
                client=client,
                out_dir=Path(tmp) / f"run_{idx}",
                max_turns=max_turns,
                compact_at=compact_at,
                keep_workspace=False,
            )
            score_file = Path(res["out_dir"]) / "score.json"
            if score_file.is_file():
                score = json.loads(score_file.read_text(encoding="utf-8"))
                rates.append(float(score.get("hidden_pass_rate", 0.0)))
            else:
                rates.append(0.0)
    return rates


@app.command()
def filter_pool(
    stage: str = typer.Option(..., "--stage", help="function 或 repo"),
    input_jsonl: Path = typer.Option(..., "--input", help="输入任务 JSONL"),
    output_jsonl: Path = typer.Option(..., "--output", help="筛选后输出 JSONL"),
    report_json: Optional[Path] = typer.Option(None, "--report", help="pass@8 明细报告路径"),
    model_path: str = typer.Option(..., "--model", help="当前 checkpoint 路径"),
    n: int = typer.Option(8, "--n", help="每题采样次数"),
    min_passed: int = typer.Option(1, "--min-passed", help="保留最少通过次数（默认 1/8）"),
    max_passed: int = typer.Option(6, "--max-passed", help="保留最多通过次数（默认 6/8）"),
    temperature: float = typer.Option(0.8, "--temperature"),
    top_p: float = typer.Option(0.95, "--top-p"),
    max_tokens: int = typer.Option(2048, "--max-tokens"),
    tensor_parallel_size: int = typer.Option(1, "--tp"),
    max_model_len: int = typer.Option(32768, "--max-model-len"),
    repo_base_url: str = typer.Option("http://127.0.0.1:8000/v1", "--repo-base-url", help="仓库级多轮服务地址"),
    repo_max_turns: int = typer.Option(60, "--repo-max-turns"),
    compact_at: int = typer.Option(27000, "--compact-at"),
    prune_solved_only: bool = typer.Option(False, "--prune-solved-only", help="仅移除全部通过的题目"),
) -> None:
    """用当前 checkpoint 测 pass@8 并只保留 1/8 至 6/8 的题目。"""
    tasks = load_jsonl(input_jsonl)

    if stage == "function":
        from vllm import LLM

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_model_len)

        def eval_fn(task: dict[str, Any]) -> list[float]:
            msgs = build_function_prompt_messages(task)
            replies = _generate_n_vllm(
                llm,
                tokenizer,
                msgs,
                n=n,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            rates = []
            for reply in replies:
                outcome = evaluate_function_task(task, _extract_code(reply), enable_tamper_scan=False)
                rates.append(outcome.pass_rate)
            return rates

    elif stage == "repo":

        def eval_fn(task: dict[str, Any]) -> list[float]:
            return _evaluate_repo_candidates(
                task,
                model_path=model_path,
                n=n,
                max_turns=repo_max_turns,
                compact_at=compact_at,
                base_url=repo_base_url,
            )

    else:
        raise typer.BadParameter(f"stage 必须为 function 或 repo，收到: {stage}")

    if prune_solved_only:
        kept, removed = prune_fully_solved_tasks(tasks, lambda t: max(eval_fn(t) or [0.0]))
        save_jsonl(output_jsonl, kept)
        typer.echo(f"清理完成：保留 {len(kept)}/{len(tasks)}，移除全对题 {len(removed)}")
        return

    kept, report = filter_tasks_by_pass_at_8(
        tasks,
        eval_fn,
        num_samples=n,
        min_passed=min_passed,
        max_passed=max_passed,
    )
    save_jsonl(output_jsonl, kept)
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(
            json.dumps([r.__dict__ for r in report], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    typer.echo(f"[{stage}] pass@{n} 筛选完成：保留 {len(kept)}/{len(tasks)}（区间 {min_passed}/{n}–{max_passed}/{n}）")


if __name__ == "__main__":
    app()
