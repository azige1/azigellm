"""第三阶段：生成「终态校验」pytest 文件（verifier）。

输入为第一阶段的题面 + truth（以及第二阶段的初始测试），输出
``test_final_state.py``：只有当任务被正确完成时才全部通过。
truth 会原样转给 LLM，使测试能断言精确的期望终态。
"""
from __future__ import annotations

import textwrap
from pathlib import Path
import sys
from typing import Optional

# 允许从任意工作目录以脚本方式运行
sys.path.insert(0, str(Path().resolve()))
from task_factory import extract_python_block, is_valid_python, batch_chat_completions

SYSTEM_MSG = """You are a senior Python engineer who writes robust pytest suites.
Write a robust pytest suite that validates the **FINAL** state of the operating-system / container **after** the student has
completed the task described.
Use the privileged *truth* data to assert the exact expected end state for the task to be completed.

Rules:
* The filename must be ``test_final_state.py`` (show it in a header comment).
* Use **only** the Python standard library and ``pytest`` (no third-party libs).
* Failures must clearly explain **what is still wrong**.
* When you check for files or directories, always use their *absolute* paths exactly as given (no relative paths).
* Ensure that the the state of the OS matches the truth after the task is completed.
* Write the code in a fenced code block that can be parsed to get a single python file.

Evaluation approach guidelines:
* Prefer semantic validation over exact string matching. For example:
  - Parse JSON/YAML and check keys/values instead of comparing raw strings.
  - Use `in` or regex to check for required content instead of exact line matching.
  - Check numeric values with tolerance (abs(actual - expected) < epsilon) instead of string comparison.
* For service-based tasks, test that services respond correctly:
  - Use subprocess to curl/wget endpoints and check response codes and content.
  - Check that ports are listening (socket connect or ss/netstat).
  - Verify processes are running (subprocess: pgrep, ps).
* For command-output tasks, run the command and check its exit code and stdout.
* For file tasks, validate structure and key content, not exact byte-for-byte match.
* Never check exact whitespace, trailing newlines, or formatting that the task didn't explicitly specify.
* For easy tasks, tests should check simple outcomes (file exists, content contains expected value).
* For hard tasks, tests can check complex multi-faceted state but should still validate semantics over exact formatting."""

USER_TEMPLATE = """The task description is: {task_description}
The truth value is: {truth}
Task difficulty: {difficulty}
The tests to check the initial container state, before the task is completed, are:
{initial_test_py}
Write the code in a fenced code block that can be parsed."""


def generate_final_tests_batch(
    items: list[tuple[str, ...]],
    *,
    model: str = "qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 128,
) -> list[Optional[str]]:
    """批量生成终态 pytest 文件。

    items 为 ``(task_description, truth, initial_test_py)`` 或
    ``(task_description, truth, initial_test_py, difficulty)``；
    返回与输入等长的列表，失败位置为 ``None``。
    """

    messages: list[list[dict[str, str]]] = []
    for item in items:
        task_description, truth, initial_test_py = item[0], item[1], item[2]
        difficulty = item[3] if len(item) > 3 else "medium"
        prompt = USER_TEMPLATE.format(
            task_description=task_description,
            truth=truth,
            initial_test_py=initial_test_py,
            difficulty=difficulty,
        )
        messages.append([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ])

    responses = batch_chat_completions(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    results: list[Optional[str]] = []
    for resp in responses:
        if resp is None:
            results.append(None)
            continue
        try:
            content = textwrap.dedent(resp.choices[0].message.content)
            parsed = extract_python_block(content)
            if is_valid_python(parsed):
                results.append(parsed)
            else:
                results.append(None)
        except Exception:
            results.append(None)
    return results
