"""结构化反馈渲染：执行报错反馈与覆盖率不足反馈。"""

from codeharvest.synthesis.tests.parsing import render_coverage_hints

TRACEBACK_TAIL_LINES = 15


def _test_source_lines(fut, test_id: str | None) -> dict[int, str]:
    """取某轮测试源码的 1 起行号映射，用于反馈中附出错行源码。"""
    if test_id is None:
        return {}
    test_source = fut.tests.get(test_id, "")
    return {i: line for i, line in enumerate(test_source.split("\n"), start=1)}


def render_error_feedback(fut) -> str:
    """把 run_tests_errors 渲染为结构化反馈文本（英文，与提示词语言一致）。

    每条失败给出测试方法名、异常类型与消息、出错行号与该行源码、traceback 末 15 行。
    """
    exec_stats = fut.exec_stats or {}
    run_tests_errors = exec_stats.get("run_tests_errors", {})

    blocks = []
    for test_id in sorted(run_tests_errors.keys(), key=str):
        lines_by_no = _test_source_lines(fut, test_id)
        for error in run_tests_errors[test_id]:
            message = error.get("message", "")
            message_lines = message.split("\n")
            tail = "\n".join(message_lines[-TRACEBACK_TAIL_LINES:])

            # 从 traceback 首行提取异常类型与消息
            # traceback 最后一个非空行是「异常类型: 消息」
            exc_header = next(
                (line.strip() for line in reversed(message_lines) if line.strip()), ""
            )

            line_no = error.get("line")
            line_source = lines_by_no.get(line_no, "") if line_no else ""

            block = [
                f"Test method: {error.get('test', '')}",
                f"Exception: {exc_header}",
                f"Failing line: {line_no}",
                f"Failing line source: {line_source}",
                f"Traceback (last lines):\n{tail}",
            ]
            if error.get("defect_reason"):
                block.append(f"Defect reason: {error['defect_reason']}")
            blocks.append("\n".join(block))

    if not blocks:
        generic_error = exec_stats.get("error", "") if exec_stats else ""
        return f"Execution failed:\n{generic_error}"

    return "The following test methods failed:\n\n" + "\n\n".join(blocks)


def render_coverage_feedback(fut) -> str:
    """覆盖率反馈：头部给出已覆盖/总分支数，正文为带 UNCOVERED BRANCH 标注的源码。"""
    coverage = fut.coverage or {}
    covered = coverage.get("covered_branches", 0)
    total = coverage.get("total_branches", 0)
    header = f"Covered branches: {covered} / {total}\n\n"
    return header + render_coverage_hints(fut)
