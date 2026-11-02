"""从大模型输出中抽取测试代码并渲染覆盖率标注。"""


def extract_fenced_code(output) -> str:
    outputlines = output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[0] + 1 : indexlines[1]])


def collect_test_sources(outputs) -> list[str]:
    outputs = [output[0] for output in outputs]

    code_blocks = []
    for output in outputs:
        code = extract_fenced_code(output)
        code_blocks.append(code)
    return code_blocks


def render_coverage_hints(fut) -> str:
    """按 1 起行号渲染被测函数源码，未覆盖分支行前插入标注注释。"""
    coverage = fut.coverage
    with open(fut.file_path, "r") as f:
        code = f.read()

    unexecuted_lines = coverage.get("unexecuted_lines", [])
    # JSON 往返后字典键为字符串，统一转成整数行号
    unevaluated_branches = {
        int(line): value for line, value in coverage.get("unevaluated_branches", {}).items()
    }
    start_line = coverage.get("start_line", 0)
    end_line = coverage.get("end_line", 0)

    def annotate(line_no, line):
        indentation = "".join(it for it in line[: len(line) - len(line.lstrip())])
        if line_no in unevaluated_branches:
            condition, destinations = unevaluated_branches[line_no]
            destination_text = ", ".join(str(d) for d in destinations)
            return f"{indentation}# UNCOVERED BRANCH: {condition} -> missing jump to line(s): {destination_text}\n{line}"
        if line_no in unexecuted_lines:
            return f"{indentation}# UNEXECUTED LINE\n{line}"
        return line

    annotated_lines = []
    for line_no, line in enumerate(code.split("\n"), start=1):
        if start_line <= line_no <= end_line:
            annotated_lines.append(annotate(line_no, line))

    return "\n".join(annotated_lines)
