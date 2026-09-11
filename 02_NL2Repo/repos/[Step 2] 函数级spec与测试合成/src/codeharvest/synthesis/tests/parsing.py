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
    coverage = fut.coverage
    with open(fut.file_path, "r") as f:
        code = f.read()

    unexecuted_lines = coverage.get("unexecuted_lines", [])
    unevaluated_branches = [
        int(b) for b in coverage.get("unevaluated_branches", {}).keys()
    ]
    start_line = coverage.get("start_line", 0)
    end_line = coverage.get("end_line", 0)

    def annotate(i, line):
        indentation = "".join(it for it in line[: len(line) - len(line.lstrip())])
        
        
        
        if i in unevaluated_branches:
            return f"{indentation}# UNEVALUATED BRANCH\n{line}"
        return line

    annotated_lines = []
    for i, line in enumerate(code.split("\n")):
        if i in range(start_line, end_line + 1):
            annotated_lines.append(annotate(i, line))

    return "\n".join(annotated_lines)
