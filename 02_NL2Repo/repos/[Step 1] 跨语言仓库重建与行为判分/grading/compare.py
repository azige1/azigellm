"""输出比对的公共函数。

两套任务共用同一套「逐行比对」语义：
- 每行去掉所有空白字符后再比较，空行忽略；
- 两端行数一致且逐行相等才算该用例通过；
- cpp2rust 额外允许「整段输出是同一个浮点数」的兜底相等。
"""


def normalize_lines(stdout):
    """stdout -> 去空白后的非空行列表。"""
    return [
        "".join(line.split())
        for line in stdout.strip().splitlines()
        if line.strip()
    ]


def count_matched_lines(ref_lines, gen_lines):
    """逐行相等计数；行数不同或为空则视为不可用，返回 None。"""
    if not ref_lines or len(ref_lines) != len(gen_lines):
        return None
    return sum(1 for a, b in zip(ref_lines, gen_lines) if a == b), len(ref_lines)


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def outputs_equal(out_a, out_b):
    """整段输出相等判定：先按字符串，再按浮点数兜底。"""
    s_a, s_b = out_a.strip(), out_b.strip()
    if s_a == s_b:
        return True
    if _is_float(s_a) and _is_float(s_b):
        return float(s_a) == float(s_b)
    return False


def kwargs_style_args(params):
    """{'a': 1, 'b': 'x'} -> ['--a', '1', '--b', 'x']（py2node 用例风格）。"""
    args = []
    for key, value in params.items():
        args.extend([f"--{key}", str(value)])
    return args


def positional_args(record, key_field="file_name"):
    """去掉文件名字段后按位置传参（cpp2rust 用例风格）。"""
    return [str(v) for k, v in record.items() if k != key_field]
