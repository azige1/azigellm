"""从大模型输出中解析新 docstring / 类型信息。"""

import random
import ast
from collections import defaultdict

def extract_code_reply(output):
    outputlines = output.split("\n")
    indexlines = [i for i, line in enumerate(outputlines) if "```" in line]
    if len(indexlines) < 2:
        return ""
    return "\n".join(outputlines[indexlines[-2] + 1 : indexlines[-1]])

def parse_refined_docstring(output):
    code = extract_code_reply(output)
    if code == "":
        return ""
    try:
        tree = ast.parse(code).body[-1]
        docstring = ast.get_docstring(tree)  
        return docstring
    except SyntaxError:
        return ""

def collect_refined_specs(outputs):
    outputs = [output[0] for output in outputs]
    specs = [parse_refined_docstring(output) for output in outputs]
    return specs

def format_captured_value(arg):
    serialized_inputs = arg["serialized_inputs"]
    serialized_output = arg["serialized_output"]
    serialized_inputs = [f"{k}={v}" for k, v in serialized_inputs.items()]
    serialized_inputs = "\n".join(serialized_inputs)
    new_string = (
        f"Input Arguments:\n{serialized_inputs}\n\nOutput:\n{serialized_output}"
    )
    return new_string

def collect_captured_types(args):
    input_types = [arg["input_types"] for arg in args]
    merged_input_types = defaultdict(list)
    for input_type in input_types:
        for k, v in input_type.items():
            merged_input_types[k].append(v)
    merged_input_types = "\n".join(
        [f"{k} : {', '.join(list(set(v)))}" for k, v in merged_input_types.items()]
    )

    output_types = [arg["output_type"] for arg in args]
    output_types = ", ".join(list(set(output_types)))
    return merged_input_types, output_types

def format_io_examples(captured_args):
    captured_args_strings = [format_captured_value(arg) for arg in captured_args]

    if len(captured_args_strings) > 0:
        example_io = random.sample(captured_args_strings, 1)[0]
        example_substring = f"Example IO:\n{example_io}\n\n"
    else:
        example_substring = ""

    return example_substring
