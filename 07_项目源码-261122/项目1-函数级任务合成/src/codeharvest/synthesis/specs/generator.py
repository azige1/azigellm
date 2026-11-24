"""规格合成器主流程。"""

import os

import fire
from tqdm import tqdm

from codeharvest.synthesis.specs.config import SpecGenConfig
from codeharvest.synthesis.specs.task import SpecTask
from codeharvest.synthesis.specs.parsing import collect_refined_specs
from codeharvest.llm.batch import CompletionEngine
from codeharvest.utils.io import read_test_targets, dump_codegen_tasks
from codeharvest.core.codegen_task import make_codegen_task
from codeharvest.workspace import EXECUTION_DIR

MIN_BRANCH_COVERAGE = 0.8


def is_spec_ready(fut) -> bool:
    """只处理自等价通过且分支覆盖率达标的函数。"""
    if not fut.is_passing:
        return False
    coverage = fut.coverage or {}
    branch_cov = coverage.get("branch_coverage_percentage", 0) / 100
    return branch_cov >= MIN_BRANCH_COVERAGE


class SpecRefiner:

    @staticmethod
    def generate(args):
        functions = SpecRefiner.load_input_functions(args)

        if args.limit:
            functions = functions[: args.limit]

        tasks = SpecRefiner.prepare_tasks(functions)
        payloads = [task.chat_messages for task in tasks]

        outputs = CompletionEngine.run_chat_completions(args, payloads)
        results = collect_refined_specs(outputs)

        codegen_probs = []
        for func, spec in zip(functions, results):
            if not spec:
                continue
            codegen_probs.append(make_codegen_task(func, spec))

        dump_codegen_tasks(
            codegen_probs, EXECUTION_DIR / f"{args.exp_id}_specgen.json"
        )

    @staticmethod
    def load_input_functions(args):
        in_file = args.in_file or f"{args.exp_id}_out.json"
        file_path = in_file
        if not os.path.exists(str(file_path)):
            file_path = EXECUTION_DIR / in_file
        functions = read_test_targets(file_path)
        return [f for f in functions if is_spec_ready(f)]

    @staticmethod
    def prepare_tasks(functions):
        tasks = []

        for func in functions:
            generated_test = "\n".join(func.tests.values())
            tasks.append(SpecTask(func_meth=func, generated_test=generated_test))

        return tasks

if __name__ == "__main__":
    args = fire.Fire(SpecGenConfig)

    SpecRefiner.generate(args)
