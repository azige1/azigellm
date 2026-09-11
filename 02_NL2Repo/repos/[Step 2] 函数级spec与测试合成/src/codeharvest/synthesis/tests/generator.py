"""等价测试生成器主流程。"""

import ast
import argparse
import tiktoken
import fire
from tqdm import tqdm

from codeharvest.core import FunctionRecord, MethodRecord, PromptContext, TestBatch
from codeharvest.core.targets import make_code_under_test

from codeharvest.analysis.ast_tools.transforms import MethodsRemover
from codeharvest.synthesis.context import build_prompt_context
from codeharvest.synthesis.tests import TestTask, TestGenConfig
from codeharvest.llm.batch import CompletionEngine
from codeharvest.synthesis.tests.parsing import collect_test_sources
from codeharvest.parallel import run_jobs_parallel_iter
from codeharvest.utils.io import (
    read_callable_records,
    read_test_targets,
    dump_callable_records,
    dump_test_targets,
)
from codeharvest.workspace import EXTRACTED_DATA_DIR, TESTGEN_DIR, current_timestamp

class EquivalenceTestGenerator:

    @staticmethod
    def generate(args):
        functions = read_callable_records(EXTRACTED_DATA_DIR / args.in_file)

        if args.function:
            functions = [f for f in functions if f.name == args.function]

        tasks = EquivalenceTestGenerator.prepare_tasks(args, functions)
        EquivalenceTestGenerator._generate(args, tasks, write_to_file=True)

    @staticmethod
    def _generate(args, tasks, test_id=0, write_to_file=False):
        payloads = [task.chat_messages for task in tasks]
        outputs = CompletionEngine.run_chat_completions(args, payloads)
        results = collect_test_sources(outputs)
        futs = [make_code_under_test(task.func_meth) for task in tasks]

        for fut, test, op, msgs in zip(futs, results, outputs, payloads):
            tests = TestBatch(
                tests={f"test_{test_id}": test},
                gen_model=args.model_name,
                gen_date=current_timestamp(),
            )

            if args.save_chat:
                msgs.append({"role": "assistant", "content": op[0]})
                tests.update_chat_messages(msgs)

            fut.update_history(tests)

        if write_to_file:
            TESTGEN_DIR.mkdir(parents=True, exist_ok=True)
            dump_test_targets(
                futs, TESTGEN_DIR / f"{args.exp_id}_generate.json"
            )

        return futs

    @staticmethod
    def filter(args):
        futs = read_test_targets(TESTGEN_DIR / args.in_file)

        for fut in tqdm(futs):
            assert fut.exec_stats is not None
            filtered_tests = {}

            for sample_id, stats in fut.exec_stats.items():
                failed_tests = stats.get("failed_names", [])
                errored_tests = stats.get("errored_names", [])

                tests_to_filter = failed_tests + errored_tests
                test = fut.tests.get(sample_id, None)
                assert test is not None

                if tests_to_filter == []:
                    filtered_tests[sample_id] = test
                    continue

                transformer = MethodsRemover(ast.parse(test), tests_to_filter)
                cleaned_test = ast.unparse(transformer.transform())
                filtered_tests[sample_id] = cleaned_test

            fut.update_history(
                TestBatch(tests=filtered_tests, operation="filter", gen_date=current_timestamp())
            )

        dump_test_targets(futs, TESTGEN_DIR / f"{args.exp_id}_filter.json")

    @staticmethod
    def prepare_tasks(args, functions) -> list[TestTask]:
        context_gen_tasks = [(args.context_type, func, 6000) for func in functions]
        context_iter = run_jobs_parallel_iter(
            build_prompt_context,
            context_gen_tasks,
            num_workers=8,
            use_progress_bar=True,
            progress_bar_desc="Generating contexts",
        )

        tasks = []

        for func, task_result in zip(functions, context_iter):
            if task_result.is_success():
                context = task_result.result
                func.add_context(context)
                tasks.append(TestTask(func_meth=func))
            else:
                print(f"Error generating context:\n{task_result.exception_tb}")

        return tasks

    @staticmethod
    def update_tasks(tasks, results) -> list[TestTask]:
        for task, result in zip(tasks, results):
            task.update(result)

        return tasks

if __name__ == "__main__":
    args = fire.Fire(TestGenConfig)

    EquivalenceTestGenerator.generate(args)
