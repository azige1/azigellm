"""过采样生成：同一函数生成多份候选测试。"""

import fire
from tqdm import tqdm

from codeharvest.core import TestBatch
from codeharvest.synthesis.tests import TestGenConfig, TestTask
from codeharvest.synthesis.tests.parsing import collect_test_sources
from codeharvest.llm.batch import CompletionEngine
from codeharvest.utils.io import read_test_targets, dump_test_targets
from codeharvest.workspace import TESTGEN_DIR, current_timestamp

class TestOversampler:

    @staticmethod
    def oversample(args):
        futs = read_test_targets(TESTGEN_DIR / args.in_file)

        
        ref_tests = [fut.tests["test_0"] for fut in futs]
        tasks = TestOversampler.prepare_tasks(futs)
        tasks = TestOversampler.update_tasks(tasks, ref_tests)

        for i in range(1, args.oversample_rounds + 1):
            payloads = [task.chat_messages for task in tasks]
            outputs = CompletionEngine.run_chat_completions(args, payloads)
            results = collect_test_sources(outputs)

            for fut, test in zip(futs, results):
                tests = TestBatch(
                    tests=fut.tests,
                    operation="oversample",
                    gen_model=args.model_name,
                    gen_date=current_timestamp(),
                )
                tests.add(f"test_{i}", test)
                fut.update_history(tests)

            tasks = TestOversampler.update_tasks(tasks, results)

        dump_test_targets(futs, TESTGEN_DIR / f"{args.exp_id}_oversample.json")

    

    @staticmethod
    def prepare_tasks(functions) -> list[TestTask]:
        tasks = []

        for func in tqdm(functions):
            tasks.append(TestTask(func_meth=func))

        return tasks

    @staticmethod
    def update_tasks(tasks, results) -> list[TestTask]:
        for task, result in zip(tasks, results):
            task.update(result)

        return tasks

if __name__ == "__main__":
    args = fire.Fire(TestGenConfig)

    TestOversampler.oversample(args)
