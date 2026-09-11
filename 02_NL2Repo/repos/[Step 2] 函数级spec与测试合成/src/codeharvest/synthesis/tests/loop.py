"""生成-执行智能体：多轮迭代直到覆盖率达标。"""

import os
import fire
from tempfile import NamedTemporaryFile

from codeharvest.core import *
from codeharvest.utils.io import *
from codeharvest.workspace import *

from codeharvest.execution.runner import EquivalenceRunner
from codeharvest.synthesis.tests.config import GenExecConfig
from codeharvest.synthesis.tests.generator import EquivalenceTestGenerator
from codeharvest.synthesis.tests.task import TestTask
from codeharvest.synthesis.tests.parsing import render_coverage_hints

class GenerateExecuteAgent:

    @staticmethod
    def agent_loop(args):
        functions = read_callable_records(EXTRACTED_DATA_DIR / args.in_file)
        functions = functions[:15]  

        if args.function:
            functions = [f for f in functions if f.name == args.function]

        assert len(functions) > 0, "No functions found for the given input"

        final_output_file = EXECUTION_DIR / f"{args.exp_id}_out.json"
        worklist: list[TestTask] = EquivalenceTestGenerator.prepare_tasks(args, functions)
        count = len(worklist)

        current_results: list[FunctionTestTarget | MethodTestTarget] = []
        futs = []

        for round in range(1, args.max_rounds + 1):
            print(f"Starting round {round}/{args.max_rounds}")
            
            futs = GenerateExecuteAgent.generate(args, futs, worklist, round)
            futs = GenerateExecuteAgent.execute(args, futs)
            status_map, _continue = GenerateExecuteAgent.filter(futs, worklist, args)

            
            current_results = GenerateExecuteAgent._update_current_results(
                current_results, futs, status_map, round, args.max_rounds
            )

            
            worklist = GenerateExecuteAgent._update_worklist(worklist, futs, status_map, round)

            if not _continue or len(worklist) == 0:
                print(f"Reached minimum criteria. Stopping at round {round}")
                break

            if round == args.max_rounds:
                print(f"Reached max rounds. Stopping at round {round}")
                break

            good_ratio = (count - len(worklist)) / count
            print(f"Round {round} completed. Status: {good_ratio:.2f} good FUTs.\n")

        
        sorted_results = GenerateExecuteAgent._sort_results(functions, current_results)
        dump_test_targets(sorted_results, final_output_file)

    @staticmethod
    def generate(args, futs, worklist, round):
        new_futs = EquivalenceTestGenerator._generate(args, worklist, test_id=round - 1)
        if round == 1:
            futs = new_futs
        else:
            new_tests = {f.id: f.test_history.history[-1] for f in new_futs}
            futs = [f for f in futs if f.id in new_tests]
            for f in futs:
                f.update_history(new_tests[f.id])
        return futs

    @staticmethod
    def execute(args, futs):
        with NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmp_file:
            dump_test_targets(futs, tmp_file.name)
            args.in_file = tmp_file.name
            EquivalenceRunner.run(args)

        
        futs = read_test_targets(EXECUTION_DIR / f"{args.exp_id}_out.json")
        return futs

    @staticmethod
    def filter(futs, tasks, args):
        good_futs, status_map = 0, {}

        for i, fut in enumerate(futs):
            fut_coverage = fut.coverage.get("branch_coverage_percentage", 0) / 100
            if fut.is_passing:
                if fut_coverage < args.min_cov:
                    status_map[i] = (False, "improve_coverage", render_coverage_hints(fut))
                else:
                    good_futs += 1
                    status_map[i] = (True, None, None)
                continue

            
            if fut.exec_stats is None:
                continue

            
            status_map[i] = (False, "fix_error", fut.errors)

        _continue = (good_futs / len(futs)) < args.min_valid
        return status_map, _continue

    

    @staticmethod
    def _update_worklist(worklist, futs, status_map, round):
        return [
            GenerateExecuteAgent._update_task(
                worklist[i], futs[i].tests[f"test_{round-1}"], ut, feedback
            )
            for i, (passing, ut, feedback) in status_map.items()
            if not passing
        ]

    @staticmethod
    def _update_task(task, result, update_type, feedback):
        task.update(result, feedback, update_type)
        return task

    @staticmethod
    def _update_current_results(current_results, futs, status_map, round, max_rounds):
        for i, (passing, _, _) in status_map.items():
            if passing:
                current_results.append(futs[i])
            elif round == max_rounds:
                current_results.append(futs[i])
        return current_results

    @staticmethod
    def _sort_results(functions, current_results):
        sorted_results = []
        for f in functions:
            for res in current_results:
                if res.id == f.id:
                    sorted_results.append(res)
                    break
        assert len(current_results) == len(functions)
        assert len(sorted_results) == len(functions)
        return sorted_results

if __name__ == "__main__":
    args = fire.Fire(GenExecConfig)
    GenerateExecuteAgent.agent_loop(args)
