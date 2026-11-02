"""生成-执行智能体：多轮迭代直到覆盖率达标。"""

import json
import fire
from tempfile import NamedTemporaryFile

from codeharvest.core import *
from codeharvest.core.targets import make_code_under_test
from codeharvest.utils.io import *
from codeharvest.workspace import *

from codeharvest.execution.runner import EquivalenceRunner
from codeharvest.synthesis.tests.config import GenExecConfig
from codeharvest.synthesis.tests.generator import EquivalenceTestGenerator
from codeharvest.synthesis.tests.task import TestTask
from codeharvest.synthesis.tests.feedback import (
    render_error_feedback,
    render_coverage_feedback,
)

EXECUTION_LOST_FEEDBACK = (
    "The execution service returned no result for this test (lost during execution). "
    "Rewrite the complete test so that it can be executed."
)


class GenerateExecuteAgent:

    @staticmethod
    def agent_loop(args):
        functions = read_callable_records(EXTRACTED_DATA_DIR / args.in_file)

        if args.limit:
            functions = functions[: args.limit]

        if args.function:
            functions = [f for f in functions if f.name == args.function]

        assert len(functions) > 0, "No functions found for the given input"

        final_output_file = EXECUTION_DIR / f"{args.exp_id}_out.json"
        worklist: list[TestTask] = EquivalenceTestGenerator.prepare_tasks(args, functions)
        count = len(worklist)

        current_results: dict[str, FunctionTestTarget | MethodTestTarget] = {}
        futs = []
        rounds_report = []
        first_round_pass = 0
        first_round_seen = 0
        self_equivalence_intercepts = 0

        for round in range(1, args.max_rounds + 1):
            print(f"Starting round {round}/{args.max_rounds}")

            futs = GenerateExecuteAgent.generate(args, futs, worklist, round)
            futs = GenerateExecuteAgent.execute(args, futs)
            futs_by_id = {f.id: f for f in futs}

            status_map = GenerateExecuteAgent.filter(futs_by_id, worklist, args)

            if round == 1:
                for fut in futs_by_id.values():
                    first_round_seen += 1
                    if fut.is_passing:
                        first_round_pass += 1
                    verdict = (fut.exec_stats or {}).get("verdict")
                    if verdict == "test_defect":
                        self_equivalence_intercepts += 1

            current_results = GenerateExecuteAgent._update_current_results(
                current_results, futs_by_id, status_map, round, args.max_rounds
            )
            worklist = GenerateExecuteAgent._update_worklist(
                worklist, futs_by_id, status_map, round
            )

            good_ratio = (count - len(worklist)) / count if count else 0
            round_entry = {
                "round": round,
                "processed": len(status_map),
                "cumulative_good_ratio": good_ratio,
            }
            if round == 1:
                round_entry["first_round_pass_rate"] = (
                    first_round_pass / first_round_seen if first_round_seen else 0
                )
                round_entry["self_equivalence_intercepts"] = self_equivalence_intercepts
            rounds_report.append(round_entry)
            print(f"Round {round} completed. Status: {good_ratio:.2f} good FUTs.\n")

            if good_ratio >= args.min_valid or len(worklist) == 0:
                print(f"Reached minimum criteria. Stopping at round {round}")
                break

            if round == args.max_rounds:
                print(f"Reached max rounds. Stopping at round {round}")
                break

        # 最后一轮仍未达标或执行丢失的函数照常写出
        for task in worklist:
            if task.func_meth.id in current_results:
                continue
            fallback = make_code_under_test(task.func_meth)
            fallback.update_exec_stats({"error": EXECUTION_LOST_FEEDBACK})
            current_results[task.func_meth.id] = fallback

        GenerateExecuteAgent._write_rounds_report(args.exp_id, rounds_report)

        sorted_results = GenerateExecuteAgent._sort_results(functions, current_results)
        dump_test_targets(sorted_results, final_output_file)

    @staticmethod
    def _write_rounds_report(exp_id: str, rounds_report: list[dict]) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{exp_id}_rounds.json"
        with open(report_path, "w") as f:
            json.dump(rounds_report, f, indent=2)
        print(f"轮次报告: {report_path}")

    @staticmethod
    def generate(args, futs, worklist, round):
        new_futs = EquivalenceTestGenerator._generate(args, worklist, test_id=round - 1)
        if round == 1:
            futs = new_futs
        else:
            # 已有执行记录的函数追加新一轮测试；上一轮执行丢失的函数直接使用新生成的记录
            previous = {f.id: f for f in futs}
            merged = []
            for new_fut in new_futs:
                old_fut = previous.get(new_fut.id)
                if old_fut is None:
                    merged.append(new_fut)
                    continue
                old_fut.update_history(new_fut.test_history.history[-1])
                merged.append(old_fut)
            futs = merged
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
    def filter(futs_by_id: dict, worklist: list[TestTask], args):
        """按 fut.id 对齐执行结果与工作清单，返回 {func_id: (passing, update_type, feedback)}。"""
        status_map = {}

        for task in worklist:
            func_id = task.func_meth.id
            fut = futs_by_id.get(func_id)

            if fut is None:
                status_map[func_id] = (False, "fix_error", EXECUTION_LOST_FEEDBACK)
                continue

            fut_coverage = fut.coverage.get("branch_coverage_percentage", 0) / 100
            if fut.is_passing:
                if fut_coverage < args.min_cov:
                    status_map[func_id] = (
                        False,
                        "improve_coverage",
                        render_coverage_feedback(fut),
                    )
                else:
                    status_map[func_id] = (True, None, None)
                continue

            if fut.exec_stats is None:
                status_map[func_id] = (False, "fix_error", EXECUTION_LOST_FEEDBACK)
                continue

            status_map[func_id] = (False, "fix_error", render_error_feedback(fut))

        return status_map

    @staticmethod
    def _update_worklist(worklist, futs_by_id, status_map, round):
        updated = []
        for task in worklist:
            passing, update_type, feedback = status_map.get(
                task.func_meth.id, (True, None, None)
            )
            if passing:
                continue
            fut = futs_by_id.get(task.func_meth.id)
            result_test = ""
            if fut is not None and f"test_{round-1}" in fut.tests:
                result_test = fut.tests[f"test_{round-1}"]
            task.update(result_test, feedback, update_type)
            updated.append(task)
        return updated

    @staticmethod
    def _update_current_results(current_results, futs_by_id, status_map, round, max_rounds):
        for func_id, (passing, _, _) in status_map.items():
            fut = futs_by_id.get(func_id)
            if fut is None:
                continue
            if passing:
                current_results[func_id] = fut
            elif round == max_rounds:
                current_results[func_id] = fut
        return current_results

    @staticmethod
    def _sort_results(functions, current_results: dict):
        """按抽取顺序输出结果；执行阶段丢失且无记录的函数不再断言失败。"""
        sorted_results = []
        for f in functions:
            if f.id in current_results:
                sorted_results.append(current_results[f.id])
        return sorted_results

if __name__ == "__main__":
    args = fire.Fire(GenExecConfig)
    GenerateExecuteAgent.agent_loop(args)
