"""规格合成器主流程。"""

import fire
from tqdm import tqdm

from codeharvest.synthesis.specs.config import SpecGenConfig
from codeharvest.synthesis.specs.task import SpecTask
from codeharvest.synthesis.specs.parsing import collect_refined_specs
from codeharvest.llm.batch import CompletionEngine
from codeharvest.utils.io import read_test_targets, dump_codegen_tasks
from codeharvest.core.codegen_task import make_codegen_task
from codeharvest.workspace import WORKSPACE_DIR

class SpecRefiner:

    @staticmethod
    def generate(args):
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        functions = read_test_targets(WORKSPACE_DIR / args.in_file)

        functions = functions[:5]

        tasks = SpecRefiner.prepare_tasks(functions)
        payloads = [task.chat_messages for task in tasks]

        outputs = CompletionEngine.run_chat_completions(args, payloads)
        results = collect_refined_specs(outputs)

        codegen_probs = []
        for func, spec in zip(functions, results):
            codegen_probs.append(make_codegen_task(func, spec))  

        dump_codegen_tasks(
            codegen_probs, WORKSPACE_DIR / f"{args.exp_id}_specgen.json"
        )

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
