"""等价测试执行器：串行 / 并行调度。"""

import fire
from tqdm import tqdm
import traceback

from codeharvest.workspace import *
from codeharvest.core import *
from codeharvest.utils.io import *
from codeharvest.parallel import run_jobs_parallel_iter

from codeharvest.execution.config import ExecutionConfig
from codeharvest.execution.service import SandboxServiceManager
from codeharvest.execution.equivalence import run_target_on_port, run_target_mp

class EquivalenceRunner:
    @staticmethod
    def run(args):
        futs = read_test_targets(TESTGEN_DIR / args.in_file)
        print(f"Loaded {len(futs)} functions under test")

        if args.function:
            futs = [f for f in futs if f.name == args.function]

        new_futs = []
        if args.execution_multiprocess == 0:
            new_futs = EquivalenceRunner._run_futs_sequential(futs, args)
        else:
            new_futs = EquivalenceRunner._run_futs_parallel(futs, args)

        SandboxServiceManager.shutdown()
        dump_test_targets(new_futs, EXECUTION_DIR / f"{args.exp_id}_out.json")

    @staticmethod
    def _run_futs_sequential(futs, args):
        new_futs = []
        for i, fut in tqdm(enumerate(futs), desc="Running tests", total=len(futs)):
            port = args.port
            local = args.local
            image = args.image
            try:
                output = run_target_on_port(fut, port, local, image, reuse_port=True)
            except Exception as e:
                print(f"Error@{fut.repo_id}:\n{repr(e)}")
                tb = traceback.format_exc()
                print(tb)
                continue
            new_futs.append(output[2])

            if (i + 1) % 20 == 0:
                dump_test_targets(
                    new_futs, EXECUTION_DIR / f"{args.exp_id}_out.json"
                )

        return new_futs

    @staticmethod
    def _run_futs_parallel(futs, args):
        new_futs = []

        for i in range(0, len(futs), args.batch_size):
            batch = [(f, args.local, args.image) for f in futs[i : i + args.batch_size]]

            outputs = run_jobs_parallel_iter(
                run_target_mp,
                batch,
                num_workers=args.execution_multiprocess,
                timeout_per_task=args.timeout_per_task,
                use_progress_bar=True,
            )

            for o in outputs:
                if o.is_success():
                    new_futs.append(o.result[2])  
                else:
                    print(f"Error: {o.exception_tb}")

            SandboxServiceManager.shutdown()
            dump_test_targets(
                new_futs, EXECUTION_DIR / f"{args.exp_id}_out.json"
            )

        return new_futs

if __name__ == "__main__":
    exec_args = fire.Fire(ExecutionConfig)
    EXECUTION_DIR.mkdir(parents=True, exist_ok=True)
    EquivalenceRunner.run(exec_args)
