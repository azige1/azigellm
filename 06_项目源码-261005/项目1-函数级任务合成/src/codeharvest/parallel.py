"""多进程执行工具：带超时与异常回收的并行任务队列。"""

import sys
import queue
import resource
import traceback
from typing import Any, Optional
from enum import Enum
import multiprocessing as mp
from concurrent.futures import TimeoutError
from typing import Callable, Any, Iterator

import attrs
import tqdm
from pebble import concurrent, ProcessPool, ProcessExpired
import platform

class JobTimeoutError(TimeoutError):
    pass

def make_job_queue() -> queue.Queue[Any]:
    manager = mp.Manager()
    return manager.Queue()

QueueEmptyException = queue.Empty

def run_callable_in_subprocess(
    func: Callable,
    *args,
    _timeout: None | int = None,
    _use_spawn: bool = True,
    **kwargs,
):
    mode = "spawn" if _use_spawn else "fork"
    c_func = concurrent.process(timeout=_timeout, context=mp.get_context(mode))(func)
    future = c_func(*args, **kwargs)

    try:
        result = future.result()
        return result

    except TimeoutError:
        raise JobTimeoutError

class JobStatus(Enum):
    SUCCESS = 0
    EXCEPTION = 1
    TIMEOUT = 2
    PROCESS_EXPIRED = 3

@attrs.define(eq=False, repr=False)
class JobResult:
    status: JobStatus

    result: None | Any = None
    exception_tb: None | str = None

    def is_success(self) -> bool:
        return self.status == JobStatus.SUCCESS

    def is_timeout(self) -> bool:
        return self.status == JobStatus.TIMEOUT

    def is_exception(self) -> bool:
        return self.status == JobStatus.EXCEPTION

    def is_process_expired(self) -> bool:
        return self.status == JobStatus.PROCESS_EXPIRED

def configure_worker_limits(limit: int) -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))

def run_jobs_parallel_iter(
    func: Callable,
    tasks: list[Any],
    num_workers: int = 2,
    timeout_per_task: None | int = None,
    use_progress_bar: bool = False,
    progress_bar_desc: None | str = None,
    max_tasks_per_worker: None | int = None,
    use_spawn: bool = True,
    max_mem: int = 1024 * 1024 * 1024 * 4,
) -> Iterator[JobResult]:

    mode = "spawn" if use_spawn else "fork"

    with ProcessPool(
        
        max_workers=num_workers,
        max_tasks=0 if max_tasks_per_worker is None else max_tasks_per_worker,
        context=mp.get_context(mode),
        
    ) as pool:
        future = pool.map(func, tasks, timeout=timeout_per_task)

        iterator = future.result()
        if use_progress_bar:
            pbar = tqdm.tqdm(
                desc=progress_bar_desc,
                total=len(tasks),
                dynamic_ncols=True,
            )
        else:
            pbar = None

        succ = timeouts = exceptions = expirations = 0

        while True:
            try:
                result = next(iterator)

            except StopIteration:
                break

            except TimeoutError as error:
                yield JobResult(
                    status=JobStatus.TIMEOUT,
                )

                timeouts += 1

            except ProcessExpired as error:
                yield JobResult(
                    status=JobStatus.PROCESS_EXPIRED,
                )
                expirations += 1

            except Exception as error:
                exception_tb = traceback.format_exc()

                yield JobResult(
                    status=JobStatus.EXCEPTION,
                    exception_tb=exception_tb,
                )
                exceptions += 1

            else:
                yield JobResult(
                    status=JobStatus.SUCCESS,
                    result=result,
                )

                succ += 1

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    succ=succ, timeouts=timeouts, exc=exceptions, p_exp=expirations
                )
                sys.stdout.flush()
                sys.stderr.flush()

def run_jobs_parallel(
    func: Callable,
    tasks: list[Any],
    num_workers: int = 2,
    timeout_per_task: None | int = None,
    use_progress_bar: bool = False,
    progress_bar_desc: None | str = None,
    max_tasks_per_worker: None | int = None,
    use_spawn: bool = True,
) -> list[JobResult]:

    task_results: list[JobResult] = list(
        run_jobs_parallel_iter(
            func=func,
            tasks=tasks,
            num_workers=num_workers,
            timeout_per_task=timeout_per_task,
            use_progress_bar=use_progress_bar,
            progress_bar_desc=progress_bar_desc,
            max_tasks_per_worker=max_tasks_per_worker,
            use_spawn=use_spawn,
        )
    )

    return task_results
