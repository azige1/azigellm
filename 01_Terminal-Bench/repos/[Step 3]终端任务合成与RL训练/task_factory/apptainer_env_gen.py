"""第四阶段（Apptainer 路线）：生成 .def 定义文件并构建 SIF 验证。

适用于无 Docker 守护进程的集群环境（HPC），用 Apptainer 代替 Docker
作为容器运行时。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path
import sys
import re
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.insert(0, str(Path().resolve()))

from task_factory import batch_chat_completions

SYSTEM_MSG = """ You are an expert in Apptainer/Singularity.
You are given a task description and will be tested so that the initial state of the container is set up in a way that an agent can be tested on the task.
Make sure that the container is set up in a way that an agent can be tested on the task.
Basically ensure that the task is valid when the container is built: Clone a repository, create a file, create a directory, create a process, etc.
Install pytest in the container.
Don't include the tests in the response (no %test)
The agent will not have root access. So make sure that the right permissions are set for the files and directories.
Always use this image: docker://ubuntu:22.04
To add it to the def file, use:
Bootstrap: localimage
From: ./ubuntu_22.04.sif"""

BASE_USER_TEMPLATE = """
Using the task description template and pytest failures below, output a complete
Apptainer `.def` file.

Question description given to the agent:
{task_description}

Here is some ground truth data that might be useful to you:
{truth}

Here are the tests that will be run on the container:
{test_py}

Previous failures (may be empty):
{failures}

Respond with the Apptainer `.def` file only. You should think step by step and then write the file. The file should be valid and buildable.
Make sure that you create the right files and directories for the task.
Eg: for a csv task you will have to create a csv file. For a process cleanup task you will have to create processes.
Don't include the tests in the response or copy a test file.
Don't add any of the output files or directories that the student will create.
Don't create / touch empty files for the agent.
Remember to install pytest in the container.
The home path is /home/user.
Don't override HOME in the %environment section; let Apptainer bind the host $HOME.
"""


def build_and_test_sif(def_template: str, test_py: str) -> tuple[bool, str]:
    """用 .def 构建 Apptainer 镜像，并在容器内执行给定的 pytest 代码。

    参数:
        def_template: Apptainer ``.def`` 文件文本。
        test_py: 用于验证容器初始状态的 pytest 模块文本。
    """
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # 1. 落盘定义文件与测试文件
        def_path = td_path / "container.def"
        def_path.write_text(def_template)

        test_file = td_path / "test_initial_state.py"
        test_file.write_text(test_py)

        # 2. 构建 SIF 镜像
        sif_path = td_path / "img.sif"
        build_rc = subprocess.run(["apptainer", "build", str(sif_path), str(def_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180).returncode
        if build_rc:
            print(f"Apptainer build failed: {build_rc}")
            return False, "Apptainer build failed"

        # 3. 容器内执行 pytest
        proc = subprocess.run(
            [
                "apptainer",
                "exec",
                "--fakeroot",
                "--userns",
                "--writable-tmpfs",
                "--cleanenv",
                str(sif_path),
                "pytest",
                "-q",
                str(test_file.name),
            ],
            cwd=td,
            capture_output=True,
            text=True,
        )

        # 先删 SIF 再清临时目录
        if sif_path.exists():
            sif_path.unlink()

        shutil.rmtree(td_path, ignore_errors=True)
        # 4. 返回成功与否与合并输出
        return proc.returncode == 0, proc.stdout + proc.stderr


def extract_def_template(def_template: str) -> str:
    """清洗模型输出，得到可直接落盘的 Apptainer 定义文本。

    模型本应只回复 .def 内容，但可能带 Markdown 围栏或解释文字。
    这里取第一个围栏代码块（没有围栏则视整段为定义），再去掉公共缩进。
    """
    cleaned = def_template.replace("\r\n", "\n").strip()

    fence_re = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\n(?P<code>[\s\S]*?)```", re.MULTILINE)
    match = fence_re.search(cleaned)
    if match:
        cleaned = match.group("code").strip()

    cleaned = textwrap.dedent(cleaned).strip()
    return cleaned

def batch_generate_def_files(
    items: List[Tuple[str, str, str]],
    *,
    model: str = "qwen/Qwen2.5-3B-Instruct",
    temperature: float = 0.6,
    max_tokens: int = 2048,
    max_concurrency: int = 64,
) -> List[Optional[str]]:
    """批量生成 .def 并并行构建验证。

    items 为 ``(task_description, truth, test_py)`` 列表；
    返回与输入等长的列表：通过验证的 def 文本或 ``None``。
    """

    messages: list[list[dict[str, str]]] = []
    for task_description, truth, test_py in items:
        prompt = BASE_USER_TEMPLATE.format(
            task_description=task_description,
            truth=truth,
            test_py=test_py,
            failures="None yet",
        )
        messages.append([
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ])

    responses = batch_chat_completions(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        num_completions=1,
        max_concurrency=max_concurrency,
    )

    results: List[Optional[str]] = [None] * len(items)

    def worker(index: int, item: Tuple[str, str, str], resp_obj) -> Tuple[int, Optional[str]]:
        try:
            if resp_obj is None:
                return index, None
            content = resp_obj.choices[0].message.content
            def_text = extract_def_template(content)
            _task_description, _truth, test_py = item
            ok, _ = build_and_test_sif(def_text, test_py)
            return index, (def_text if ok else None)
        except Exception:
            return index, None

    futures = []
    with ThreadPoolExecutor(max_workers=64) as executor:
        for idx, (item, resp) in enumerate(zip(items, responses)):
            futures.append(executor.submit(worker, idx, item, resp))

        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, value = fut.result()
            results[idx] = value

    return results


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = ap.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"
    final_test_path = task_path / "test_final_state.py"

    test_py = initial_test_path.read_text()
    def_text = def_path.read_text()

    success, output = build_and_test_sif(def_text, test_py)
    print(success)
    print(output)
