"""评测执行后端：本地 Docker 容器、E2B 沙箱、宿主机直跑。

三个后端共享同一套接口（``ExecContext``），上层只关心
``exec_run_with_timeout`` 与上下文管理器协议。docker 与 e2b 均为惰性导入，
仅使用对应后端时才需要安装。
"""

from abc import ABC, abstractmethod
import logging
import os
import shutil
import signal
import subprocess
import time
from enum import auto
from strenum import StrEnum
from pathlib import Path
import tempfile
from typing import Optional, Type
from types import TracebackType

from rebuild_spec.core.constants import CopyFiles
from rebuild_spec.core.specs import EnvSpec
from rebuild_spec.core.gitops import release_logger


class ExecBackend(StrEnum):
    LOCAL = auto()
    E2B = auto()
    HOST = auto()


class ExecContext(ABC):
    """一次评测执行的上下文：负责准备环境、执行命令、回收结果。"""

    def __init__(
        self,
        spec: EnvSpec,
        logger: logging.Logger,
        timeout: int,
        num_cpus: int,
        log_dir: Path,
        files_to_copy: Optional[CopyFiles] = None,
        files_to_collect: Optional[list[str]] = None,
        rebuild_image: bool = False,
    ):
        # 上下文的生命周期不一定覆盖整个对象的生命周期
        self.spec = spec
        self.logger = logger
        self.timeout = timeout
        self.num_cpus = num_cpus
        self.log_dir = log_dir
        self.files_to_collect = files_to_collect

    @abstractmethod
    def exec_run_with_timeout(self, command: str) -> tuple[str, bool, float]:
        """执行命令，返回 (输出, 是否超时, 耗时)。"""
        raise NotImplementedError

    def __enter__(self):
        return self

    @abstractmethod
    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        raise NotImplementedError


class DockerContext(ExecContext):
    """本地 Docker 后端。"""

    def __init__(
        self,
        spec: EnvSpec,
        logger: logging.Logger,
        timeout: int,
        num_cpus: int,
        log_dir: Path,
        files_to_copy: Optional[CopyFiles] = None,
        files_to_collect: Optional[list[str]] = None,
        rebuild_image: bool = False,
    ):
        from rebuild_spec.core.containers import (
            cleanup_container,
            copy_to_container,
            copy_from_container,
            create_container,
            exec_run_with_timeout,
        )

        super().__init__(
            spec,
            logger,
            timeout,
            num_cpus,
            log_dir,
            files_to_copy=files_to_copy,
            files_to_collect=files_to_collect,
        )

        self.client = docker.from_env()
        self.container = create_container(
            client=self.client,
            image_name=spec.repo_image_key,
            container_name=spec.get_container_name(),
            nano_cpus=num_cpus,
            logger=logger,
        )
        self.container.start()
        if files_to_copy:
            for _, f in files_to_copy.items():
                copy_to_container(self.container, f["src"], f["dest"])  # type: ignore
        self._cleanup_container = cleanup_container
        self._copy_from_container = copy_from_container

    def exec_run_with_timeout(self, command: str) -> tuple[str, bool, float]:
        from rebuild_spec.core.containers import exec_run_with_timeout

        output = exec_run_with_timeout(self.container, command, self.timeout)

        if self.files_to_collect:
            for fname in self.files_to_collect:
                file = Path(self.spec.repo_directory) / fname
                # 先确认文件存在再取回
                exit_code, test_output = self.container.exec_run(
                    f"test -e {file}", demux=True
                )
                if exit_code == 0:
                    self._copy_from_container(self.container, file, self.log_dir / fname)
        return output

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        self._cleanup_container(self.client, self.container, self.logger)
        release_logger(self.logger)


class E2BContext(ExecContext):
    """E2B 云沙箱后端：一个沙箱复用多条命令。"""

    def __init__(
        self,
        spec: EnvSpec,
        logger: logging.Logger,
        timeout: int,
        num_cpus: int,
        log_dir: Path,
        files_to_copy: Optional[CopyFiles] = None,
        files_to_collect: Optional[list[str]] = None,
        rebuild_image: bool = False,
    ):
        from e2b_code_interpreter import Sandbox

        super().__init__(
            spec,
            logger,
            timeout,
            num_cpus,
            log_dir,
            files_to_copy=files_to_copy,
            files_to_collect=files_to_collect,
        )

        # 沙箱按最长一小时续命
        self.sb = Sandbox(timeout=60 * 60)
        self.sb.commands.run("curl -LsSf https://astral.sh/uv/install.sh | sh")

        # 在沙箱里完成环境安装
        self.sb.files.write("setup.sh", spec.setup_script)
        self.sb.commands.run("bash setup.sh")

        if files_to_copy:
            for _, f in files_to_copy.items():
                with open(f["src"], "r") as fp:  # type: ignore
                    content = fp.read()
                    self.sb.files.write(f["dest"].name, content)  # type: ignore

    def exec_run_with_timeout(self, command: str) -> tuple[str, bool, float]:
        """在 E2B 沙箱执行命令；用沙箱存活状态近似判断是否超时。"""
        start_time = time.time()
        result = self.sb.commands.run(command, timeout=self.timeout)
        if self.files_to_collect is not None:
            for fname in self.files_to_collect:
                with (self.log_dir / fname).open("w") as f:
                    f.write(self.sb.files.read(f"testbed/{fname}"))
        timed_out = not self.sb.is_running()
        end_time = time.time()
        return result.stderr, timed_out, end_time - start_time

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        self.sb.kill()
        release_logger(self.logger)


class HostContext(ExecContext):
    """宿主机后端：不需要 Docker，在临时目录里评测本地任务仓库。

    约定：``spec.repo_directory`` 由调用方预先创建（经 ``build_spec`` 的
    ``repo_directory`` 参数传入），本上下文在其中检出任务的工作提交、
    创建独立虚拟环境、执行评测脚本并回收结果文件，退出时清理临时目录。
    """

    def __init__(
        self,
        spec: EnvSpec,
        logger: logging.Logger,
        timeout: int,
        num_cpus: int,
        log_dir: Path,
        files_to_copy: Optional[CopyFiles] = None,
        files_to_collect: Optional[list[str]] = None,
        rebuild_image: bool = False,
    ):
        super().__init__(
            spec,
            logger,
            timeout,
            num_cpus,
            log_dir,
            files_to_copy=files_to_copy,
            files_to_collect=files_to_collect,
        )
        self.repo_dir = Path(spec.repo_directory)
        self.work_dir = self.repo_dir.parent
        self.venv_dir = self.repo_dir / ".venv"
        self._repo_origin = self.spec.instance.get("local_path")
        self._base_commit = self.spec.instance.get("base_commit")
        # 锁文件与基准 venv 同目录（tasks/{name}/requirements.lock）
        host_venv = self.spec.instance.get("host_venv")
        self._lock_file = (
            str(Path(host_venv).parent / "requirements.lock") if host_venv else None
        )

        self._prepare_worktree()
        if files_to_copy:
            for _, f in files_to_copy.items():
                shutil.copyfile(f["src"], self.work_dir / Path(f["dest"]).name)  # type: ignore
        self._prepare_venv()

    def _prepare_worktree(self) -> None:
        """从任务的本地仓库克隆一份到工作目录并检出骨架提交。

        用独立克隆而不是 git worktree：同一任务的多次评测并发执行时，
        worktree 需要改写源仓库的元数据，容易争用锁文件。
        """
        import git

        if not self._repo_origin or not self._base_commit:
            raise ValueError("HOST 后端要求记录带 local_path 与 base_commit 字段")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        git.Repo.clone_from(self._repo_origin, str(self.repo_dir), no_checkout=True)
        git.Repo(self.repo_dir).git.checkout(self._base_commit)
        self.logger.info(f"任务仓库已克隆到 {self.repo_dir}")

    def _prepare_venv(self) -> None:
        """按锁文件在临时目录里建独立虚拟环境，避免并发评测互相踩踏。"""
        python = self.spec.instance["setup"].get("python", "3.11")
        subprocess.run(
            ["uv", "venv", "--python", str(python), str(self.venv_dir)],
            check=True,
            capture_output=True,
        )
        env = dict(os.environ, VIRTUAL_ENV=str(self.venv_dir))
        if self._lock_file and Path(self._lock_file).exists():
            subprocess.run(
                ["uv", "pip", "install", "-r", self._lock_file],
                check=True,
                capture_output=True,
                env=env,
            )

    def exec_run_with_timeout(self, command: str) -> tuple[str, bool, float]:
        """在宿主机上执行评测脚本（独立进程组，超时时整组终止）。"""
        start_time = time.time()
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=str(self.work_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _ = proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            output, _ = proc.communicate()
        if self.files_to_collect:
            for fname in self.files_to_collect:
                src = self.repo_dir / fname
                if src.exists():
                    shutil.copyfile(src, self.log_dir / fname)
        end_time = time.time()
        return output or "", timed_out, end_time - start_time

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        try:
            shutil.rmtree(self.work_dir, ignore_errors=True)
        finally:
            release_logger(self.logger)


__all__ = []
