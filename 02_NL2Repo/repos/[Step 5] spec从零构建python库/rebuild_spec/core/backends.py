"""评测执行后端：本地 Docker 容器、Modal 沙箱、E2B 沙箱。

三个后端共享同一套接口（``ExecContext``），上层只关心
``exec_run_with_timeout`` 与上下文管理器协议。
"""

from abc import ABC, abstractmethod
import docker
import logging
import modal
import modal.io_streams
import os
from enum import auto
from e2b_code_interpreter import Sandbox
from strenum import StrEnum
from pathlib import Path
import time
from typing import Optional, Type
from types import TracebackType

from rebuild_spec.core.constants import CopyFiles
from rebuild_spec.core.specs import EnvSpec
from rebuild_spec.core.gitops import release_logger
from rebuild_spec.core.containers import (
    cleanup_container,
    create_container,
    copy_from_container,
    copy_to_container,
    exec_run_with_timeout,
)


class ExecBackend(StrEnum):
    LOCAL = auto()
    MODAL = auto()
    E2B = auto()


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

    def exec_run_with_timeout(self, command: str) -> tuple[str, bool, float]:
        output = exec_run_with_timeout(self.container, command, self.timeout)

        if self.files_to_collect:
            for fname in self.files_to_collect:
                file = Path(self.spec.repo_directory) / fname
                # 先确认文件存在再取回
                exit_code, test_output = self.container.exec_run(
                    f"test -e {file}", demux=True
                )
                if exit_code == 0:
                    copy_from_container(self.container, file, self.log_dir / fname)
        return output

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
        cleanup_container(self.client, self.container, self.logger)
        release_logger(self.logger)


class ModalContext(ExecContext):
    """Modal 云沙箱后端：镜像需先推送到公开镜像仓库。"""

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

        self.app = modal.App.lookup("rebuild-spec", create_if_missing=True)

        # 镜像必须已经存在于镜像仓库（registry 由环境变量指定）
        registry = os.environ.get("REBUILD_SPEC_REGISTRY", "your-registry")
        reponame = spec.repo.split("/")[-1]
        image_name = f"{registry}/{reponame}:v0".lower()
        image = modal.Image.from_registry(image_name, force_build=rebuild_image)
        if files_to_copy:
            for _, f in files_to_copy.items():
                image = image.add_local_file(str(f["src"]), str(f["dest"]))  # type: ignore
        self.image = image

    def exec_run_with_timeout(self, command: str) -> tuple[str, bool, float]:
        """在 Modal 沙箱中执行命令，结果文件经临时卷取回。"""
        start_time = time.time()
        with modal.Volume.ephemeral() as vol:
            if self.files_to_collect:
                command += " && "
                for fname in self.files_to_collect:
                    remote_file = Path(self.spec.repo_directory) / fname
                    cp_cmd = f"test -e {str(remote_file)} && cp {str(remote_file)} /vol/{fname}; "
                    command += cp_cmd
            self.sandbox = modal.Sandbox.create(
                "bash",
                "-c",
                command,
                image=self.image,
                cpu=self.num_cpus,
                timeout=self.timeout,
                app=self.app,
                volumes={"/vol": vol},
            )
            self.sandbox.wait()

            return_code = self.sandbox.returncode
            # 124 是 timeout 命令的约定退出码
            if return_code == 124:
                timed_out = True
            else:
                timed_out = False

            if self.files_to_collect:
                fnames = vol.listdir("")
                for fname in fnames:
                    fname = fname.path
                    with (self.log_dir / fname).open("wb") as f:
                        for data in vol.read_file(fname):
                            f.write(data)

            self.sandbox.terminate()
            end_time = time.time()
            return self.sandbox.stderr.read(), timed_out, end_time - start_time

    def __exit__(
        self,
        exctype: Optional[Type[BaseException]],
        excinst: Optional[BaseException],
        exctb: Optional[TracebackType],
    ) -> None:
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
