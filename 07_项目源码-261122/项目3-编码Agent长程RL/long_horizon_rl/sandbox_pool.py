"""沙箱容器池：按任务预热容器、写时复制快照启动、丢弃可写层复位与官方镜像判分容器（3.5 节）。"""

from __future__ import annotations

import logging
import os
import queue
import shlex
import shutil
import subprocess
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from rebuild_spec.core.constants import ABSOLUTE_REPO_DIR
from rebuild_spec.core.containers import (
    cleanup_container,
    copy_from_container,
    copy_to_container,
    create_container,
    exec_run_with_timeout,
    image_exists_locally,
)
from rebuild_spec.core.specs import build_spec

logger = logging.getLogger(__name__)


_EXIT_MARKER = "__COMMAND_EXIT_CODE__="


def _split_exit_marker(output: str) -> tuple[str, int, bool]:
    """从容器命令输出末尾解析退出码标记，返回 ``(output, exit_code, timed_out)``。"""
    text = output or ""
    pos = text.rfind(_EXIT_MARKER)
    if pos < 0:
        return text, -1, False
    code_text = text[pos + len(_EXIT_MARKER):].strip().splitlines()[0] if text[pos + len(_EXIT_MARKER):].strip() else ""
    try:
        code = int(code_text)
    except ValueError:
        code = -1
    return text[:pos].rstrip("\n"), code, False


@dataclass
class ContainerLease:
    """借出给单条 rollout 轨迹的容器或隔离工作区句柄。"""

    task_id: str
    lease_id: str
    image_name: str
    container: Any = None
    workspace_dir: Optional[Path] = None
    venv_bin: Optional[Path] = None
    from_pool: bool = False
    is_docker: bool = True
    env_vars: dict[str, str] = field(default_factory=dict)
    forbidden_dirs: list[Path] = field(default_factory=list)


class SandboxPool:
    """维护每个任务 ``pool_size_per_task`` 个预热容器的沙箱池。

    - 预热：根据下一步任务清单提前在后台从依赖快照镜像启动容器，并完成 Python
      解释器与包依赖导入健康检查；
    - 启动：rollout 借出容器，在独立工作区 ``/testbed`` 与隔离临时目录 ``/tmp/run_{id}``
      中执行工具命令（只读依赖层 + 写时复制可写层）；
    - 复位：轨迹结束后直接销毁容器丢弃可写层，并异步补齐预热池，避免状态泄漏；
    - 判分：严格在基于官方镜像新建的独立容器中执行，不复用 rollout 容器。
    """

    def __init__(
        self,
        *,
        pool_size_per_task: int = 10,
        image_prefix: str = "lhrl.task",
        nano_cpus: int = 2_000_000_000,
        mem_limit: str = "4g",
        use_docker: bool = True,
        max_prewarm_workers: int = 8,
        docker_client: Any = None,
    ):
        self.pool_size_per_task = pool_size_per_task
        self.image_prefix = image_prefix
        self.nano_cpus = nano_cpus
        self.mem_limit = mem_limit
        self.use_docker = use_docker
        self._client = docker_client
        self._pools: dict[str, queue.Queue[ContainerLease]] = {}
        self._task_records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_prewarm_workers, thread_name_prefix="sandbox-prewarm")
        self._closed = False
        # 每个任务正在创建中的容器数，补池时与队列深度一起计入，避免并发补池超额
        self._creating: dict[str, int] = {}

    def _get_docker_client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    def snapshot_image_name(self, task_id: str, task_record: Optional[dict[str, Any]] = None) -> str:
        """返回预构建的只含锁版本依赖的快照镜像名。"""
        if task_record and task_record.get("snapshot_image"):
            return str(task_record["snapshot_image"])
        safe_id = task_id.lower().replace("/", "_").replace(":", "_")
        return f"{self.image_prefix}.{safe_id}:deps"

    def official_image_name(self, task_record: dict[str, Any]) -> str:
        """返回用于独立判分的官方任务镜像名（来自项目二 EnvSpec 口径）。"""
        if task_record.get("official_image"):
            return str(task_record["official_image"])
        spec = build_spec(task_record, "scratch", absolute=True)
        return spec.repo_image_key

    def build_dependency_snapshot_image(
        self,
        task_record: dict[str, Any],
        *,
        rebuild: bool = False,
    ) -> str:
        """离线构建只含锁版本依赖的分层快照镜像（工作区为空目录）。"""
        task_id = task_record["instance_id"]
        image_tag = self.snapshot_image_name(task_id, task_record)
        if not self.use_docker:
            return image_tag

        client = self._get_docker_client()
        repo_name, tag = image_tag.split(":", 1)
        if not rebuild and image_exists_locally(client, repo_name, tag, logger):
            return image_tag

        base_image = self.official_image_name(task_record)
        python_ver = str((task_record.get("setup") or {}).get("python", "3.11"))
        pip_pkgs = (task_record.get("setup") or {}).get("pip_packages") or []
        pip_line = " ".join(shlex.quote(str(p)) for p in pip_pkgs)
        install_extra = f"RUN uv pip install {pip_line}\n" if pip_line else ""
        dockerfile = (
            f"FROM {base_image}\n"
            f"RUN mkdir -p {ABSOLUTE_REPO_DIR} /tmp/pip-cache && "
            f"find {ABSOLUTE_REPO_DIR} -mindepth 1 -maxdepth 1 ! -name .venv -exec rm -rf {{}} +\n"
            f"ENV PIP_CACHE_DIR=/tmp/pip-cache\n"
            f"RUN test -d {ABSOLUTE_REPO_DIR}/.venv || uv venv --python {shlex.quote(python_ver)} {ABSOLUTE_REPO_DIR}/.venv\n"
            f"ENV VIRTUAL_ENV={ABSOLUTE_REPO_DIR}/.venv\n"
            f"ENV PATH={ABSOLUTE_REPO_DIR}/.venv/bin:$PATH\n"
            f"{install_extra}"
            f"WORKDIR {ABSOLUTE_REPO_DIR}\n"
        )
        with tempfile.TemporaryDirectory(prefix="lhrl-img-") as build_dir:
            df_path = Path(build_dir) / "Dockerfile"
            df_path.write_text(dockerfile, encoding="utf-8")
            client.images.build(path=build_dir, tag=image_tag, rm=True)
        return image_tag

    def _verify_container_health(self, container: Any, task_record: dict[str, Any]) -> bool:
        """对预热容器执行解释器与关键依赖导入检查。"""
        src_dir = task_record.get("src_dir", "")
        pkg_hint = Path(src_dir).name if src_dir else ""
        check_cmd = "python -c 'import sys, pytest; print(sys.version_info[:2])'"
        if pkg_hint:
            # 确认锁文件中的常见第三方依赖可用，不导入尚未实现的目标包自身
            check_cmd = "bash -c 'source /testbed/.venv/bin/activate 2>/dev/null; python -c \"import sys, pytest\"'"
        exit_code, _ = container.exec_run(check_cmd)
        return exit_code == 0

    def _create_prewarmed_lease(self, task_id: str, task_record: dict[str, Any]) -> ContainerLease:
        """从任务快照镜像启动一个干净容器，准备待实现工作区并完成健康检查。"""
        lease_id = uuid.uuid4().hex[:12]
        image_name = self.snapshot_image_name(task_id, task_record)

        if not self.use_docker:
            from agent_rollout.run_loop import prepare_workspace

            repo_dir, workspace = prepare_workspace(task_record)
            return ContainerLease(
                task_id=task_id,
                lease_id=lease_id,
                image_name=image_name,
                container=None,
                workspace_dir=repo_dir,
                venv_bin=workspace.venv_bin,
                from_pool=True,
                is_docker=False,
                forbidden_dirs=list(workspace.forbidden_dirs),
            )

        client = self._get_docker_client()
        container_name = f"lhrl.pool.{task_id}.{lease_id}".lower().replace("/", "_")
        tmp_dir = f"/tmp/run_{lease_id}"
        pip_cache = f"/tmp/pip_cache_{lease_id}"
        container = client.containers.run(
            image=image_name,
            name=container_name,
            command="tail -f /dev/null",
            detach=True,
            nano_cpus=self.nano_cpus,
            mem_limit=self.mem_limit,
            environment={
                "TMPDIR": tmp_dir,
                "PIP_CACHE_DIR": pip_cache,
            },
        )
        try:
            container.exec_run(f"mkdir -p {tmp_dir} {pip_cache} {ABSOLUTE_REPO_DIR}")
            # 把待实现版本拷贝进空白工作区（不含参考提交历史与隐藏测试）
            local_repo = task_record.get("local_path")
            if local_repo and Path(local_repo).exists():
                self._sync_skeleton_into_container(container, Path(local_repo), task_record)
            if not self._verify_container_health(container, task_record):
                raise RuntimeError(f"预热容器健康检查失败: {task_id}")
        except Exception:
            cleanup_container(client, container, logger)
            raise

        return ContainerLease(
            task_id=task_id,
            lease_id=lease_id,
            image_name=image_name,
            container=container,
            workspace_dir=Path(ABSOLUTE_REPO_DIR),
            from_pool=True,
            is_docker=True,
            env_vars={"TMPDIR": tmp_dir, "PIP_CACHE_DIR": pip_cache},
        )

    @staticmethod
    def _sync_skeleton_into_container(container: Any, local_repo: Path, task_record: dict[str, Any]) -> None:
        """在容器内检出无历史的待实现版本并执行可编辑安装。"""
        with tempfile.TemporaryDirectory(prefix="lhrl-skel-") as tmp:
            bundle = Path(tmp) / "skeleton.bundle"
            base_branch = task_record.get("base_branch", "base")
            subprocess.run(
                ["git", "-C", str(local_repo), "bundle", "create", str(bundle), base_branch],
                check=True,
                capture_output=True,
            )
            copy_to_container(container, bundle, Path("/tmp/skeleton.bundle"))
            # /testbed 中保留依赖环境 .venv，因此先克隆到临时目录再拷入工作区
            cmd = (
                f"bash -c 'rm -rf /tmp/workspace_clone && "
                f"git clone /tmp/skeleton.bundle /tmp/workspace_clone && "
                f"git -C /tmp/workspace_clone remote remove origin && "
                f"cp -a /tmp/workspace_clone/. {ABSOLUTE_REPO_DIR}/ && "
                f"rm -rf /tmp/workspace_clone /tmp/skeleton.bundle && cd {ABSOLUTE_REPO_DIR} && "
                f"source .venv/bin/activate && "
                f"uv pip install -e . --no-deps'"
            )
            exit_code, out = container.exec_run(cmd)
            if exit_code != 0:
                raise RuntimeError(f"容器工作区同步失败: {out}")

    def _get_queue(self, task_id: str) -> queue.Queue[ContainerLease]:
        with self._lock:
            if task_id not in self._pools:
                self._pools[task_id] = queue.Queue()
            return self._pools[task_id]

    def prewarm_task(self, task_record: dict[str, Any], target_count: Optional[int] = None) -> int:
        """把指定任务的预热池补齐到 ``target_count`` 个容器，返回新创建个数。"""
        task_id = task_record["instance_id"]
        with self._lock:
            self._task_records[task_id] = task_record
        q = self._get_queue(task_id)
        target = self.pool_size_per_task if target_count is None else target_count
        # 计算缺口与预留名额在同一把锁内完成，并计入正在创建的容器
        with self._lock:
            in_flight = self._creating.get(task_id, 0)
            deficit = max(0, target - q.qsize() - in_flight)
            self._creating[task_id] = in_flight + deficit
        created = 0
        try:
            for _ in range(deficit):
                if self._closed:
                    break
                try:
                    lease = self._create_prewarmed_lease(task_id, task_record)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("预热容器失败 task=%s: %s", task_id, exc)
                    continue
                with self._lock:
                    self._creating[task_id] = max(0, self._creating.get(task_id, 0) - 1)
                    closed = self._closed
                    if not closed:
                        q.put(lease)
                if closed:
                    # 池已关闭：创建完成的容器直接销毁，不再入队
                    self.release(lease, replenish=False)
                    break
                created += 1
        finally:
            with self._lock:
                self._creating[task_id] = max(0, self._creating.get(task_id, 0) - (deficit - created))
        return created

    def prewarm_tasks_async(self, task_records: list[dict[str, Any]]) -> None:
        """在后台并行预热下一批任务的沙箱容器（与当前训练步重叠）。"""
        for record in task_records:
            if self._closed:
                break
            self._executor.submit(self.prewarm_task, record)

    def acquire(self, task_record: dict[str, Any]) -> ContainerLease:
        """从预热池取出一个容器；池空时即时从快照镜像新建。"""
        task_id = task_record["instance_id"]
        with self._lock:
            self._task_records[task_id] = task_record
        q = self._get_queue(task_id)
        try:
            lease = q.get_nowait()
            return lease
        except queue.Empty:
            lease = self._create_prewarmed_lease(task_id, task_record)
            lease.from_pool = False
            return lease

    def release(self, lease: ContainerLease, *, replenish: bool = True) -> None:
        """丢弃可写层完成复位，并在后台补齐一个全新预热容器。"""
        if lease.is_docker and lease.container is not None:
            client = self._get_docker_client()
            try:
                cleanup_container(client, lease.container, logger)
            except Exception as exc:  # noqa: BLE001
                logger.warning("销毁容器失败 lease=%s: %s", lease.lease_id, exc)
        elif lease.workspace_dir is not None:
            shutil.rmtree(lease.workspace_dir.parent, ignore_errors=True)

        if replenish and not self._closed:
            record = self._task_records.get(lease.task_id)
            if record is not None:
                self._executor.submit(self.prewarm_task, record)

    def exec_in_lease(self, lease: ContainerLease, command: str, timeout_s: int = 300) -> tuple[str, int, bool]:
        """在租借容器或工作区内执行命令，返回 ``(output, exit_code, timed_out)``。"""
        if not lease.is_docker:
            from agent_rollout.tools import Workspace, run_bash

            ws = Workspace(root=lease.workspace_dir, venv_bin=lease.venv_bin, forbidden_dirs=list(lease.forbidden_dirs))
            for forbidden in ws.forbidden_dirs:
                if str(Path(forbidden).resolve()) in command or "hidden_tests" in command:
                    return "不允许访问该路径", 1, False
            # 按配置超时限制调用
            proc = subprocess.Popen(
                ["bash", "-c", command],
                cwd=str(ws.root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=ws.bash_env(),
                start_new_session=True,
            )
            try:
                output, _ = proc.communicate(timeout=timeout_s)
                return output or "", int(proc.returncode or 0), False
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), 9)
                output, _ = proc.communicate()
                return output or "", -1, True

        # 在当前 shell 中激活依赖环境，并在输出末尾附加退出码标记
        inner = f"cd {ABSOLUTE_REPO_DIR} && {{ source .venv/bin/activate 2>/dev/null || true; }} && {{ {command}\n}}; echo {_EXIT_MARKER}$?"
        wrapped = f"bash -c {shlex.quote(inner)}"
        output, timed_out, _ = exec_run_with_timeout(lease.container, wrapped, timeout=timeout_s)
        if timed_out:
            return output, -1, True
        return _split_exit_marker(output)

    def create_scoring_container(self, task_record: dict[str, Any], candidate_patch: Path) -> Any:
        """按 3.2 与 3.5 节要求，基于官方镜像新建一个全新判分容器（不复用 rollout 容器）。"""
        client = self._get_docker_client()
        official_image = self.official_image_name(task_record)
        run_id = uuid.uuid4().hex[:12]
        container_name = f"lhrl.score.{task_record['instance_id']}.{run_id}".lower().replace("/", "_")
        container = create_container(
            client=client,
            image_name=official_image,
            container_name=container_name,
            logger=logger,
            nano_cpus=self.nano_cpus,
        )
        container.start()
        copy_to_container(container, candidate_patch, Path("/patch.diff"))
        return container

    def collect_from_container(self, container: Any, remote_path: Path, local_dst: Path) -> None:
        copy_from_container(container, remote_path, local_dst)

    def destroy_container(self, container: Any) -> None:
        cleanup_container(self._get_docker_client(), container, logger)

    def pool_depth(self, task_id: str) -> int:
        q = self._pools.get(task_id)
        return q.qsize() if q is not None else 0

    def close(self) -> None:
        """关闭线程池并销毁所有尚未借出的预热容器。"""
        self._closed = True
        # 等待正在创建的容器结束：创建完成后发现已关闭会自行销毁
        self._executor.shutdown(wait=True, cancel_futures=True)
        for task_id, q in list(self._pools.items()):
            while not q.empty():
                try:
                    lease = q.get_nowait()
                    self.release(lease, replenish=False)
                except queue.Empty:
                    break
            logger.debug("已清理任务 %s 的预热容器池", task_id)
