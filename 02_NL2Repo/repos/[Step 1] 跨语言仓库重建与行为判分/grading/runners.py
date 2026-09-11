"""判分执行后端：把参考实现与生成实现跑起来。

LocalRunner：直接在宿主机调用 python3 / node / 可执行文件。
ContainerRunner：在一个断网容器里跑——每次执行前把文件/目录 docker cp 进去，
按后缀选择解释器（.py -> python3，.mjs/.js -> node，其他视为二进制直接执行）。
"""

import os
import shlex
import subprocess
import uuid

from harness import config, sandbox


class LocalRunner:
    """宿主机直跑。"""

    def run_source(self, source_path, args, timeout=5):
        """跑参考实现（解释器跑源文件）。"""
        return self._run([config.PYTHON_BIN, source_path] + list(args), timeout)

    def run_binary(self, bin_path, args, timeout=3):
        """跑已编译的参考/生成二进制，返回 (stdout, 是否出错)。"""
        if not os.path.exists(bin_path):
            return None, True
        stdout, code = self._run([bin_path] + list(args), timeout)
        return stdout, code != 0

    def run_package(self, pkg_dir, entry_file, args, timeout=5):
        """跑生成物：entry_file 是包内入口文件名。"""
        entry_path = os.path.join(pkg_dir, entry_file)
        if not os.path.exists(entry_path):
            return None, -1
        return self._run([config.NODE_BIN, entry_path] + list(args), timeout)

    @staticmethod
    def _run(cmd_list, timeout):
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True, timeout=timeout)
            return result.stdout, result.returncode
        except Exception:
            return None, -1


class ContainerRunner:
    """断网容器内跑；用 close() 销毁容器。"""

    def __init__(self, image, timeout=5):
        self.timeout = timeout
        name = f"grade-{uuid.uuid4().hex[:8]}"
        self.container = sandbox.Container.start(name, image)
        if not self.container:
            raise RuntimeError(f"无法创建判分容器（镜像 {image}）")

    def close(self):
        if self.container:
            self.container.remove()
            print(f"\n[CLEANUP] 判分容器已销毁: {self.container.name}")
            self.container = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    # -- 内部工具 --

    def _exec(self, command, timeout):
        try:
            result = subprocess.run(
                ["docker", "exec", self.container.name, "/bin/sh", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            return result.stdout, result.returncode
        except Exception:
            return None, -1

    def _stage_file(self, host_path, container_path):
        """清掉旧文件再拷入；二进制补可执行权限。"""
        if not os.path.exists(host_path):
            return False
        self._exec(f"rm -rf {shlex.quote(container_path)}", timeout=10)
        ok, _, _ = self.container.copy_in(host_path, container_path, timeout=10)
        if not ok:
            return False
        if not host_path.endswith((".py", ".mjs", ".js")):
            self._exec(f"chmod +x {shlex.quote(container_path)}", timeout=5)
        return True

    # -- 对外接口（与 LocalRunner 对齐） --

    def run_source(self, source_path, args, timeout=None):
        """跑参考实现：优先用源文件旁的 <模块名>_executable 预编译产物。"""
        timeout = timeout or self.timeout
        src_dir = os.path.dirname(source_path)
        base = os.path.basename(source_path)
        module = base[: -len(".py")] if base.endswith(".py") else base
        executable = os.path.join(src_dir, f"{module}_executable")
        if os.path.exists(executable):
            return self.run_binary(executable, args, timeout)
        container_path = f"{config.CONTAINER_WORKSPACE}/{base}"
        if not self._stage_file(source_path, container_path):
            return None, -1
        return self._exec(
            f"python3 {shlex.quote(container_path)} " + " ".join(map(str, args)), timeout
        )

    def run_binary(self, bin_path, args, timeout=None):
        """把二进制拷进容器直接执行，返回 (stdout, 是否出错)。"""
        timeout = timeout or self.timeout
        container_path = f"{config.CONTAINER_WORKSPACE}/exec_bin"
        if not self._stage_file(bin_path, container_path):
            return None, True
        stdout, code = self._exec(
            shlex.quote(container_path) + " " + " ".join(map(str, args)), timeout
        )
        return stdout, code != 0

    def run_package(self, pkg_dir, entry_file, args, timeout=None):
        """整个产物目录拷进容器，用 node 跑入口文件。"""
        timeout = timeout or self.timeout
        if not os.path.exists(pkg_dir):
            return None, -1
        container_pkg = f"{config.CONTAINER_WORKSPACE}/candidate_pkg"
        self._exec(f"rm -rf {shlex.quote(container_pkg)}", timeout=10)
        ok, _, _ = self.container.copy_in(pkg_dir, container_pkg, timeout=10)
        if not ok:
            return None, -1
        entry = f"{container_pkg}/{entry_file}"
        return self._exec(f"node {shlex.quote(entry)} " + " ".join(map(str, args)), timeout)
