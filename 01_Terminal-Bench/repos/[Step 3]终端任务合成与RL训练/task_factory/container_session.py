"""基于 PTY 的 Apptainer 容器交互会话。

在预构建的 SIF 镜像里启动一个长期存活的 Apptainer 实例，再通过伪终端
（PTY）开一个交互 shell。每条命令执行后输出一个带退出码的唯一标记，
以此确定命令结束位置并读取退出状态。
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import pty
import queue
import re
import shutil
import subprocess
import tempfile
import termios
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")  # 匹配 ANSI 转义序列

# 基础镜像本地路径（.def 里引用 ./ubuntu_22.04.sif 时会被替换为该绝对路径）
BASE_SIF_PATH = os.environ.get("UBUNTU_BASE_SIF", "./ubuntu_22.04.sif")


class ContainerShellSession:
    """Apptainer 容器 + PTY 交互 shell 的封装。"""

    def __init__(
        self,
        container_sif_path: str,
        initial_test_path: str,
        final_test_path: str,
        def_path: str,
        max_actions: int = 50,
        verbose: bool = True,
        read_timeout: float = 10.0,
    ):
        self.sif_path = Path(container_sif_path).expanduser().resolve()
        self.initial_test_path = Path(initial_test_path).expanduser().resolve()
        self.final_test_path = Path(final_test_path).expanduser().resolve()
        self.def_path = Path(def_path).expanduser().resolve()

        self.max_actions = max_actions
        self.verbose = verbose
        self.read_timeout = read_timeout

        self.temp_dir: Optional[Path] = None
        self.action_history: List[Dict[str, str]] = []
        self.instance_name: Optional[str] = None

        self.shell_process: Optional[subprocess.Popen] = None
        self.master_fd: Optional[int] = None
        self.slave_fd: Optional[int] = None

        self.output_queue: "queue.Queue[str]" = queue.Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # 命令结束标记，格式 {marker}:{exit_code}
        self._marker = f"__CMD_DONE__{uuid.uuid4().hex}__"

    # ---------------------------- 底层 PTY 读写 ----------------------------
    def _reader_loop(self) -> None:
        """后台线程：非阻塞读取 PTY 输出并推入队列。"""
        fd = self.master_fd
        if fd is None:
            return
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while (
                not self._stop_event.is_set()
                and self.shell_process
                and self.shell_process.poll() is None
            ):
                try:
                    data = os.read(fd, 16384)
                    if data:
                        text = data.decode("utf-8", errors="replace")
                        self.output_queue.put_nowait(text)
                        continue
                except BlockingIOError:
                    pass
                except OSError as e:
                    if getattr(e, "errno", None) in (errno.EBADF, errno.EIO):
                        break
                    raise
                time.sleep(0.005)
        finally:
            try:
                while True:
                    data = os.read(fd, 16384)
                    if not data:
                        break
                    text = data.decode("utf-8", errors="replace")
                    self.output_queue.put_nowait(text)
            except Exception:
                pass

    def _drain_queue(self) -> str:
        chunks: List[str] = []
        while True:
            try:
                chunks.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return "".join(chunks)

    def _read_until_marker(self, timeout: Optional[float] = None) -> Tuple[str, Optional[int]]:
        """持续读取直到看到结束标记行。

        返回 ``(不含标记的输出, 退出码)``；超时时退出码为 ``None``。
        """
        if timeout is None:
            timeout = self.read_timeout

        deadline = time.time() + timeout
        buf = []

        marker_match = None
        while time.time() < deadline:
            chunk = self._drain_queue()
            if chunk:
                buf.append(chunk)
                joined = "".join(buf)
                # 命令输出里可能恰好含标记文本，取最后一次出现
                for line in joined.splitlines():
                    if self._marker in line:
                        if ":" in line:
                            parts = line.rsplit(":", 1)
                            if len(parts) == 2 and parts[0].endswith(self._marker):
                                try:
                                    code = int(parts[1].strip())
                                except ValueError:
                                    code = None
                                marker_match = (joined, code)
                if marker_match:
                    full_out, code = marker_match
                    first_marker_index = full_out.find(self._marker)
                    cleaned = full_out[:first_marker_index]
                    return cleaned, code
            time.sleep(0.002)

        return "".join(buf), None

    # ---------------------------- shell 生命周期 ----------------------------
    def _start_shell(self) -> bool:
        """在 PTY 上启动交互式 Apptainer shell。"""
        if self.shell_process:
            return True

        self.master_fd, self.slave_fd = pty.openpty()

        # 关闭回显，避免命令被重复显示
        try:
            attrs = termios.tcgetattr(self.slave_fd)
            attrs[3] = attrs[3] & ~termios.ECHO
            termios.tcsetattr(self.slave_fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        cmd = [
            "apptainer", "shell",
            "--containall",
            "--cleanenv",
            "--pwd", "/home/user",
            f"instance://{self.instance_name}",
        ]

        try:
            self.shell_process = subprocess.Popen(
                cmd,
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                close_fds=True,
                start_new_session=True,
            )

            self._stop_event.clear()
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()

            time.sleep(0.05)
            if self.shell_process.poll() is not None:
                if self.verbose:
                    print("Apptainer shell exited early with code:", self.shell_process.returncode)
                leftover = self._drain_queue()
                if leftover:
                    print("Shell start output:\n", leftover)
                return False

            # 初始化 shell：固定提示符、设定 HOME，并打印一次标记确认就绪
            init_script = (
                "set -o pipefail 2>/dev/null; "
                "export PS1='[$PWD]$ '; "
                "export HOME=/home/user; "
                "cd \"$HOME\" 2>/dev/null || true; "
                f"printf '{self._marker}:0\\n'"
            )
            os.write(self.master_fd, (init_script + "\n").encode("utf-8"))
            _, code = self._read_until_marker(timeout=10.0)
            if code is None:
                if self.verbose:
                    print("Shell init timed out.")
                return False

            if self.verbose:
                init_out = self._drain_queue()
                if init_out:
                    print(f"Shell started. Initial output:\n{init_out}")

            return True
        except Exception as e:
            if self.verbose:
                print(f"Failed to start shell: {e}")
            return False

    def _stop_shell(self):
        """停止交互 shell 并关闭 PTY 文件描述符。"""
        try:
            self._stop_event.set()
            if self.reader_thread:
                try:
                    self.reader_thread.join(timeout=1.0)
                except Exception:
                    pass
                self.reader_thread = None

            if self.shell_process and self.shell_process.poll() is None:
                try:
                    os.write(self.master_fd, b"exit\n")
                except Exception:
                    pass
                try:
                    self.shell_process.wait(timeout=2)
                except Exception:
                    self.shell_process.terminate()
                    try:
                        self.shell_process.wait(timeout=2)
                    except Exception:
                        self.shell_process.kill()
        finally:
            self.shell_process = None

            for fd in (self.master_fd, self.slave_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass
            self.master_fd = None
            self.slave_fd = None

    def _stop_instance(self) -> None:
        """停止 Apptainer 实例。"""
        if self.instance_name:
            subprocess.run(
                ["apptainer", "instance", "stop", self.instance_name],
                capture_output=True
            )
            self.instance_name = None

    # ---------------------------- 公开 API ----------------------------
    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def initialize(self, run_initial_tests: bool = True) -> bool:
        """启动容器实例与交互 shell，可选地先跑初始状态测试。"""
        if self.verbose:
            print(f"[setup] Initializing container environment with {self.sif_path.name}...")

        if not self.sif_path.exists():
            if self.verbose:
                print(f"[setup] SIF not found at {self.sif_path}, attempting to build...")
            if self.def_path.exists():
                self.build_container()
            else:
                print("[setup] Neither SIF nor def file exists")
                return False

        self.temp_dir = Path(tempfile.mkdtemp(prefix="agent_env_")).resolve()

        self.instance_name = f"agent_{uuid.uuid4().hex[:8]}"
        start_cmd = [
            "apptainer", "instance", "start",
            "--containall",
            "--writable-tmpfs",
            "--bind", f"{self.temp_dir}:{self.temp_dir}",
            "--cleanenv",
            str(self.sif_path),
            self.instance_name,
        ]
        if self.verbose:
            print(f"[setup] Starting instance with command: {' '.join(start_cmd)}")
        start_proc = subprocess.run(start_cmd, capture_output=True, text=True)
        if start_proc.returncode != 0:
            if self.verbose:
                print(f"[setup] Instance start failed: {start_proc.stdout + start_proc.stderr}")
            return False
        else:
            if self.verbose:
                print(f"[setup] Instance started: {start_proc.stdout + start_proc.stderr}")

        if not self._start_shell():
            if self.verbose:
                print("[setup] Failed to start interactive shell")
            self._stop_instance()
            return False
        else:
            if self.verbose:
                print("[setup] Interactive shell started")

        if run_initial_tests:
            if not self.run_initial_tests():
                if self.verbose:
                    print("[setup] Initial state tests failed")
                self._stop_shell()
                self._stop_instance()
                return False

        if self.verbose:
            print("[setup] Container environment ready")
        self.exec("cd /home/user")
        return True

    def exec(self, command: str, timeout: Optional[float] = None) -> Tuple[bool, str]:
        """在交互 shell 中执行一条命令。

        返回 ``(是否成功, 清理后的输出)``。
        """
        if not self.shell_process:
            if not self._start_shell():
                return False, "Failed to start shell"

        if self.shell_process.poll() is not None:
            if self.verbose:
                print(f"[warn] Shell process died (exit code: {self.shell_process.returncode}), restarting...")
            self.shell_process = None
            if not self._start_shell():
                return False, "Shell process died and restart failed"

        if not self.reader_thread or not self.reader_thread.is_alive():
            if self.verbose:
                print("[warn] Reader thread died, restarting shell...")
            self._stop_shell()
            if not self._start_shell():
                return False, "Reader thread died and shell restart failed"

        _ = self._drain_queue()

        # 包装命令：执行后打印 标记:退出码；heredoc 不能用 {} 分组，单独处理
        if "<<" in command:
            wrapped = f"{command}\ncode=$?; printf '{self._marker}:%s\\n' \"$code\""
        else:
            wrapped = f"{{ {command}; }}; code=$?; printf '{self._marker}:%s\\n' \"$code\""

        try:
            os.write(self.master_fd, (wrapped + "\n").encode("utf-8"))
        except Exception as e:
            return False, f"Command write failed: {e}"

        time.sleep(0.01)
        if self.shell_process.poll() is not None:
            return False, f"Shell died immediately after command (exit code: {self.shell_process.returncode})"

        raw_out, code = self._read_until_marker(timeout=timeout)

        if code is None:
            if self.verbose:
                print(f"[warn] Command timed out after {timeout or self.read_timeout}s")
            return False, f"Command timed out. Partial output:\n{raw_out[:500]}"

        cleaned = ANSI_RE.sub("", raw_out)
        cleaned = cleaned.replace("\r", "")

        success = (code == 0)
        return success, cleaned

    def run_initial_tests(self) -> bool:
        """把初始状态测试写入容器并执行 pytest。"""
        if self.verbose:
            print("[test] Running initial state tests...")

        with open(self.initial_test_path, "r") as f:
            test_file_text = f.read()

        test_path_in_container = "/home/user/test_initial.py"

        max_retries = 1
        retry_delay = 0.1

        for attempt in range(max_retries):
            marker = f"EOF_TEST_FILE_{uuid.uuid4().hex}"
            write_cmd = (
                f"cat <<'{marker}' > {test_path_in_container}\n"
                f"{test_file_text}\n"
                f"{marker}\n"
            )

            success, output = self.exec(write_cmd)
            if success:
                break

            if self.verbose:
                print(f"[warn] Write attempt {attempt + 1}/{max_retries} failed: {output[:100]}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                if self.verbose:
                    print(f"[test] Failed to write test file after {max_retries} attempts: {output}")
                return False

        test_success, test_output = self.exec(f"pytest -q {test_path_in_container}")

        self.exec(f"rm -f {test_path_in_container}")

        if not test_success:
            if self.verbose:
                print(f"Initial test output:\n{test_output}")
        else:
            if self.verbose:
                print("[test] Initial state tests passed")

        return test_success

    def run_final_tests(self) -> Tuple[bool, str]:
        """把终态测试写入容器并执行 pytest，返回 ``(是否通过, 输出)``。"""
        if self.verbose:
            print("[test] Running final state tests...")

        with open(self.final_test_path, "r") as f:
            test_file_text = f.read()

        test_path_in_container = "/home/user/test_final.py"

        max_retries = 1
        retry_delay = 0.1

        for attempt in range(max_retries):
            marker = f"EOF_TEST_FILE_{uuid.uuid4().hex}"
            write_cmd = (
                f"cat <<'{marker}' > {test_path_in_container}\n"
                f"{test_file_text}\n"
                f"{marker}\n"
            )
            ok, write_out = self.exec(write_cmd)

            if ok:
                break

            if self.verbose:
                print(f"[warn] Write attempt {attempt + 1}/{max_retries} failed: {write_out[:100]}")

            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                if self.verbose:
                    print(f"[test] Failed to write final test file after {max_retries} attempts: {write_out}")
                return False, write_out

        test_success, test_output = self.exec(f"pytest -q {test_path_in_container}")

        self.exec(f"rm -f {test_path_in_container}")

        if self.verbose:
            if test_success:
                print("[test] Final state tests passed!")
            else:
                print("[test] Final state tests failed")
                print(test_output)

        return test_success, test_output

    def cleanup(self):
        """释放临时目录、shell 与容器实例。"""
        self._stop_shell()
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self._stop_instance()
        if self.verbose:
            print("[cleanup] Cleaned up temporary files and processes")

    def build_container(self):
        """按 def 文件重新构建 SIF。"""
        with open(self.def_path, "r") as f:
            def_text = f.read()
        def_text = def_text.replace("./ubuntu_22.04.sif", BASE_SIF_PATH)
        with open(self.def_path, "w") as f:
            f.write(def_text)

        if "chmod 755 /home/user" not in def_text:
            # 在 %post 段末尾补一条 home 目录权限修正
            def_lines = [line for line in def_text.split("\n") if "%" in line]
            post_idx = [i for i, line in enumerate(def_lines) if "post" in line.lower()]
            next_section_header = def_lines[post_idx[0] + 1]
            def_text = def_text.replace(
                next_section_header, next_section_header + "\n    chmod 755 /home/user\n"
            )
            with open(self.def_path, "w") as f:
                f.write(def_text)

        build_rc = subprocess.run(
            ["apptainer", "build", str(self.sif_path), str(self.def_path)],
            capture_output=True,
            text=True,
        )
        if build_rc.returncode != 0:
            print(f"Apptainer build failed: {build_rc.stdout + build_rc.stderr}")
            return False
        return build_rc.returncode == 0

    def get_prompt(self) -> str:
        """返回带当前工作目录的提示符字符串（交互模式用）。"""
        success, output = self.exec("pwd")
        if success and output:
            current_dir = output.strip().splitlines()[-1]
            return f"({self.sif_path.name}) {current_dir} $ "
        return f"({self.sif_path.name}) $ "


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-path", type=str, default="tasks/sample_task")
    args = parser.parse_args()
    task_path = Path(args.task_path)
    def_path = task_path / "container.def"
    initial_test_path = task_path / "test_initial_state.py"
    final_test_path = task_path / "test_final_state.py"
    container_sif_path = task_path / "container.sif"

    task_description = ""
    truth = ""
    if Path("sample_task.json").exists():
        with open("sample_task.json", "r") as f:
            task_data = json.load(f)
        task_description = task_data.get("description", "")
        truth = task_data.get("truth", "")

    env = ContainerShellSession(
        container_sif_path=container_sif_path,
        initial_test_path=initial_test_path,
        final_test_path=final_test_path,
        def_path=def_path,
        verbose=True,
    )
    if not container_sif_path.exists():
        env.build_container()

    if not env.initialize(run_initial_tests=True):
        raise SystemExit(1)

    try:
        print("\nStarting interactive session with the container...")
        print("Type 'exit' or 'quit' to finish.")
        if task_description:
            print(f"Task description: {task_description}")

        while True:
            try:
                prompt = env.get_prompt()
                command = input(prompt)
                if command.lower() in ["exit", "quit"]:
                    break
                if not command.strip():
                    continue
                success, output = env.exec(command)
                if output:
                    print(output)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive session.")
                break
    finally:
        env.run_final_tests()
        env.cleanup()
