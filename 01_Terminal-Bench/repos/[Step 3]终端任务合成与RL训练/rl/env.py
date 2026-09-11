"""SkyRL 环境封装：把 Harbor 任务目录变成可训练的文本环境。

每个 episode 懒加载地构建任务 Docker 镜像并启动容器；agent 的文本动作
按 ``<command>`` / ``<action>done</action>`` 协议解析后在容器内执行；
episode 结束时在容器内跑终态测试，通过则 reward += 1。
"""
import subprocess
import sys
import pathlib
import tempfile
import uuid
from pathlib import Path
import re
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput

sys.path.insert(0, str(pathlib.Path().resolve()))
from task_factory.rollout_agent import parse_agent_action
import time
import threading

# 限制并发容器数，防止资源耗尽
_CONTAINER_LOCK = threading.Lock()
_MAX_CONTAINERS = 10


class DockerTaskSandbox:
    """训练用的 Docker 容器沙箱：构建镜像、起容器、执行命令、跑终态测试。"""

    def __init__(self, dockerfile_path, final_test_path, verbose=False):
        self.dockerfile_path = Path(dockerfile_path)
        self.final_test_path = Path(final_test_path)
        self.verbose = verbose
        self.image_tag = None
        self.container_name = None
        self.instance_name = None  # None 表示未运行

    def initialize(self, run_initial_tests=False):
        tag = f"rl-task-{self.dockerfile_path.parent.parent.name}-{uuid.uuid4().hex[:8]}"
        try:
            proc = subprocess.run(
                ["docker", "build", "-t", tag, "-f", str(self.dockerfile_path), str(self.dockerfile_path.parent)],
                env={**__import__("os").environ, "DOCKER_BUILDKIT": "1"},
                capture_output=True, text=True, timeout=600,
            )
            build_ok = proc.returncode == 0
        except subprocess.TimeoutExpired:
            build_ok = False
        if not build_ok:
            if self.verbose:
                print(f"Docker build failed or timed out")
            return False
        self.image_tag = tag

        cname = f"rl-run-{uuid.uuid4().hex[:8]}"
        try:
            proc = subprocess.run(
                ["docker", "run", "-d", "--name", cname, tag, "sleep", "3600"],
                capture_output=True, text=True, timeout=120,
            )
            if proc.returncode != 0:
                if self.verbose:
                    print(f"Docker run failed: {proc.stderr}")
                subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=30)
                return False
        except subprocess.TimeoutExpired:
            # 超时容器必须先删再删镜像，否则会静默失败
            subprocess.run(["docker", "rm", "-f", cname], capture_output=True, timeout=30)
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True, timeout=30)
            return False
        self.container_name = cname
        self.instance_name = cname
        return True

    def exec(self, command, timeout=30):
        try:
            proc = subprocess.run(
                ["docker", "exec", self.container_name, "bash", "-c", command],
                capture_output=True, text=True, timeout=timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout}s"
        except Exception as e:
            return False, str(e)

    def run_final_tests(self):
        test_py = self.final_test_path.read_text(encoding="utf-8")
        subprocess.run(
            ["docker", "exec", self.container_name, "mkdir", "-p", "/tests"],
            capture_output=True, timeout=10,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(test_py)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["docker", "cp", tmp_path, f"{self.container_name}:/tests/test_final_state.py"],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                return False, f"docker cp failed: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "docker cp timed out"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        self.exec("pip3 install -q pytest 2>/dev/null || pip install -q pytest 2>/dev/null", timeout=60)
        return self.exec("cd /home/user && python3 -m pytest /tests/test_final_state.py -v", timeout=120)

    def cleanup(self):
        if self.container_name:
            subprocess.Popen(["docker", "rm", "-f", self.container_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.container_name = None
        if self.image_tag:
            subprocess.Popen(["docker", "rmi", "-f", self.image_tag], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.image_tag = None
        self.instance_name = None


class TerminalTaskEnv(BaseTextEnv):
    """终端任务环境，兼容 SkyRL 的 BaseTextEnv 接口。"""

    def __init__(self, env_config: Dict[str, Any] | None = None, extras: Dict[str, Any] | None = None):
        super().__init__()
        env_config = env_config
        task_dir = extras["extra_info"].get("task_dir")
        task_dir = Path(task_dir)

        dockerfile_path = task_dir / "environment" / "Dockerfile"
        final_test_path = task_dir / "tests" / "test_final_state.py"
        self.start_time = time.time()
        self.max_time = extras["extra_info"].get("max_time", 600)
        # max_time 可能是字符串，统一转 int
        if isinstance(self.max_time, str):
            self.max_time = int(self.max_time)
        self.reward = 0

        self.max_turns = extras.get("max_turns", 16)
        verbose_mode = extras["extra_info"].get("verbose", False)
        # 输出截断上限，防止超长输出撑爆内存
        self.max_output_length = extras["extra_info"].get("max_output_length", 50000)

        self.env = DockerTaskSandbox(
            dockerfile_path=dockerfile_path,
            final_test_path=final_test_path,
            verbose=verbose_mode,
        )
        # 懒初始化：构造时不建容器，避免 Ray actor 延迟创建时在训练阶段外
        # 就启动大量容器
        self._initialized = False

    def __del__(self):
        """对象销毁时尽力清理已初始化的环境。"""
        if hasattr(self, '_initialized') and self._initialized:
            try:
                self.env.cleanup()
            except Exception:
                pass

    def step(self, action: str) -> BaseTextEnvStepOutput:
        # 首次 step 才真正初始化；已清理的环境不可复用
        if not self._initialized or self.env.instance_name is None:
            if self.env.instance_name is None and self._initialized:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "❌ Environment was cleaned up and cannot be reused"}],
                    reward=0.0,
                    done=True,
                    metadata={"goal_reached": False, "env_cleaned_up": True},
                )
            init_success = self.env.initialize(run_initial_tests=False)
            if not init_success:
                return BaseTextEnvStepOutput(
                    observations=[{"role": "user", "content": "❌ Failed to initialize container environment"}],
                    reward=0.0,
                    done=True,
                    metadata={"goal_reached": False, "init_failed": True},
                )
            self._initialized = True

        self.turns += 1
        action = parse_agent_action(action)
        goal_reached = False

        done = False
        if action["type"] == "done":
            done = True
            result_back = "Done"

        elif action["type"] == "command":
            command = action["command"] or ""
            success, output = self.env.exec(command)

            truncated_msg = ""
            if len(output) > self.max_output_length:
                output = output[:self.max_output_length]
                truncated_msg = f"\n[Output truncated: showing first {self.max_output_length} of {len(output)} characters]"

            if success:
                result_back = f"Command executed successfully. Output: {output}{truncated_msg}\n\n(exit_code={0 if success else 1})"
            else:
                result_back = f"Command failed. Output: {output}{truncated_msg}\n\n(exit_code={0 if success else 1})"

        else:
            result_back = "Could not parse a single <command>...</command> or <action>done</action>. Please respond with exactly one of those."
        # 终止条件：轮数用尽或超时
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        timed_out = False

        if self.turns >= self.max_turns:
            done = True
        elif elapsed_time > self.max_time:
            done = True
            timed_out = True

        if self.env.verbose:
            print(f"Time taken so far: {elapsed_time:.2f}s")

        if done:
            if self._initialized:
                if not timed_out:
                    # 超时的 episode 不评分
                    success, test_output = self.env.run_final_tests()
                    goal_reached = success
                    if success:
                        self.reward += 1
                self.env.cleanup()
                # 标记为未初始化，防止被复用
                self._initialized = False

        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": result_back}],
            reward=self.reward,
            done=done,
            metadata={"goal_reached": goal_reached},
        )
