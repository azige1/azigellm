"""命令执行沙盒：本地 shell 与 Docker 容器两种后端。

两层防护：
1. 命令黑名单正则——阻断引入外部依赖的写法（npm 包 / crates），
   让 agent 只能用目标语言的标准库完成任务；
2. Docker 容器以 --network none 启动，从环境上断网。
"""

import os
import re
import subprocess

from . import config

# 阻断「引用外部 npm 包」的命令写法（允许相对路径 ./ 与绝对路径 /）
NPM_IMPORT_PATTERNS = [
    r"require\s*\(\s*['\"](?![./]).*?['\"]\s*\)",
    r"import\s+.*?from\s+['\"](?![./]).*?['\"]",
    r"import\s*\(\s*['\"](?![./]).*?['\"]\s*\)",
    r"import\s+['\"](?![./]).*?['\"]",
]

# 阻断「引入外部 Rust crate」的命令写法
CARGO_DEP_PATTERNS = [
    r"cargo\s+add\s+\S+",
    r"cargo\s+install\s+\S+",
    r"(echo|printf).*['\"]\s*\w+\s*=\s*['\"][\d\.\w-]+['\"].*\>>?.*Cargo\.toml",
    r"(echo|printf).*\b(\w+)\s*=\s*[\"\']\d[\d\.\w-]*[\"\'].*>>?.*Cargo\.toml",
]

NPM_BLOCK_MSG = (
    "Blocked: loading external NPM packages is not allowed. "
    "Use only relative (./) or absolute (/) local file paths."
)
CARGO_BLOCK_MSG = (
    "Blocked: installing or using external Rust crates is not allowed. "
    "Use only std and local files."
)


def hit_blocked_pattern(command, patterns):
    """命令命中任一黑名单正则则返回 True。"""
    return any(re.search(p, command, re.IGNORECASE) for p in patterns)


def run_local_shell(command, blocked_patterns=None, block_msg=None, timeout=60):
    """在宿主机执行 shell 命令，返回合并了 stdout/stderr 的观测字符串。"""
    print(f"\n[shell] {command}")

    if blocked_patterns and hit_blocked_pattern(command, blocked_patterns):
        print(block_msg)
        return block_msg

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, env=os.environ.copy(),
        )
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except Exception as exc:
        return f"Error: {exc}"


def strip_cargo_dependencies(cargo_toml_path):
    """清空 Cargo.toml 的 [dependencies] 段。

    agent 若在产物里声明了外部依赖，直接抹掉并返回提示语；未违规则返回 None。
    """
    if not cargo_toml_path or not os.path.exists(cargo_toml_path):
        return None
    try:
        with open(cargo_toml_path, "r", encoding="utf-8") as f:
            content = f.read()
        if not re.search(r"^\s*\[dependencies\]", content, re.MULTILINE):
            return None
        dep_match = re.search(
            r"^\s*\[dependencies\](.*?)(?=^\s*\[|\Z)", content, re.MULTILINE | re.DOTALL
        )
        if not dep_match:
            return None
        dep_section = re.sub(r"#.*$", "", dep_match.group(1), flags=re.MULTILINE)
        if not re.search(r"^\s*\w+\s*=\s*[\"']", dep_section, re.MULTILINE):
            return None
        new_content = re.sub(
            r"^\s*\[dependencies\].*?(?=^\s*\[|\Z)", "[dependencies]",
            content, flags=re.MULTILINE | re.DOTALL,
        )
        with open(cargo_toml_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return "External dependencies are not allowed. Use only the Rust standard library (std)."
    except Exception as exc:
        print(f"[WARN] 检查 Cargo.toml 失败: {exc}")
        return None


def docker_run(cmd, timeout=60):
    """执行一条 docker 相关命令，返回 (是否成功, stdout, stderr)。"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Command timeout: {cmd}"
    except Exception as exc:
        return False, "", str(exc)


def docker_cp(src_path, container_name, dst_path, timeout=60):
    """docker cp 的封装，用参数列表形式避免 shell 转义问题。"""
    try:
        result = subprocess.run(
            ["docker", "cp", src_path, f"{container_name}:{dst_path}"],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "Command timeout"
    except Exception as exc:
        return False, "", str(exc)


def ensure_image(image_name):
    """确认镜像在本地存在，不存在则尝试 docker pull。"""
    ok, stdout, _ = docker_run(
        f"docker images --format '{{{{.Repository}}}}:{{{{.Tag}}}}' | grep '^{image_name}$'",
        timeout=10,
    )
    if ok and image_name in stdout:
        print(f"[OK] 镜像 {image_name} 已在本地")
        return True
    print(f"[PULL] 本地没有镜像 {image_name}，尝试拉取...")
    ok, _, stderr = docker_run(f"docker pull {image_name}", timeout=300)
    if ok:
        print(f"[OK] 镜像 {image_name} 拉取成功")
        return True
    print(f"[ERROR] 镜像 {image_name} 拉取失败: {stderr}")
    return False


class Container:
    """一个断网容器的生命周期管理：创建、执行、拷入拷出、销毁。"""

    def __init__(self, name, image, working_dir=None):
        self.name = name
        self.image = image
        self.working_dir = working_dir or config.CONTAINER_WORKSPACE

    @classmethod
    def start(cls, name, image):
        """以断网方式后台启动容器，失败返回 None。"""
        if not ensure_image(image):
            return None
        # 同名残留容器先删掉
        ok, stdout, _ = docker_run(
            f"docker ps -a --filter name={name} --format '{{{{.Names}}}}'"
        )
        if ok and name in stdout:
            print(f"[CLEANUP] 删除同名残留容器 {name}")
            docker_run(f"docker rm -f {name}")

        cmd = (
            f"docker run -d --name {name} "
            f"--network none "
            f"-w {config.CONTAINER_WORKSPACE} "
            f"{image} tail -f /dev/null"
        )
        ok, _, stderr = docker_run(cmd, timeout=30)
        if not ok:
            print(f"[ERROR] 容器启动失败: {stderr}")
            return None

        # 网络隔离自检：只剩回环网卡
        ok, stdout, _ = docker_run(
            f"docker exec {name} cat /sys/class/net/lo/iflink", timeout=10
        )
        if ok and stdout.strip() == "1":
            print(f"[OK] 容器 {name} 网络隔离正常（仅 lo）")

        print(f"[OK] 容器已创建: {name}")
        return cls(name, image)

    def is_alive(self):
        """确认容器存在且处于运行状态。"""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format={{.State.Running}}", self.name],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"容器 '{self.name}' 不存在或不可访问")
            if result.stdout.strip() != "true":
                raise RuntimeError(f"容器 '{self.name}' 不在运行中")
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("容器状态检查超时，请确认 Docker 服务已启动") from exc
        except FileNotFoundError as exc:
            raise RuntimeError("找不到 docker 命令，请先安装 Docker") from exc

    def exec(self, command, blocked_patterns=None, block_msg=None, timeout=60):
        """在容器内执行 shell 命令，返回观测字符串。"""
        print(f"\n[container shell] {command}")
        if blocked_patterns and hit_blocked_pattern(command, blocked_patterns):
            print(block_msg)
            return block_msg
        docker_cmd = ["docker", "exec", "-w", self.working_dir, self.name, "/bin/sh", "-c", command]
        try:
            result = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=timeout)
            return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        except subprocess.TimeoutExpired:
            return f"Error: Command execution timeout ({timeout}s)"
        except Exception as exc:
            return f"Error: {exc}"

    def copy_in(self, host_path, container_path, timeout=30):
        """把宿主机文件拷进容器。"""
        return docker_cp(host_path, self.name, container_path, timeout=timeout)

    def copy_out(self, container_path, host_path, timeout=30):
        """把容器内目录内容拷回宿主机。"""
        return docker_run(
            f"docker cp {self.name}:{container_path} {host_path}", timeout=timeout
        )

    def remove(self):
        """销毁容器。"""
        docker_run(f"docker rm -f {self.name}")
