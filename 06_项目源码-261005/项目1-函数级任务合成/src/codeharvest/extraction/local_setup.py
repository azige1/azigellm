"""本地模式依赖隔离：为每个仓库用 uv 建独立虚拟环境并安装依赖。

虚拟环境放在 VENVS_DIR/{repo_id}，不放进仓库目录，避免抽取、切片与 PyCG 扫到 site-packages。
"""

import os
import subprocess

from codeharvest.workspace import REPOS_DIR, VENVS_DIR


def repo_venv_dir(repo_path: str) -> str:
    """仓库对应的虚拟环境目录（按仓库目录名区分）。"""
    repo_id = os.path.basename(os.path.normpath(repo_path))
    return os.path.join(str(VENVS_DIR), repo_id)


def repo_venv_python(repo_path: str) -> str | None:
    """仓库虚拟环境的 python 路径，未创建时返回 None。"""
    venv_dir = repo_venv_dir(repo_path)
    for candidate in (
        os.path.join(venv_dir, "bin", "python"),
        os.path.join(venv_dir, "Scripts", "python.exe"),
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def setup_repo_venv(repo_path: str, python_version: str = "3.11") -> bool:
    """为单个仓库建 venv 并安装依赖、仓库本身、coverage 与 pytest，失败返回 False。"""
    if repo_venv_python(repo_path) is not None:
        return True

    venv_dir = repo_venv_dir(repo_path)
    os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
    try:
        subprocess.run(["uv", "venv", "--python", python_version, venv_dir], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"仓库 {repo_path} 创建 venv 失败（需要预装 uv）: {e!r}")
        return False

    python = repo_venv_python(repo_path)
    if python is None:
        return False

    pip_cmd = ["uv", "pip", "install", "--python", python]

    install_specs = []
    for requirements_name in ("requirements.txt", "requirements-dev.txt", "requirements_dev.txt"):
        requirements = os.path.join(repo_path, requirements_name)
        if os.path.exists(requirements):
            install_specs.append(["-r", requirements])
    if os.path.exists(os.path.join(repo_path, "pyproject.toml")) or os.path.exists(
        os.path.join(repo_path, "setup.py")
    ):
        install_specs.append(["-e", "."])

    for spec in install_specs:
        try:
            subprocess.run(pip_cmd + spec, cwd=repo_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"仓库 {repo_path} 安装 {' '.join(spec)} 失败: {e!r}")

    # 执行沙箱需要 coverage（分支 JSON 报告需 7.x），真实调用采集需要 pytest
    try:
        subprocess.run(pip_cmd + ["coverage>=7.4", "pytest"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"仓库 {repo_path} 安装 coverage / pytest 失败: {e!r}")
        return False

    return True


def setup_all_repo_venvs() -> None:
    """为仓库目录下的所有仓库建独立虚拟环境。"""
    repo_dirs = [d for d in REPOS_DIR.iterdir() if d.is_dir()]
    for repo_dir in repo_dirs:
        print(f"准备仓库环境: {repo_dir.name}")
        setup_repo_venv(str(repo_dir))
    print("仓库环境准备完成。")
