"""环境规格（EnvSpec）：把一条数据记录翻译成镜像构建与评测脚本。

- ``EnvSpec`` 是抽象基类，定义安装脚本、评测脚本、镜像名等公共属性；
- 三个子类分别对应三类数据：整库重建（from-scratch）、issue 修复（swe 风格）、
  单函数补全（simple）；
- 模块底部是两类 Dockerfile 模板。
"""

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, cast, Optional

from rebuild_spec.core.constants import (
    ABSOLUTE_REPO_DIR,
    RELATIVE_REPO_DIR,
    RepoRecord,
    FunctionRecord,
)


@dataclass
class EnvSpec(ABC):
    """单条记录的执行环境描述：安装脚本 + 评测脚本 + 镜像命名。"""

    absolute: bool
    repo: str
    # 容器内仓库目录
    repo_directory: str
    instance: Union[RepoRecord, FunctionRecord]

    @property
    def setup_script(self) -> str:
        self.repo_script_list = self.make_repo_script_list()
        return (
            "\n".join(["#!/bin/bash", "set -euxo pipefail"] + self.repo_script_list)
            + "\n"
        )

    @property
    def eval_script(self) -> str:
        self.eval_script_list = self.make_eval_script_list()
        return (
            "\n".join(["#!/bin/bash", "set -uxo pipefail"] + self.eval_script_list)
            + "\n"
        )
        # 注意这里不加 -e：结尾还要回收测试结果，不能提前退出

    @property
    def base_image_key(self) -> str:
        return "rebuildspec.base:latest"

    @property
    def repo_image_key(self) -> str:
        """仓库镜像名由安装脚本的哈希决定：脚本一变镜像自动重建。

        旧镜像不会自动清理，磁盘紧张时请手动 prune。
        """
        hash_object = hashlib.sha256()
        hash_object.update(str(self.setup_script).encode("utf-8"))
        hash_value = hash_object.hexdigest()
        val = hash_value[:22]  # 22 位已足够避免碰撞
        repo = self.repo.split("/")[-1].split("__")[-1].split("-")[0]
        return f"rebuildspec.repo.{repo}.{val}:v0".lower()

    @property
    def repo_image_tag(self) -> str:
        """推送到镜像仓库时使用的 tag。registry 取自环境变量
        ``REBUILD_SPEC_REGISTRY``，未设置时使用占位值，请在推送前配置。"""
        registry = os.environ.get("REBUILD_SPEC_REGISTRY", "your-registry")
        repo = self.repo.split("/")[-1]
        tag = f"{registry}/{repo}:v0".lower()
        if "__" in repo:  # swe 风格实例：仓库名里带编号，需要额外哈希区分
            repo = repo.split("__")[-1].split("-")[0]
            hash_object = hashlib.sha256()
            hash_object.update(str(self.setup_script).encode("utf-8"))
            hash_value = hash_object.hexdigest()
            val = hash_value[:22]
            tag = f"{registry}/{repo}.{val}:v0".lower()
        return tag

    def get_container_name(self, run_id: Optional[str] = None) -> str:
        repo = self.repo.split("/")[-1]
        if not run_id:
            return f"rebuildspec.eval.{repo}"
        return f"rebuildspec.eval.{repo}.{run_id}".lower()

    @property
    def base_dockerfile(self) -> str:
        return render_base_dockerfile(self.platform)

    @property
    def repo_dockerfile(self) -> str:
        return render_repo_dockerfile(self.platform)

    @property
    def platform(self) -> str:
        return "linux/x86_64"

    @abstractmethod
    def make_repo_script_list(self) -> list[str]:
        pass

    @abstractmethod
    def make_eval_script_list(self) -> list[str]:
        pass


class FromScratchSpec(EnvSpec):
    """整库重建：从空实现出发，对照契约文档补齐整个库。"""

    def make_repo_script_list(self) -> list[str]:
        """生成镜像内的环境安装命令序列。"""
        specs = self.instance["setup"]
        repo = self.instance["repo"]
        env_setup_commit = self.instance["reference_commit"]
        base_commit = self.instance["base_commit"]

        setup_commands = [
            # 浅克隆，避免在历史里翻到原始实现
            f"git clone --depth 1 -o origin https://github.com/{repo} {self.repo_directory}",
            f"chmod -R 777 {self.repo_directory}",  # 让非 root 用户也能跑测试
            f"cd {self.repo_directory}",
            # 分别取回环境安装提交与基线提交
            f"git fetch --depth 1 origin {env_setup_commit} {base_commit}",
            f"git reset --hard {env_setup_commit}",
            # 删掉远端，后续看不到新提交
            "git remote remove origin",
            f"uv venv --python {specs['python']}",
            "source .venv/bin/activate",
            "which python",
        ]

        # 系统级预装
        if "pre_install" in specs and specs["pre_install"] is not None:
            for pre_install in specs["pre_install"]:
                if "apt-get install" in pre_install and "-y" not in pre_install:
                    pre_install = pre_install.replace(
                        "apt-get install", "apt-get install -y --no-install-recommends"
                    )
                elif "apt install" in pre_install and "-y" not in pre_install:
                    pre_install = pre_install.replace(
                        "apt install", "apt install -y --no-install-recommends"
                    )
                setup_commands.append(pre_install)

        # 依赖文件
        if "packages" in specs and specs["packages"] is not None:
            for package in specs["packages"]:
                cmd = f"uv pip install -r {package}"
                setup_commands.append(cmd)

        # 额外的 pip 包
        if "pip_packages" in specs and specs["pip_packages"] is not None:
            pip_packages = [f'"{one}"' for one in specs["pip_packages"]]
            pip_packages = " ".join(pip_packages)
            cmd = f"uv pip install {pip_packages}"
            setup_commands.append(cmd)

        if "install" in specs and specs["install"] is not None:
            if specs["install"].startswith("pip"):
                install = "uv " + specs["install"]
            else:
                raise ValueError(
                    f"install command should always start with pip, but you have {specs['install']}"
                )
            setup_commands.append(install)
        setup_commands.append(
            "uv pip install -U pytest pytest-cov coverage pytest-json-report"
        )
        setup_commands.append(f"git reset --hard {base_commit}")
        return setup_commands

    def make_eval_script_list(self) -> list[str]:
        """生成评测命令序列：打补丁、跑测试、留结果。"""
        diff_path = "/patch.diff" if self.absolute else "../patch.diff"
        eval_script_list = [
            f"cd {self.repo_directory}",
            "source .venv/bin/activate",
            f"git reset --hard {self.instance['base_commit']}",
            f"git apply --allow-empty -v {diff_path}",
            "git status",
            f"{self.instance['test']['test_cmd']} --json-report --json-report-file=report.json --continue-on-collection-errors{{coverage}} {{test_ids}} > test_output.txt 2>&1",
            "echo $? > pytest_exit_code.txt",
        ]
        return eval_script_list


class FunctionTaskSpec(EnvSpec):
    """单函数补全：所有小题共用一个轻量镜像。"""

    def make_repo_script_list(self) -> list[str]:
        setup_commands = [
            f"mkdir {self.repo_directory} && cd {self.repo_directory}",
            "uv venv --python 3.12",
            "source .venv/bin/activate",
            "uv pip install -U pytest pytest-cov coverage pytest-json-report",
            "which python",
        ]
        return setup_commands

    def make_eval_script_list(self) -> list[str]:
        eval_script_list = [
            f"cd {self.repo_directory}",
            "source .venv/bin/activate",
            "cat /patch.diff > test.py",
            "pytest test.py > test_output.txt 2>&1",
            "echo $? > pytest_exit_code.txt",
        ]
        return eval_script_list


class SweTaskSpec(EnvSpec):
    """issue 修复（swe 风格）：在安装脚本之外，评测时还要先装一次项目。"""

    def make_repo_script_list(self) -> list[str]:
        specs = self.instance["setup"]
        repo = self.instance["repo"]
        version = int(str(specs["python"]).split(".")[-1])
        if version < 7:
            specs["python"] = 3.7

        base_commit = self.instance["base_commit"]
        setup_commands = [
            # 浅克隆，避免在历史里翻到修复提交
            f"git clone --depth 1 -o origin https://github.com/{repo} {self.repo_directory}",
            f"chmod -R 777 {self.repo_directory}",
            f"cd {self.repo_directory}",
            f"git fetch --depth 1 origin {base_commit}",
            "git remote remove origin",
            f"uv venv --python {specs['python']}",
            "source .venv/bin/activate",
            "which python",
        ]

        if "pre_install" in specs and specs["pre_install"] is not None:
            for pre_install in specs["pre_install"]:
                if "apt-get install" in pre_install and "-y" not in pre_install:
                    pre_install = pre_install.replace(
                        "apt-get install", "apt-get install -y --no-install-recommends"
                    )
                elif "apt install" in pre_install and "-y" not in pre_install:
                    pre_install = pre_install.replace(
                        "apt install", "apt install -y --no-install-recommends"
                    )
                setup_commands.append(pre_install)

        if "packages" in specs and specs["packages"] is not None:
            if isinstance(specs["packages"], list):
                for package in specs["packages"]:
                    if ".txt" in package:
                        cmd = f"uv pip install -r {package}"
                    else:
                        cmd = f"uv pip install {package}"
                    setup_commands.append(cmd)
            elif isinstance(specs["packages"], str):
                if ".txt" in specs["packages"]:
                    cmd = f"uv pip install -r {specs['packages']}"
                else:
                    cmd = f"uv pip install {specs['packages']}"
                setup_commands.append(cmd)
            else:
                raise TypeError(
                    f"{specs['packages']} has a type other than string and list so couldn't be parsed."
                )

        if "pip_packages" in specs and specs["pip_packages"] is not None:
            pip_packages = [one.split(";")[0].strip() for one in specs["pip_packages"]]
            pip_packages = [f'"{one}"' for one in pip_packages]
            pip_packages = " ".join(pip_packages)
            cmd = f"uv pip install {pip_packages}"
            setup_commands.append(cmd)
        setup_commands.append(
            "uv pip install pytest pytest-cov coverage pytest-json-report"
        )
        return setup_commands

    def make_eval_script_list(self) -> list[str]:
        specs = self.instance["setup"]
        results = []
        if "install" in specs and specs["install"] is not None:
            installs = specs["install"].split("; ")
            for one in installs:
                if "python -m pip install" in one:
                    install = one.replace("python -m ", "uv run python -m ")
                    install = "uv pip install pip && " + install
                else:
                    install = one
                if install.startswith("pip"):
                    install = "uv " + install
                elif install.startswith("python setup.py"):
                    install = install.replace("python ", "uv run python ")
                results.append(install)
        eval_script_list = (
            [
                f"cd {self.repo_directory}",
                "source .venv/bin/activate",
                f"git reset --hard {self.instance['base_commit']}",
                "git apply --allow-empty -v /patch.diff",
            ]
            + results
            + [
                "git status",
                f"{self.instance['test']['test_cmd']} --json-report --json-report-file=report.json --continue-on-collection-errors{{coverage}} {{test_ids}} > test_output.txt 2>&1",
                "echo $? > pytest_exit_code.txt",
            ]
        )
        return eval_script_list


def specs_from_records(
    dataset: Union[list[Union[RepoRecord, FunctionRecord]], list[EnvSpec]],
    dataset_type: str,
    absolute: bool,
) -> list[EnvSpec]:
    """把一批记录转成 EnvSpec 列表；已经是 EnvSpec 时原样返回（幂等）。"""
    if isinstance(dataset[0], EnvSpec):
        return cast(list[EnvSpec], dataset)
    return list(
        map(
            lambda instance: build_spec(instance, dataset_type, absolute),
            cast(list["RepoRecord"], dataset),
        )
    )


def build_spec(
    instance: Union[RepoRecord, FunctionRecord], dataset_type: str, absolute: bool
) -> EnvSpec:
    """按数据类型构造对应的 EnvSpec。"""
    repo_directory = ABSOLUTE_REPO_DIR if absolute else RELATIVE_REPO_DIR
    if isinstance(instance, EnvSpec):
        return instance
    if dataset_type == "scratch":
        return FromScratchSpec(
            repo=instance["instance_id"],
            repo_directory=repo_directory,
            instance=instance,
            absolute=absolute,
        )
    elif dataset_type == "swebench":
        return SweTaskSpec(
            repo=instance["instance_id"],
            repo_directory=repo_directory,
            instance=instance,
            absolute=absolute,
        )
    elif dataset_type == "simple":
        return FunctionTaskSpec(
            repo="simple",  # 所有函数补全小题共用一个镜像
            repo_directory=repo_directory,
            instance=instance,
            absolute=absolute,
        )
    else:
        raise NotImplementedError(
            f"{dataset_type} is not supported.\nWe only support scratch and swebench instances for now."
        )


# ----------------------------------------------------------------------------
# Dockerfile 模板
# ----------------------------------------------------------------------------

# 改了基础镜像后，所有仓库镜像都要重建
_DOCKERFILE_BASE = r"""
FROM --platform={platform} ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt update && apt install -y \
wget \
build-essential \
libffi-dev \
libtiff-dev \
python3 \
python3-pip \
python-is-python3 \
jq \
curl \
locales \
locales-all \
tzdata \
&& rm -rf /var/lib/apt/lists/*

# 安装最新版 Git
RUN apt-get update && apt-get install software-properties-common -y
RUN add-apt-repository ppa:git-core/ppa -y
RUN apt-get update && apt-get install git -y

# 安装 uv（安装脚本需要 curl 与证书）
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV PATH="/root/.cargo/bin/:$PATH"
"""

_DOCKERFILE_REPO = r"""FROM --platform={platform} rebuildspec.base:latest

COPY ./setup.sh /root/
RUN chmod +x /root/setup.sh
RUN /bin/bash /root/setup.sh

WORKDIR /testbed/

# 进容器即激活测试环境
RUN echo "source /testbed/.venv/bin/activate" > /root/.bashrc
"""


def render_base_dockerfile(platform: str) -> str:
    return _DOCKERFILE_BASE.format(platform=platform)


def render_repo_dockerfile(platform: str) -> str:
    return _DOCKERFILE_REPO.format(platform=platform)


__all__ = []
