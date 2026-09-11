"""全局常量与数据记录的结构定义。

这里集中存放：
- 数据集记录（整库重建 / 单函数补全）的 pydantic 模型；
- 日志目录、评测后端、测试状态等常量；
- 内置仓库分组（lite / all / 单库）。
"""

from enum import Enum
from pathlib import Path
from typing import Dict, ItemsView, KeysView

from pydantic import BaseModel


class RepoRecord(BaseModel):
    """一条“整库重建”记录：包含仓库地址、基线提交与测试配置。"""

    instance_id: str
    repo: str
    base_commit: str
    reference_commit: str
    setup: dict
    test: Dict[str, str]
    src_dir: str

    def __getitem__(self, item: str):
        return getattr(self, item)

    def keys(self) -> KeysView[str]:
        """以字典键视图的形式暴露模型字段名，方便按 dict 方式遍历。"""
        return self.__annotations__.keys()


class FunctionRecord(BaseModel):
    """一条“单函数补全”记录：题面、参考解与测试代码。"""

    instance_id: str
    prompt: str
    canonical_solution: str
    test: str

    def __getitem__(self, item: str):
        return getattr(self, item)

    def keys(self) -> KeysView[str]:
        """以字典键视图的形式暴露模型字段名。"""
        return self.__annotations__.keys()


class CopyFiles(BaseModel):
    """需要拷入执行环境的文件集合（评测脚本与补丁）。"""

    eval_script: Dict[str, Path]
    patch: Dict[str, Path]

    def __getitem__(self, item: str):
        return getattr(self, item)

    def items(self) -> ItemsView[str, object]:
        """把模型字段摊平成键值对，供上层逐个拷贝。"""
        return self.dict().items()


# 克隆仓库后检出的工作分支名
BASE_BRANCH = "rebuild"

# 日志目录
BASE_IMAGE_BUILD_DIR = Path("logs/build_images/base")
REPO_IMAGE_BUILD_DIR = Path("logs/build_images/repo")
RUN_PYTEST_LOG_DIR = Path("logs/pytest")

# 测试迁移类型
FAIL_TO_PASS = "FAIL_TO_PASS"
FAIL_TO_FAIL = "FAIL_TO_FAIL"
PASS_TO_PASS = "PASS_TO_PASS"
PASS_TO_FAIL = "PASS_TO_FAIL"

# 支持的评测后端
EVAL_BACKENDS = ["local", "modal", "e2b"]
# 有 root 权限的后端（local / modal）使用容器内绝对路径
ABSOLUTE_REPO_DIR = "/testbed"
# 无 root 权限的后端（e2b）使用家目录下的相对路径
RELATIVE_REPO_DIR = "testbed"

# 小规模分组：适合课堂作业快速跑通全流程
REPO_GROUP_LITE = [
    "tinydb",
    "simpy",
    "deprecated",
    "wcwidth",
    "voluptuous",
    "cachetools",
    "imapclient",
    "marshmallow",
    "jinja",
    "cookiecutter",
    "portalocker",
    "parsel",
    "pyjwt",
    "chardet",
    "babel",
    "minitorch",
]

# 完整分组
REPO_GROUP_ALL = [
    "statsmodels",
    "python-progressbar",
    "xarray",
    "imbalanced-learn",
    "web3.py",
    "scrapy",
    "seaborn",
    "pypdf",
    "pexpect",
    "pytest",
    "pylint",
    "joblib",
    "dulwich",
    "virtualenv",
    "minitorch",
    "networkx",
    "requests",
    "sphinx",
    "jedi",
    "moviepy",
    "loguru",
    "paramiko",
    "geopandas",
    "bitstring",
    "fastapi",
    "chardet",
    "tornado",
    "python-prompt-toolkit",
    "attrs",
    "PyBoy",
    "pydantic",
    "filesystem_spec",
    "tlslite-ng",
    "graphene",
    "mimesis",
    "babel",
    "dnspython",
    "portalocker",
    "cookiecutter",
    "pyjwt",
    "python-rsa",
    "more-itertools",
    "simpy",
    "click",
    "fabric",
    "jinja",
    "flask",
    "sqlparse",
    "marshmallow",
    "imapclient",
    "tinydb",
    "cachetools",
    "voluptuous",
    "parsel",
    "wcwidth",
    "deprecated",
]

# 每个仓库同时作为单库分组
SPLIT: Dict[str, list] = {
    "all": REPO_GROUP_ALL,
    "lite": REPO_GROUP_LITE,
    **{name: [name] for name in REPO_GROUP_ALL},
}

# 供命令行帮助信息使用的全部仓库名
SPLIT_ALL = REPO_GROUP_ALL


class ResolvedStatus(Enum):
    """一次重建任务的完成度。"""

    NO = "RESOLVED_NO"
    PARTIAL = "RESOLVED_PARTIAL"
    FULL = "RESOLVED_FULL"


class TestStatus(Enum):
    """单条测试用例的运行结果。"""

    FAILED = "FAILED"
    PASSED = "PASSED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    XFAIL = "XFAIL"


# 安装与测试阶段的日志标记
INSTALL_FAIL = ">>>>> Init Failed"
INSTALL_PASS = ">>>>> Init Succeeded"
INSTALL_TIMEOUT = ">>>>> Init Timed Out"
RESET_FAILED = ">>>>> Reset Failed"
TESTS_ERROR = ">>>>> Tests Errored"
TESTS_FAILED = ">>>>> Some Tests Failed"
TESTS_PASSED = ">>>>> All Tests Passed"
TESTS_TIMEOUT = ">>>>> Tests Timed Out"


# 统计补丁时不视为代码的文件后缀
NON_TEST_EXTS = [
    ".json",
    ".png",
    "csv",
    ".txt",
    ".md",
    ".jpg",
    ".jpeg",
    ".pkl",
    ".yml",
    ".yaml",
    ".toml",
]
