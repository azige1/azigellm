"""全局配置：路径与环境变量。

所有可调项都从环境变量读取，默认值仅用于本地样例调试。
"""

import os
from pathlib import Path

# 作业包根目录（harness/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 数据集与产物布局
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"          # 源语言库源码（需自行放入）
CASES_DIR = DATA_DIR / "testcases"          # 黑盒判分用例（jsonl）
OUTPUT_DIR = PROJECT_ROOT / "output"        # agent 生成产物
LOG_DIR = PROJECT_ROOT / "logs"             # agent 轨迹日志

# 容器内固定路径
CONTAINER_WORKSPACE = "/workspace"
CONTAINER_DATASET_DIR = f"{CONTAINER_WORKSPACE}/dataset"
CONTAINER_OUTPUT_DIR = "/output"

# 任务容器名前缀
CONTAINER_PREFIX = "xbuild"

# LLM 接入（OpenAI 兼容协议）
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "your-model-name")

# 判分用的解释器，可被环境变量覆盖
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python3")
NODE_BIN = os.environ.get("NODE_BIN", "node")


def docker_image(task_name: str) -> str:
    """返回某任务使用的判分镜像。

    镜像不包含在本作业包内，需要自行构建或拉取，
    并用环境变量 REBUILD_DOCKER_IMAGE 指向它。
    """
    return os.environ.get(
        "REBUILD_DOCKER_IMAGE",
        f"<your-registry>/{task_name}-arena:latest",
    )
