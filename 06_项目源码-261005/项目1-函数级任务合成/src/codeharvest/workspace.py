"""工作区路径配置：读取 settings.yml，集中管理数据目录。"""

import os
from datetime import datetime
from pathlib import Path
import yaml

current_dir = Path(__file__).parent
config_path = current_dir / "settings.yml"
with open(config_path, "r") as file:
    config: dict[str, str] = yaml.safe_load(file)  

HOME_DIR = Path(os.path.expanduser("~"))
WORKSPACE_DIR = HOME_DIR / config["workspace_dir"]
REPOS_DIR = HOME_DIR / config["repos_dir"]
CACHE_DIR = HOME_DIR / config["cache_dir"]

GRAPHS_DIR = WORKSPACE_DIR / "repo_graphs"
INTERESTING_FUNCS_DIR = WORKSPACE_DIR / "interesting_functions"
TESTGEN_DIR = WORKSPACE_DIR / "testgen"
EXECUTION_DIR = WORKSPACE_DIR / "execution"
SPECGEN_DIR = WORKSPACE_DIR / "specgen"
REPORTS_DIR = WORKSPACE_DIR / "reports"
TASKS_DIR = WORKSPACE_DIR / "tasks"
CAPTURED_CALLS_DIR = WORKSPACE_DIR / "captured_calls"
# 每个仓库的独立虚拟环境放在仓库目录之外，避免抽取与切片扫到 site-packages
VENVS_DIR = WORKSPACE_DIR / "repo_venvs"

EXTRACTED_DATA_DIR = WORKSPACE_DIR / "extracted_data"

CACHE_PATH = CACHE_DIR / "cache.json"
EXTRACTION_DIR = WORKSPACE_DIR / "extracted_data"

BASE_DOCKERFILE = current_dir / "extraction/docker/base_image.dockerfile"
PACKAGER_BIN_DIR = "/usr/local/bin:$PATH"

def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
