#!/bin/bash
# 安装 SkyRL 训练栈：克隆 SkyRL、建独立 venv、打兼容补丁、装依赖。
# 必须在仓库根目录下运行（脚本会自动切到根目录）。
set -e

cd "$(dirname "$0")/.."

if [ ! -d "SkyRL" ]; then
  git clone https://github.com/novasky-ai/SkyRL.git
fi

python3.13 -m venv /tmp/sky
source /tmp/sky/bin/activate

# 定位 CUDA_HOME：先查标准路径，再从 PATH 里的 nvcc 反推
if [ -d "/usr/local/cuda" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
  export CUDA_HOME=/usr/local/cuda
elif [ -f "/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc" ]; then
  export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
else
  NVCC_PATH=$(which nvcc 2>/dev/null || true)
  if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))
  else
    echo "ERROR: Could not find nvcc. Set CUDA_HOME manually and re-run." >&2
    exit 1
  fi
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "Using CUDA_HOME=$CUDA_HOME"

pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install packaging wheel setuptools_scm "setuptools<75"

# 应用 SkyRL 兼容补丁（幂等，可重复执行）
python3.13 scripts/apply_skyrl_patches.py

PIP_NO_BUILD_ISOLATION=1 pip install -e "SkyRL[fsdp]"
pip install "ray[default]==2.51.1"
pip install harbor --upgrade  # 安装最新 harbor（含 mini-swe-agent 支持）
pip install -e .

# 把 nvcc 软链进 venv 的 nvidia 包目录，供 flashinfer JIT 使用
VENV_NVCC_DIR="/tmp/sky/lib64/python3.13/site-packages/nvidia/cu13/bin"
if [ ! -f "$VENV_NVCC_DIR/nvcc" ]; then
  mkdir -p "$VENV_NVCC_DIR"
  ln -sf "$CUDA_HOME/bin/nvcc" "$VENV_NVCC_DIR/nvcc"
  echo "Symlinked nvcc into venv at $VENV_NVCC_DIR/nvcc"
fi

# 在 CUDA_HOME 下补 lib64 -> lib 软链，让链接器找到 libcudart
if [ ! -e "$CUDA_HOME/lib64" ]; then
  ln -sf "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
  echo "Symlinked $CUDA_HOME/lib64 -> lib"
fi

# 修复 prometheus_fastapi_instrumentator 的路由 bug
PROM_FILE="/tmp/sky/lib64/python3.13/site-packages/prometheus_fastapi_instrumentator/routing.py"
if [ -f "$PROM_FILE" ]; then
  sed -i 's/route_name = route.path/route_name = getattr(route, "path", None)/' "$PROM_FILE"
fi
