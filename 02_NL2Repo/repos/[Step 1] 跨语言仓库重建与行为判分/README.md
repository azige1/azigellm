# 跨语言仓库重建与行为判分

一份「让 agent 把整个代码仓库翻译到另一门语言、再用黑盒行为测试判分」的完整作业框架。

## 背景

衡量代码模型能力的一种方式，是看它能否**从零重建一个真实仓库**：给 agent 一个
用源语言写成的库和它的测试程序，要求在不看库内部实现、不引用任何现成等价库的
前提下，把测试程序逐文件重写成目标语言，且行为与原实现完全一致。

本作业包含两类翻译任务：

| 任务标识   | 源语言 | 目标语言          | 生成物形态                |
|------------|--------|-------------------|---------------------------|
| `py2node`  | Python | Node.js ESM       | 一组 `.mjs` 模块 + 入口文件 |
| `cpp2rust` | C++    | Rust（仅 std）    | Cargo 工程 + 编译出的二进制 |

共同点：

- **黑盒**：agent 只能看测试程序的源码和参考程序的行为，不能翻库本身的实现；
- **零外部依赖**：py2node 只能用 Node.js 内置模块（禁 npm 包），cpp2rust 只能用
  标准库（禁 crates.io），框架会在 agent 的命令里做正则拦截；
- **行为判分**：不读生成代码，用同一组 CLI 参数分别跑参考实现与生成实现，
  比对 stdout（详见「判分方式」）。

## 目录结构

```
.
├── README.md
├── requirements.txt
├── run_benchmark.py        # 做题入口：批量让 agent 完成翻译
├── grade.py                # 判分入口：黑盒行为判分
├── stats_report.py         # 统计报表：micro/macro 指标 + Bootstrap 置信区间
├── harness/                # 公共核心（做题侧）
│   ├── config.py           #   路径与环境变量
│   ├── llm.py              #   LLM 客户端 + 两种 agent 对话循环
│   ├── sandbox.py          #   本地 shell / Docker 容器两种执行后端 + 命令黑名单
│   ├── batch.py            #   任务收集、单题流程、进程池调度、轨迹与用量统计
│   └── tasks/              #   任务适配层：每类任务一个 TaskSpec
│       ├── py2node.py      #     Python -> Node.js 的提示词与参考产物定位
│       └── cpp2rust.py     #     C++ -> Rust 的提示词与参考产物定位
├── grading/                # 判分侧
│   ├── compare.py          #   stdout 归一化与比对
│   ├── runners.py          #   本地 / 容器两种判分执行后端
│   └── score.py            #   逐用例计分、指标汇总、错误分类
├── docker_env/
│   ├── cpp2rust.Dockerfile #   cpp2rust 容器镜像构建文件
│   └── compile_cpp_tests.sh#   批量编译 C++ 参考二进制
├── tools/
│   └── filter_cases.py     #   按清单过滤用例 jsonl
└── data/
    ├── dataset/            # 被翻译的库源码（需自行准备，见其中 README）
    └── testcases/          # 黑盒判分用例（jsonl，请勿改动）
        ├── py2node/sample_lib_a.jsonl
        └── cpp2rust/sample_lib_a.jsonl + api_lines.jsonl
```

## 环境准备

```bash
pip install -r requirements.txt

# LLM 接入（OpenAI 兼容协议）
export LLM_BASE_URL="https://your-endpoint/v1"
export LLM_API_KEY="your-key"
export LLM_MODEL="your-model-name"

# Docker 模式需要可用的 docker；判分/做题镜像需自行构建
docker --version
```

### 关于容器镜像

镜像**不包含**在本作业包内，代码中的镜像名只是占位符
（`<your-registry>/py2node-arena:latest` 等）。请自行准备：

- `py2node`：装有 Node.js 与 python3 的 Linux 镜像即可；
- `cpp2rust`：用 `docker_env/cpp2rust.Dockerfile` 构建（基础镜像需带 g++ 与
  python3），产物镜像内应有 `rustc`。

准备好后用环境变量指向它（两个任务共用一个变量，按当前跑的任务取值）：

```bash
export REBUILD_DOCKER_IMAGE="<your-registry>/py2node-arena:latest"
```

## 如何跑（做题）

统一入口 `run_benchmark.py`，`--task` 选任务、`--mode` 选执行方式：

```bash
# Docker 模式（推荐）：每题一个断网容器（--network none），
# 源文件与参考二进制拷入容器，agent 在容器内工作，/output 拷回宿主机
python run_benchmark.py --task py2node --mode docker -k 4 -m your-model

# 本地模式：agent 直接在宿主机执行命令（函数调用协议）
python run_benchmark.py --task cpp2rust --mode local -m your-model

# 本地自修复模式（仅 py2node）：agent 先自造若干组参数，
# 每次生成后本地跑源实现 vs 生成实现做行为校验，不通过则带反馈重做，最多 N 次
python run_benchmark.py --task py2node --mode local --self-repair 3 -m your-model
```

常用参数：`-k` 并发进程数；`--dry-run` 只列题目；`--yes` 跳过提示词确认。
开跑前会打印第一题的完整提示词供人工检查。

产物与日志：

- `output/<task>/<model>/packages/<库>/<题目>_pkg/`：每题的生成物；
- `output/<task>/<model>/progress.json`：断点进度（已成功的题目自动跳过）；
- `logs/<task>/<model>.jsonl`：agent 完整对话轨迹；
- `output/<task>/<model>/token_usage.log`：token 用量汇总。

## 判分方式

黑盒行为判分，不读生成代码：

1. 从 `data/testcases/<task>/*.jsonl` 读用例，每行是一条「文件名 + CLI 参数」：
   - py2node：`{"filename": "sample_lib_a/test1.py", "a": "..."}` —— 参数以
     `--key value` 形式传递；
   - cpp2rust：`{"file_name": "sample_lib_a/tests/test1.cpp", "complexity": "0"}`
     —— 参数按位置传递。
2. 同一组参数分别跑参考实现与生成实现：
   - py2node 逐行比对 stdout（每行去空白、忽略空行后逐行相等才算过），
     另统计「API 行覆盖率」（匹配行占比）；
   - cpp2rust 整段比对 stdout（字符串相等，或同为等值浮点数）。
3. 指标：
   - **全过率（All-Pass）**：一个源文件的全部用例都通过才算该文件通过；
   - **用例通过率**：逐用例的通过比例；
   - cpp2rust 额外输出按库规模三等分的难度桶，以及逐文件的失败原因分类
     （未产出可执行文件 / 前 4 例未全过 / 运行错误 / 输出不一致）。

```bash
# 判分（容器内执行，结果带缓存，重复跑会很快）
python grade.py --task py2node --mode docker -m your-model
python grade.py --task cpp2rust --mode docker -m your-model

# 宿主机直接执行（需要本机有 python3 / node / 可执行权限）
python grade.py --task py2node --mode local -m your-model

# 统计报表（含 Bootstrap 95% 置信区间，需 numpy）
python stats_report.py --results-dir output/py2node/your-model/grade_results
```

判分结果（逐文件通过率）写在 `output/<task>/<model>/grade_results/`。

## 数据布局

- `data/dataset/<task>/<库名>/`：被翻译的库源码与参考二进制，**需自行准备**
  （本包只含占位说明，见 `data/dataset/README.md`）；
- `data/testcases/<task>/`：判分用例，**请勿修改**。本包附带一个样例库
  `sample_lib_a` 的用例用于跑通流程；`cpp2rust/api_lines.jsonl` 记录每个测试
  文件覆盖的库代码行数，用于难度分桶；
- 扩充新库时的约定：库目录名即用例里的文件名前缀；cpp2rust 参考二进制可用
  `docker_env/compile_cpp_tests.sh` 批量编译；新库的「禁止参照实现」登记在
  `harness/tasks/<task>.py` 的 `FORBIDDEN_HINTS`。

## 环境变量一览

| 变量 | 含义 | 默认 |
|------|------|------|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | OpenAI 兼容接口地址 / 密钥 / 模型 | — |
| `REBUILD_DOCKER_IMAGE` | 做题与判分共用的容器镜像 | `<your-registry>/<task>-arena:latest` 占位 |
| `PYTHON_BIN` / `NODE_BIN` | 本地判分用的解释器 | `python3` / `node` |
