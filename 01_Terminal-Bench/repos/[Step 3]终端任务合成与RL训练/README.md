# 终端任务合成与 RL 训练（作业代码包）

一个「无人工标注」的终端任务生产线：用 LLM 程序化合成 Linux 终端任务，
让 agent 在容器里解题并打分，再把可解的任务整理成强化学习训练数据，
最后用 PPO/GRPO 训练终端 agent。

## 背景

训练终端 agent 的最大瓶颈是环境数据：真实任务标注贵、覆盖窄。本项目的
思路是让 LLM 同时扮演「出题人」和「阅卷人」：

- 出题：按「领域 × 难度 × 角色」组合采样，让 LLM 写出题面（`<task>`）
  与标准答案（`<truth>`），再分别生成初始状态测试、终态测试和容器环境。
- 阅卷：终态测试就是 RL 的 reward 函数——pytest 全过记 1，否则记 0。
- 训练：只有「至少被解出过一次」的任务才进入训练集，保证奖励信号密度。

整个管线有 Docker 与 Apptainer（SIF）两套容器路线：前者用于本地 /
云主机与 Harbor 评测框架对接，后者用于无 Docker 守护进程的集群环境。

## 安装

**前置：** Python 3.12+，推荐用 [uv](https://github.com/astral-sh/uv) 管理环境。

```bash
# 安装依赖
uv sync                    # 基础依赖
uv sync --extra harbor     # Harbor 评测 + 企业网关依赖
uv sync --extra train      # ray / hydra（RL 训练）
uv sync --extra dev        # pytest 等开发依赖

# SIF 路线需要 Apptainer（Linux）
bash scripts/install_apptainer.sh
bash scripts/pull_ubuntu_sif.sh

# RL 训练需要 SkyRL（Linux + GPU）
bash scripts/install_skyrl.sh
```

运行测试：

```bash
pytest                     # 全部测试
pytest -m "not slow"       # 跳过慢测试
```

## 三段流程

### 第一段：任务合成

五阶段流水线（任务草稿 → 初始测试 → 终态测试 → 容器环境 → 落盘），
每个阶段都是一次批量 LLM 调用 + 本地校验，任一阶段失败即丢弃该条。

```bash
# Harbor/Docker 路线（企业网关 Claude 后端）
python generate_harbor_tasks.py --num-tasks 10 --out-dir harbor_tasks --model claude_opus

# Apptainer/SIF 路线（本地 vLLM 后端，先启动服务）
bash scripts/launch_vllm_server.sh 4 1
python -m task_factory.pipeline --num-tasks 100 --out-dir ./tasks --model Qwen/Qwen3-32B
```

`--skip-build` 可跳过 Docker 构建验证；`--difficulty mixed` 配合
`--difficulty-distribution easy:0.2,medium:0.5,hard:0.3` 控制难度配比。

### 第二段：解题与评测

```bash
# 用 Harbor 框架跑 agent 解题（每条任务 n 次尝试）
.venv/bin/harbor run \
  --agent-import-path task_factory.llm.bedrock_agent:BedrockTerminus2 \
  --model claude_4_5 --path harbor_tasks --n-attempts 8 \
  --jobs-dir harbor_jobs --n-concurrent 10 --job-name run1

# 回收结果：复制 trial、计算 pass@k、写 solution/solution.json
python -m harbor_bridge.collect_results --jobs-dir harbor_jobs

# Terminal-Bench 基线 / 训练后对比评测
bash scripts/eval_terminal_bench.sh --mode base --model Qwen/Qwen3-8B
bash scripts/eval_terminal_bench.sh --mode checkpoint \
    --checkpoint ./exports/ppo_single_node/global_step_100 --model Qwen/Qwen3-8B
```

### 第三段：RL 训练

```bash
# 数据准备：过滤出可解任务，产出 train/validation parquet
python -m rl.prepare_data --task-dir ./tasks --output-dir ./data --build-docker

# 预构建任务镜像（可选，训练时省时间）
python scripts/prebuild_images.py --data-dir ./data

# 启动训练
ray start --head
python -m rl.main --config-dir rl/confs --config-name base
# 或参考 scripts/train/ 下的单机 / 双节点示例脚本
```

训练配置见 `rl/confs/`：`base.yaml`（Llama-3.2-3B）、`base_qwen.yaml`
（Qwen2.5-7B）、`base_qwen3_8b.yaml`（8B）、`base_t4.yaml`（T4 显卡适配）。

## 目录结构

```
├── generate_harbor_tasks.py   # Harbor 格式任务生成入口（五阶段流水线）
├── audit_tasks.py             # 任务质量审计 CLI
├── task_factory/              # 任务合成核心包
│   ├── template_gen.py        #   阶段1：题面 + 标准答案（含出题 prompt 与多样性采样）
│   ├── initial_test_gen.py    #   阶段2：初始状态 pytest
│   ├── final_test_gen.py      #   阶段3：终态 pytest（即 verifier）
│   ├── docker_env_gen.py      #   阶段4（Docker）：Dockerfile 生成 + 构建验证
│   ├── apptainer_env_gen.py   #   阶段4（SIF）：.def 生成 + 构建验证
│   ├── pipeline.py            #   SIF 路线五阶段编排
│   ├── rollout_agent.py       #   单命令 agent 协议与批量解题 rollout
│   ├── solve_tasks.py         #   批量解题入口
│   ├── container_session.py   #   PTY + Apptainer 交互式容器会话
│   ├── llm/                   #   LLM 后端（本地 vLLM / 企业网关 Bedrock 两套）
│   ├── harbor_convert/        #   SIF 任务 → Harbor 格式转换
│   └── cap_tasks/             #   CAP（CDS）领域的模板化任务生成
├── task_audit/                # 任务质量审计（静态 / 容器 / 解题结果三组件）
├── harbor_bridge/             # Harbor 框架桥接（agent + 结果收集）
├── rl/                        # RL 训练（环境封装、数据准备、训练入口、配置）
├── viewer/                    # 任务与评测轨迹浏览器（Flask）
├── scripts/                   # 安装 / 评测 / 训练 / 补丁脚本
└── tests/                     # 单元测试（网络与容器调用均已 mock）
```

## 数据目录说明

以下目录是运行时产物，不进代码包；需要时按流程自行生成：

| 目录 | 内容 | 如何生成 |
|------|------|----------|
| `harbor_tasks/` | Harbor 格式任务（instruction.md / task.toml / environment/ / tests/） | 第一段 `generate_harbor_tasks.py` |
| `tasks/` | SIF 格式任务（task.json / container.def / 两个 pytest 文件） | 第一段 `task_factory.pipeline` |
| `harbor_jobs/` | Harbor 评测输出（每个 job 下按 trial 存轨迹与 reward） | 第二段 `harbor run` |
| `data/` | 训练用 parquet（train.parquet / validation.parquet） | 第三段 `rl.prepare_data` |
| `checkpoints/`、`exports/` | 训练检查点与导出的 HF 模型 | RL 训练自动产出 |

`solution/solution.json` 的格式：`{"task_name", "num_runs", "num_success",
"pass_at_k": {"1": ...}, "trials": [{"trial_name", "reward", "success"}]}`，
由 `harbor_bridge.collect_results` 写入，供数据过滤与质量审计使用。

## 浏览器

```bash
python -m viewer.server --port 5050   # 打开 http://127.0.0.1:5050
```

读取 `harbor_tasks/` 与 `harbor_jobs/`（可用环境变量 `VIEWER_TASKS_DIR` /
`VIEWER_RUNS_DIR` 改位置），提供仪表盘、运行列表、任务目录与逐 trial 的
完整轨迹回放。

## 作业任务

> 完成以下任务时，请先阅读 `docs/任务质量审计.md` 与 `docs/实验记录.md`。

1. **跑通静态审计**：用 `python audit_tasks.py offline --tasks-dir <你的任务目录>`
   审计一批任务，解释 `structural_score` 与 `desc_overlap` 各自的含义，
   以及为什么两者都需要（它们近似不相关）。
2. **实现质量门禁**：把 `task_audit.static_checks` 接入
   `generate_harbor_tasks.py` 的第四阶段之前——`reject` 的任务不再进入
   Dockerfile 生成（省构建开销），`flag` 的任务把 flags 写入 task.toml 的
   `[metadata].quality_flags`。提示：参考 `docs/任务质量审计.md` 第 6 节。
3. **验证阈值有效性**：用一批带 `solution.json` 的任务，按
   `structural_score` 分桶统计平均 pass@1，论证或反驳 0.2 这个 flag
   阈值（注意区分「verifier 容易被糊弄」与「任务太容易」）。
4. **新增一个断言类别**：在 `task_audit/static_checks.py` 的
   `ASSERT_CATEGORIES` 里加入对「网络请求验证」（如 `curl`、`socket`、
   `requests`）的识别，补充单元测试，并说明它属于结果性还是状态性断言。
5. **出题 prompt 改造**：阅读 `task_factory/template_gen.py` 的
   `TASK_SYSTEM_PROMPT`，针对某一具体领域（如数据库迁移）写一个专项
   出题 prompt，对比 50 条生成结果在领域分布和难度上的变化。
6. **（进阶）奖励稀疏问题**：阅读 `docs/实验记录.md` 中 3B PPO 一节的
   失败分析，提出并实现一种提高奖励密度的方法（如任务过滤、课程学习、
   或过程奖励），写清你的对照实验设计。
