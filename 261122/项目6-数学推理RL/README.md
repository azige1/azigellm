# 项目6：数学推理 RLVR 训练工程

## 项目简介

本项目面向中小学及竞赛级数学推理场景，基于 `verl 0.9.1` V1 同步训练架构实现两阶段（8K 扩至 16K 上下文）DAPO 强化学习训练。模型以 thinking 模式推理模型为起点，围绕判分分流与校准、entropy 带宽正则、组内通过率峰值加权以及动态采样溢出样本复用四项机制改进推理正确率与推导完整性。

**与其他项目的衔接**：本项目为独立的数学推理 RLVR 工程，与其他项目没有代码或中间产物依赖；训练直接加载本地或 HuggingFace 格式的推理模型权重（默认 `Qwen/Qwen3-8B`）与预处理后的数学题库 Parquet 文件，输出两阶段全参 RL checkpoint。

## 目录结构

```text
项目6-数学推理RL/
├── pyproject.toml                       # 包配置与依赖声明
├── README.md                            # 项目说明文档
├── configs/
│   ├── dapo_8xa800_stage1_8k.yaml       # 8×A800 阶段一（8K）训练配置
│   ├── dapo_8xa800_stage2_16k.yaml      # 8×A800 阶段二（16K）续训配置
│   └── ref_scale_14b_24xh800.yaml       # 14B + 24×H800 大规模参考配置
├── data/
│   ├── sample_math_problems.jsonl       # 单一表达式与多小问解答题样例
│   └── sample_subq_gold.jsonl           # 小问级人工金标校验样例
├── experiments/
│   └── README.md                        # 实验设计与预期结果对照文档
├── math_reasoning_rl/
│   ├── __init__.py                      # 包导出接口
│   ├── config.py                        # 6.1–6.4 模块配置数据类
│   ├── symbolic_verify.py               # 6.1 LaTeX 抽取与 SymPy 符号等价判定
│   ├── judge_client.py                  # 6.1 OpenAI 兼容逐小问 judge 客户端
│   ├── reward_scorer.py                 # 6.1 门槛过滤与规则/judge 分流判分
│   ├── false_positive_eval.py           # 6.1 小问级金标假阳性率统计脚本
│   ├── entropy_band_loss.py             # 6.2 entropy 带宽 hinge 正则与 PPO loss
│   ├── runtime_hooks.py                 # 6.2 Ray worker_process_setup_hook 补丁
│   ├── difficulty_advantage.py          # 6.3 组通过率峰值加权 advantage
│   ├── overflow_replay_buffer.py        # 6.4 溢出样本单步复用缓冲与截断 IS
│   ├── trainer.py                       # V1 PPOTrainerSync 子类整合 6.3 与 6.4
│   ├── train_entry.py                   # 训练启动入口与 MathReasoningTaskRunner
│   └── data_prep.py                     # 题型过滤、两级去重与 Parquet 转换
├── scripts/
│   ├── prepare_data.sh                  # 数据准备脚本
│   ├── start_judge_vllm.sh              # GPU 6–7 启动 Qwen3-32B judge 服务
│   ├── run_stage1_8k.sh                 # GPU 0–5 启动阶段一（8K）训练
│   └── run_stage2_16k.sh                # GPU 0–5 启动阶段二（16K）续训
└── tests/
    ├── test_reward_and_symbolic.py      # 6.1 门槛、SymPy、分流与 FPR 单元测试
    ├── test_entropy_band_loss.py        # 6.2 hinge 正则与 hook 替换单元测试
    ├── test_difficulty_advantage.py     # 6.3 uid 分组加权 advantage 单元测试
    ├── test_overflow_buffer_and_is.py   # 6.4 单步缓冲 TTL 与截断 IS 单元测试
    └── test_data_prep_and_config.py     # 数据清洗去重与 YAML 配置校验测试
```

## 环境与安装

- 硬件（主运行档位）：AutoDL `8 × A800 80G`
  - GPU 0–5：运行策略模型 `Qwen/Qwen3-8B` 全参训练与同卡 vLLM rollout（`TP=2`）。
  - GPU 6–7：运行独立 vLLM OpenAI 兼容服务部署 `Qwen/Qwen3-32B`（`TP=2`）作为逐小问 judge。
- 基础依赖：`verl==0.9.1`（`verl[vllm]` 固定 vLLM 0.24.0、transformers 5.9.0、torch 2.11）、`sympy`、`httpx`、`datasets`、`pyarrow`。

```bash
pip install -e ".[dev]"
# 完整训练环境另需确保已安装 verl==0.9.1 及其 vllm/ray/transfer_queue 依赖
```

## 运行步骤

### 1. 数据转换、去重与题型过滤

将开源题与自有小问标注 JSONL 转换为 `verl` Parquet 格式（自动删除选择、判断、证明题，并执行规范化精确去重与 n-gram Jaccard 模糊去重）：

```bash
bash scripts/prepare_data.sh
```

### 2. 启动逐小问 Judge 服务（GPU 6–7）

```bash
bash scripts/start_judge_vllm.sh
```

### 3. 阶段一训练（8K 上下文，GPU 0–5）

```bash
bash scripts/run_stage1_8k.sh
```

### 4. 阶段二续训（16K 上下文，GPU 0–5）

```bash
bash scripts/run_stage2_16k.sh
```

### 5. 小问级金标假阳性率校准（300 条金标）

```bash
python -m math_reasoning_rl.false_positive_eval \
  --gold-file data/sample_subq_gold.jsonl \
  --judge-base-url http://127.0.0.1:8001/v1 \
  --judge-model Qwen/Qwen3-32B
```

## 数据示例

取自 `data/sample_math_problems.jsonl` 的单一表达式题与多小问解答题样例：

```json
{"qid": "k12-calc-10001", "type": "计算", "source": "opensource_math", "question": "计算：\\(\\frac{3}{4} + \\frac{5}{6} - \\frac{1}{3}\\) 的值。", "subq": [{"id": 1, "answer": "\\frac{5}{4}"}]}
{"qid": "k12-alg-30417", "type": "解答", "source": "k12_self", "question": "已知二次函数 f(x) = x^2 - 4x + 1。(1) 求函数图像的对称轴方程；(2) 求 f(x) 在实数集 R 上的最小值。", "subq": [{"id": 1, "answer": "x=2"}, {"id": 2, "answer": "最小值为 -3", "key_step": "配方"}]}
```

取自 `data/sample_subq_gold.jsonl` 的小问级金标样例：

```json
{"qid": "k12-alg-30417", "subq_id": 2, "question": "已知二次函数 f(x) = x^2 - 4x + 1。(1) 求对称轴方程；(2) 求最小值。", "std_answer": "最小值为 -3", "key_step": "配方", "all_subq": [{"id": 1, "answer": "x=2"}, {"id": 2, "answer": "最小值为 -3", "key_step": "配方"}], "student_response": "(1) 对称轴 x=2；(2) 直接代入 x=1 得 1-4+1=-2，蒙一个最小值为 -3。", "gold_verdict": 0}
```

## 技术点与代码落点

| 技术点 | 机制说明 | 代码落点 |
| --- | --- | --- |
| 6.1 规则验证与逐小问 judge 分流判分 | 先检查 `<think>...</think>` 格式与 thinking/答案两段 4-gram 重复率（>0.3 直接 −1）；单一表达式走 LaTeX 抽取与 SymPy 等价判定；多小问/解答题逐小问并发调用 OpenAI 兼容 judge，先抽取原文再给 0/1，全 1 得 1 否则 −1；支持 300 条金标 FPR 统计 | `reward_scorer.py`（`check_response_gate`、`compute_score`）、`symbolic_verify.py`（`verify_rule_answer`）、`judge_client.py`（`OpenAICompatJudgeClient`、`EXTRACT_THEN_JUDGE_PROMPT`）、`false_positive_eval.py` |
| 6.2 entropy 带宽约束 hinge 正则 | 只对回答 token 平均 entropy 落在 `[0.45, 0.65]` 之外的部分施加系数 `0.005` 的 hinge 惩罚；通过 Ray `worker_process_setup_hook` 的后置导入钩子同时替换 `losses.ppo_loss` 与 `engine_workers.ppo_loss` | `entropy_band_loss.py`（`compute_entropy_band_hinge_loss`、`entropy_band_ppo_loss`）、`runtime_hooks.py`（`setup_worker_patches`）、`train_entry.py`（`build_runtime_env_with_hooks`） |
| 6.3 组通过率峰值加权 advantage | 按 `uid` 分组计算 `A_i = (R_i - R_bar) * w(p)`，不除以标准差，其中 `w(p) = 1 + 0.6 * exp(-(p - 0.25)^2 / (2 * 0.15^2))`；配置 `seq-mean-token-sum-norm` + 固定 `loss_scale_factor` 去掉长度归一化 | `difficulty_advantage.py`（`compute_pass_rate_weight`、`compute_pass_rate_weighted_advantage`）、`trainer.py`（`MathReasoningPPOTrainer._compute_advantage`） |
| 6.4 溢出样本单步复用与截断重要性比 | 在 V1 `ReplayBuffer` 中保留凑满 batch 后溢出的合格组至下一步（严格 1 步 TTL，上步复用比例 >0.5 时暂停一步）；对复用样本计算以采样时 `rollout_log_probs` 为分母、截断上限为 `2.0` 的 token 级 `rollout_is_weights`，并写入监控指标 | `overflow_replay_buffer.py`（`OverflowReuseReplayBuffer`、`compute_truncated_importance_weights`）、`trainer.py`（`prepare_step`、`_compute_advantage`） |

## 配置项

- 主运行配置：`configs/dapo_8xa800_stage1_8k.yaml`、`configs/dapo_8xa800_stage2_16k.yaml`
  - 策略与 rollout：`Qwen/Qwen3-8B`、`n_gpus_per_node: 6`、`train_batch_size: 64`、`rollout.n: 8`、阶段一 `response_length: 8192`（`loss_scale_factor: 8192`）、阶段二 `response_length: 16384`（`loss_scale_factor: 16384`）、`lr: 1e-6`、`clip_ratio_low: 0.2`、`clip_ratio_high: 0.28`、`use_kl_loss: false`。
  - 自定义参数段 `math_rl`：`reward_routing`（`repetition_threshold: 0.3`）、`entropy_band`（`low: 0.45`、`high: 0.65`、`coef: 0.005`）、`pass_rate_weight`（`center: 0.25`、`sigma: 0.15`、`amplitude: 0.6`）、`overflow_reuse`（`max_reuse_steps: 1`、`max_reuse_ratio: 0.5`、`is_ratio_clip: 2.0`）。
- 集群规模参考配置（仅配置参考）：`configs/ref_scale_14b_24xh800.yaml`
  - 14B 蒸馏推理模型全参训练占 `16 × H800`（`nnodes: 2, n_gpus_per_node: 8`，`train_batch_size: 128`，阶段一 500 步 8K + 阶段二 400 步 16K），judge 服务 `Qwen3-235B-A22B` 另占 `8 × H800`。

## 评测指标与预期结果

- 主评测指标：内部风格评测集准确率（含关键步骤判分，每题采样 4 次取均值）与公开数学评测（GAOKAO-Bench 数学）。
- 辅助监控指标：小问级 judge 假阳性率、语种混杂回答占比（连续 3 个以上非题目语言 token）、`actor/entropy_band_mean`、`adv/mid_difficulty_group_frac`、`training/overflow_buffer/oversample_factor` 与 `training/overflow_buffer/reused_ratio`。
- 预期增量趋势（详见 `experiments/README.md`）：8×A800 + Qwen3-8B 档位下，四项改进叠加预期将内部风格评测集准确率由起点 78.5% 提升至 85.1%（+6.6），GAOKAO-Bench 数学由 82.0% 提升至 87.0%（+5.0），过采样倍数由 ~4.1 降至 ~2.3。

## 测试

所有不依赖 GPU 与外网的单元测试均通过 `pytest` 覆盖：

```bash
pytest -q
```
