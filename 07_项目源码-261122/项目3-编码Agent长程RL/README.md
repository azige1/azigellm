# 项目3：编码 Agent 长程 RL（长程上下文压缩与 step 级信用分配）

本项目基于 `verl==0.9.1`（V1 trainer）实现编码 Agent 的长程在线强化学习，串联项目一的函数级任务合成与项目二的仓库级整库重建任务及 SFT 模型，在 8×A800 80G 集群上对 Qwen3-8B 做两阶段全参 GRPO 训练，产出 HF 格式的仓库级 RL checkpoint。

## 目录结构

```text
long_horizon_rl/
  __init__.py        包入口
  config.py          3.1–3.5 扩展配置数据类
  curriculum.py      两阶段课程学习、pass@8 筛选与阶段二题池清理
  data_prep.py       将项目一/二任务转成 verl V1 parquet 数据集
  tamper_scan.py     9 条 AST + 文本静态篡改扫描规则与日志记录
  reward_eval.py     判分前测试产物清理、通过比例 reward 与本地/容器判分
  error_anchor.py    pytest/Python 报错规范化与 GiGPO anchor 提取
  step_advantage.py  GiGPO step 级折扣回报分组标准化与多段轨迹 token 权重归一
  losses.py          支持分段轨迹总 token 归一的自定义 ppo_loss
  sandbox_pool.py    基于 Docker SDK 的预热容器池、快照启动与官方镜像判分容器
  agent_loop.py      支持 9 段自摘要压缩的多轮 AgentLoop 与 TQ Worker/Manager
  trainer.py         继承 PPOTrainerSync 覆写 _compute_advantage 与预热/抽检钩子
  setup_hook.py      Ray worker 启动钩子，替换 verl ppo_loss
  main.py            Hydra 训练入口（关闭 bypass_mode 并启动 CodingTrainingRunner）
configs/
  agent_loop.yaml                         AgentLoop 注册配置
  stage1_function.yaml                    阶段一（函数级任务，8×A800，Qwen3-8B）
  stage2_repo.yaml                        阶段二（仓库级任务，8×A800，Qwen3-8B）
  stage2_qwen3_32b_yarn64k_reference.yaml Qwen3-32B YaRN 64K 参考配置（32+48×H100）
scripts/
  filter_pass_at_8.py   开训前 vLLM pass@8 难度筛选与阶段二题池清理
  prepare_data.py       生成阶段一/阶段二 verl parquet 文件
  audit_high_reward.py  每 30 步高分 top-50 轨迹静态篡改抽检
  eval_heldout.py       held-out 仓库级/函数级评测脚本
  train_stage1.sh       阶段一训练脚本
  train_stage2.sh       阶段二训练脚本
  run_ablations.sh      累加消融与参数扫描脚本
data/
  sample_function_tasks.jsonl  阶段一函数级任务样例
  sample_repo_tasks.jsonl      阶段二仓库级任务样例
tests/                         无 GPU/网络依赖的 pytest 单元测试
experiments/README.md          预期实验设计、过程观察信号与预期结果
```

## 环境与安装

- 运行档位：AutoDL 8 × A800 80G，Python >= 3.11，`verl==0.9.1`（FSDP 与 vLLM 0.24.0 共置 8 卡），Docker 引擎。
- 环境复用项目一（`codeharvest`）与项目二（`agent_rollout`、`rebuild_spec`、`task_builder`、`trajectory_tools`）：

```bash
pip install -e ../项目1-函数级任务合成 -e ../项目2-长程轨迹增强蒸馏
pip install -e .
```

### 上下游项目衔接

| 方向 | 衔接对象 | 路径与格式 |
|---|---|---|
| 输入（阶段一任务） | 项目一 `codeharvest` 导出的函数级任务包 | JSONL（含 `task_id`、`signature`、`spec`、`source_ref`、`test_source`、`split`） |
| 输入（阶段二任务） | 项目二 `task_builder` 构建的仓库级任务库 | `tasks/test.jsonl`（含 `instance_id`、`base_commit`、`setup`、`test`、`local_path`、`spec_path`、`hidden_tests_dir`） |
| 起点模型 | 项目二蒸馏 SFT 产出的 Qwen3-8B checkpoint | HF 目录，默认 `checkpoints/project2-qwen3-8b-sft` |
| 下游输出 | 本项目阶段二 RL 产出的 HF 格式 checkpoint | `checkpoints/stage2_repo/global_step_200/actor/huggingface`（由项目四直接作为仓库级教师读取） |

## 运行步骤

```bash
# 1. 阶段一：pass@8 筛选（1/8 至 6/8）+ parquet 打包 + 120 步函数级 RL
bash scripts/train_stage1.sh \
  checkpoints/project2-qwen3-8b-sft \
  ../项目1-函数级任务合成/workspace/tasks/exp1_train.jsonl \
  ../项目1-函数级任务合成/workspace/tasks/exp1_eval.jsonl

# 2. 阶段二：用阶段一末 checkpoint 重测 pass@8 + 200 步仓库级 RL
bash scripts/train_stage2.sh \
  checkpoints/stage1_function/global_step_120/actor/huggingface \
  ../项目2-长程轨迹增强蒸馏/tasks/test.jsonl

# 3. 每 30 步抽检 top-50 高分轨迹中的篡改模式
python scripts/audit_high_reward.py --rollout-dir logs/rollout_stage2 --out logs/audit_tamper.jsonl

# 4. Held-out 评测（3 次重复均值）
python scripts/eval_heldout.py \
  --repo-tasks ../项目2-长程轨迹增强蒸馏/tasks/test.jsonl \
  --function-tasks ../项目1-函数级任务合成/workspace/tasks/exp1_eval.jsonl \
  --model checkpoints/stage2_repo/global_step_200/actor/huggingface \
  --repeats 3
```

## 数据示例

阶段一函数级任务样例（见 `data/sample_function_tasks.jsonl`）：

```json
{"task_id": "tinydb-split_path-1", "signature": "def split_path(path: str) -> list[str]:", "spec": "Split a dotted field path into non-empty segments. Raise ValueError if path is empty or contains empty segments.", "source_ref": {"repo_id": "msiemens___tinydb", "file_path": "tinydb/utils.py", "commit": "v4.8.0"}, "coverage": 0.91, "verdict": "passed", "solve_rate": 0.5, "split": "train"}
```

阶段二仓库级任务样例（见 `data/sample_repo_tasks.jsonl`）：

```json
{"instance_id": "tinydb", "repo": "msiemens/tinydb", "base_commit": "0000000000000000000000000000000000000001", "setup": {"python": "3.11", "pip_packages": ["pyyaml"]}, "test": {"test_cmd": "python -m pytest tests", "test_dir": "tests"}, "src_dir": "tinydb", "local_path": "tasks/tinydb/repo", "spec_path": "tasks/tinydb/spec.md", "hidden_tests_dir": "tasks/tinydb/hidden_tests", "split": "train"}
```

## 技术点与代码落点

1. **3.1 函数级到仓库级的两阶段 curriculum**：`long_horizon_rl/curriculum.py`（`filter_tasks_by_pass_at_8`、`prune_fully_solved_tasks`）、`long_horizon_rl/data_prep.py`、`scripts/filter_pass_at_8.py`、`configs/stage1_function.yaml` 与 `configs/stage2_repo.yaml`。
2. **3.2 隐藏测试通过比例 reward 与篡改检测**：`long_horizon_rl/reward_eval.py`（`scrub_model_test_artifacts` 清理测试目录与配置、`evaluate_function_task`、`evaluate_repo_workspace`）、`long_horizon_rl/tamper_scan.py`（9 条 AST + 文本规则扫描与 `append_tamper_log`）、`scripts/audit_high_reward.py`。
3. **3.3 context compaction 分段轨迹 advantage 共享与摘要 token 训练**：`long_horizon_rl/agent_loop.py`（`CodingCompactionAgentLoop`、触发自摘要、复用 `trajectory_tools.prompts.nine_section_summary.nine_section_prompt` 与 `recent_turns_start` 组装新段并输出 `list[AgentLoopOutput]`）、`long_horizon_rl/losses.py`（`coding_ppo_loss` 按轨迹总 token 归一）、`long_horizon_rl/main.py`（强制关闭 `algorithm.rollout_correction.bypass_mode`）。
4. **3.4 规范化报错 anchor 的 GiGPO step 级分组**：`long_horizon_rl/error_anchor.py`（`extract_error_anchor`、`normalize_error_text`）、`long_horizon_rl/step_advantage.py`（`compute_discounted_step_returns`、`compute_gigpo_step_advantages`、`build_combined_token_advantages`）、`long_horizon_rl/trainer.py`（`CodingPPOTrainer._compute_advantage` 写入 TQ 张量字段）。
5. **3.5 沙箱容器池的预热与快照复位**：`long_horizon_rl/sandbox_pool.py`（`SandboxPool` 预热 10 容器/任务、快照启动、丢弃可写层复位、`create_scoring_container` 基于官方镜像新建判分容器）、`long_horizon_rl/trainer.py`（`CodingPPOTrainer.prepare_step` 异步预热）。

## 配置项

| 配置键 | 8×A800 默认值 | 32B 参考值 | 说明 |
|---|---|---|---|
| `actor_rollout_ref.rollout.max_model_len` | `32768` | `65536` | 单段窗口（32B 配 YaRN 因子 4） |
| `coding_rl.compaction.compact_threshold_tokens` | `27000` | `56000` | 触发策略自摘要的累计 token 阈值（约 85%） |
| `coding_rl.compaction.max_segments` / `keep_recent_turns` | `4` / `4` | `4` / `4` | 最大分段数与新段保留最近原文轮数 |
| `data.train_batch_size` × `rollout.n` | 阶段一 `16×8` / 阶段二 `8×8` | `32×8` | 每步采样题数与组大小 |
| `coding_rl.curriculum.stage1_max_turns` / `stage2_max_turns` | `8` / `60` | `8` / `120` | 两阶段最大工具交互轮数 |
| `coding_rl.sandbox.tool_timeout_s` / `pool_size_per_task` | `300` / `10` | `300` / `10` | 单次工具调用超时与每任务预热容器数 |
| `actor_rollout_ref.actor.optim.lr` / `kl_loss_coef` | `1e-6` / `0.001` | `1e-6` / `0.001` | 恒定学习率与 `low_var_kl` 系数 |
| `actor_rollout_ref.actor.clip_ratio_low` / `clip_ratio_high` | `0.2` / `0.28` | `0.2` / `0.28` | 非对称 PPO clip 上下界 |
| `coding_rl.step_advantage.gamma` / `omega` | `0.95` / `0.5` | `0.95` / `0.5` | GiGPO 折扣因子与 step advantage 权重 |

## 评测指标与预期结果

- 在 8×A800、Qwen3-8B 档位下，预期 held-out 仓库级任务平均测试通过率由 SFT 起点的约 `24.5%` 提升至约 `33.6%`（累计 `+9.1pp`），触发压缩子集由约 `17.8%` 提升至约 `25.5%`，函数级 held-out pass@1 由约 `52.4%` 提升至约 `60.2%`，篡改轨迹占比降至 `0.5%` 以下。完整预期曲线与消融明细见 [experiments/README.md](experiments/README.md)。

## 测试

```bash
pytest tests/
```

单元测试覆盖 9 条篡改扫描规则、判分前目录与配置清理、报错 anchor 规范化、GiGPO step 级与多段总 token 归一计算、pass@8 筛选与配置校验、以及基于 mock Docker 的容器池生命周期，无需 GPU 与外网。
