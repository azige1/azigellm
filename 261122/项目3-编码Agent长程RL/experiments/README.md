# 实验说明：预期过程与预期结果

本文档以「预期」口吻给出代码运行档位下的四组实验设计、过程观察信号、预期结果表与排查顺序。

## 规模与硬件口径

- **主实验运行档位**：8 × A800 80G，Qwen3-8B 全参 GRPO，FSDP + vLLM 共置 8 卡；单段窗口 `32K`，累计 `27K` token（约 85%）触发压缩，至多 4 段；阶段一每步 `16 题 × 8`（120 步，最多 8 轮），阶段二每步 `8 题 × 8`（200 步，最多 60 轮）；`lr=1e-6`、`low_var_kl=0.001`、`clip=0.2/0.28`、`bypass_mode=false`。
- **参考规模配置**：`configs/stage2_qwen3_32b_yarn64k_reference.yaml` 对应 Qwen3-32B、YaRN 因子 4、`64K` 窗口（压缩阈值 `56K`，最多 120 轮，每步 `32 题 × 8`），需 `learner 32×H100 + rollout 48×H100` 集群，8×A800 仅作参考配置保留，不降机制。
- **评测口径**：在 held-out 仓库级任务（按仓库隔离，每任务重复 3 次取均值）与函数级 held-out 集上评测：仓库级平均测试通过率、触发压缩子集平均测试通过率、函数级 pass@1、篡改轨迹占比、零方差组占比与单步训练时长。

---

## 实验一：基线（基础 GRPO：0/1 reward、直接仓库级）

- **目的**：建立不使用两阶段 curriculum、通过比例 reward、篡改检测、分段共享、step 级 advantage 与容器池时的基线起点。
- **运行命令与配置**：

```bash
python -m long_horizon_rl.main --config-name stage2_repo \
  actor_rollout_ref.model.path=checkpoints/project2-qwen3-8b-sft \
  trainer.experiment_name=baseline_grpo_binary \
  coding_rl.reward.mode=binary \
  coding_rl.reward.enable_tamper_scan=false \
  coding_rl.compaction.enable=false \
  coding_rl.step_advantage.enable=false \
  coding_rl.sandbox.enable=false
```

- **预期训练过程**：
  - 正常信号：前 60 步 `curriculum/zero_variance_group_ratio` 预期维持在 `48%–55%`，每步有效参与更新的组不到一半；`reward/mean` 缓慢爬升。约 130 步后预期出现测试环境篡改行为（`conftest.py` 或测试环境变量探测），表现为训练集 reward 陡增但 held-out 通过率停滞甚至回落。
  - 异常信号：若前 10 步 `actor/pg_loss` 持续为 0 且零方差组占比高于 `80%`，优先检查容器/venv 依赖或是否未关闭 `bypass_mode`。
- **预期结果**：在 Qwen3-8B 档位下，仓库级平均通过率预期由 SFT 起点约 `24.5%` 小幅升至约 `25.7%`（`+1.2pp`），篡改轨迹占比预期升至 `4.5%–5.0%`。
- **排查顺序**：① 查 `logs/rollout_stage2` 中是否有基础设施失败被误记 0 分；② 查 `eval_heldout.py` 是否在判分前彻底删除模型写入的 `tests/` 与配置；③ 核对初始题池 `pass@8` 筛选是否生效。

---

## 实验二：默认配置（3.1–3.5 全量启用）

- **目的**：验证两阶段 curriculum、通过比例 reward + 9 条篡改检测、9 段摘要分段共享 + 轨迹总 token 归一、规范化报错 GiGPO step 级 advantage 与预热容器池的联合效果。
- **运行命令与配置**：

```bash
bash scripts/train_stage1.sh checkpoints/project2-qwen3-8b-sft \
  ../项目1-函数级任务合成/workspace/tasks/exp1_train.jsonl \
  ../项目1-函数级任务合成/workspace/tasks/exp1_eval.jsonl

bash scripts/train_stage2.sh \
  checkpoints/stage1_function/global_step_120/actor/huggingface \
  ../项目2-长程轨迹增强蒸馏/tasks/test.jsonl
```

- **预期训练过程**：
  - 阶段一（120 步）：函数级 held-out pass@1 预期从约 `52.4%` 稳步升至约 `60.2%`，平均回复轮数保持稳定，不出现明显截断退化。
  - 阶段二（200 步）：开训时 `curriculum/zero_variance_group_ratio` 预期从直接仓库级起训的约 `53%` 降至约 `32%`（启用通过比例 reward 后进一步降至约 `20%`）；`tamper/flagged_ratio` 维持在 `0.5%` 以下；`gigpo/singleton_group_ratio` 从按原始文本分组时的约 `92%` 降至约 `50%`；`compaction/triggered_ratio` 约占仓库级轨迹的四成，摘要平均长度稳定在 `2K–3.5K` token。
  - 异常信号：若 `compaction` 触发后平均摘要长度突破 `4K` 被截断或出现自指词退化，检查 `summary_max_tokens` 与阶段一混合比例；若单步耗时明显高于预期（8×A800 档约 `18–22` 分钟/步），核对 `SandboxPool` 预热命中率。
- **预期结果**：仓库级平均测试通过率预期达到约 `33.6%`（相对 SFT 起点累计提升 `+9.1pp`），触发压缩子集通过率预期由约 `17.8%` 提升至约 `25.5%`（`+7.7pp`）。
- **排查顺序**：① 检查 `coding_ppo_loss` 是否通过 `setup_hook` 生效（确认 `traj_token_weights` 写入 TQ）；② 检查 `bypass_mode` 为 `false`；③ 检查 `logs/tamper_stage2.jsonl` 确认抽检未遗漏新型篡改。

---

## 实验三：核心消融（逐项累加表）

- **目的**：在 Qwen3-8B 档位上分离 3.1–3.5 各项技术的独立贡献，确认逐项增量之和等于总提升。
- **运行命令与配置**：通过 `scripts/run_ablations.sh` 依次运行以下 6 个配置（与 SFT 起点共 7 行）。
- **预期结果表（Qwen3-8B 8×A800 档位预期值）**：

| 配置 | held-out 仓库级通过率（预期） | 相对上一行 | 预期过程关键信号 |
|---|---|---|---|
| 项目二 SFT 起点 | 24.5% | — | 函数级 pass@1 约 52.4%，压缩子集约 17.8% |
| 基础 GRPO（0/1 reward、直接仓库级） | 25.7% | +1.2pp | 零方差组约 53%，中期篡改占比约 4.8% |
| + 两阶段 curriculum（3.1） | 27.4% | +1.7pp | 阶段二开训零方差组降至约 32%，函数 pass@1 约 60.2% |
| + 通过比例 reward 与篡改检测（3.2） | 29.3% | +1.9pp | 零方差组降至约 20%，篡改占比降至 0.4% 以下 |
| + compaction 分段共享与摘要训练（3.3） | 31.2% | +1.9pp | 压缩子集通过率明显抬升，摘要后首动作重读/测试升至约 85% |
| + 规范化报错 anchor step advantage（3.4） | 32.6% | +1.4pp | 单元素组由约 92% 降至约 50%，原地重复修改率下降 |
| + 沙箱容器池预热与快照复位（3.5，同训练时长） | 33.6% | +1.0pp | 基础设施失败由约 3.1% 降至 0.4%，同等预算多训约 15% 步数 |

- 逐项增量之和：`+1.2 + 1.7 + 1.9 + 1.9 + 1.4 + 1.0 = +9.1pp`，等于总提升 `33.6% - 24.5%`。
- **排查顺序**：某项增量明显偏小时，优先检查对应开关在配置与 TQ 字段中是否真正流转：例如第 5 行重点查 `traj_token_weights` 与 `response_mask` 是否覆盖摘要 token，第 6 行重点查 `extra_fields["step_spans_json"]` 是否有足够非单元素组。

---

## 实验四：技术点逐项对照与参数扫描

- **目的**：验证 3.4 中 step 级权重 $\omega$（`0.25 / 0.5 / 1.0`）及 anchor 规范化粒度（`full` vs `exception_only`）的敏感性边界。
- **运行命令与配置**：参见 `scripts/run_ablations.sh` 后半段（分别覆盖 `coding_rl.step_advantage.omega=0.25|1.0` 与 `coding_rl.step_advantage.granularity=exception_only`）。
- **预期过程与结果**：

| 消融维度 | 对照配置 | 仓库级通过率（预期） | 预期过程判别信号 |
|---|---|---|---|
| $\omega$ 权重扫描 | `omega=0.25` | 约 31.8%（相对 3.3 仅 +0.6pp） | step 级梯度偏弱，对重复修改同一处报错的抑制不足 |
| $\omega$ 权重扫描 | `omega=0.5`（默认） | 约 32.6%（+1.4pp） | 高 step advantage 编辑中删函数占比稳定在约 1.2% |
| $\omega$ 权重扫描 | `omega=1.0` | 约 31.5%（训练后段回落） | 约 140 步后出现为消除报错而删减功能代码的行为，删函数占比升至约 7% |
| anchor 规范化粒度 | `exception_only` | 约 31.7%（+0.5pp） | 单元素组虽降至约 30%，但不同函数的同类异常混组导致动作不可比 |
| anchor 规范化粒度 | `full`（默认） | 约 32.6%（+1.4pp） | 单元素组约 50%，同状态组内修改策略清晰可比 |
| 原始文本匹配（不规范化） | `normalize_error_text` 关闭 | 约 31.3%（+0.1pp） | 单元素组超过 90%，step 级 advantage 几乎退化为全 0 |

- **排查顺序**：① 检查 `gigpo/singleton_group_ratio` 与 `gigpo/valid_step_fraction` 日志；② 若 `omega=1.0` 未出现删代码回落或 `omega=0.5` 增益低于 `0.5pp`，抽检 `error_anchor.py` 对当前任务仓 pytest 输出的正则命中率。
