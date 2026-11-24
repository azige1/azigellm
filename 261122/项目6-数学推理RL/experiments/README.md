# 实验设计与预期结果（8×A800 / Qwen3-8B 运行档位）

本实验文档以 `8 × A800 80G`（GPU 0–5 跑 `Qwen/Qwen3-8B` 全参训练与 vLLM rollout，GPU 6–7 跑 `Qwen/Qwen3-32B` TP=2 judge，每步 `64 × 8 = 512` 条样本，阶段一 8K + 阶段二 16K）为基准运行档位，给出四组对照实验的预期训练过程、预期定量结果及异常处理排查顺序。

---

## 实验一：Baseline（蒸馏起点与纯规则 DAPO）

- **实验目的**：建立未引入四项专项改进前的基线表现，包括蒸馏起点的零样本评测表现与仅使用最终答案规则判分的标准 DAPO 训练效果，并暴露长 CoT 阶段语种混杂、步骤凑对、高过采样耗时等问题。
- **运行命令与配置**：
  - 蒸馏起点直接做验证评测；
  - 纯规则 DAPO Baseline 使用阶段一、二配置并将 `math_rl` 四项开关均设为关闭（`entropy_band.enabled=false`、`pass_rate_weight.enabled=false`、`overflow_reuse.enabled=false`、`reward.custom_reward_function` 仅走规则提取）。
  ```bash
  python -m math_reasoning_rl.train_entry \
    --config configs/dapo_8xa800_stage1_8k.yaml \
    math_rl.entropy_band.enabled=false \
    math_rl.pass_rate_weight.enabled=false \
    math_rl.overflow_reuse.enabled=false
  ```
- **预期训练过程**：
  - `actor/entropy`：预期在前 180 步由初始约 `0.52` 持续抬升至 `0.75–0.80` 区间，伴随中英混杂回答占比升至 `6%–7%`。
  - `training/filter_groups/evicted_samples`：随着训练推进，全对组迅速增多，预期有效组率降至 `24%` 左右，单步过采样倍数升至 `4.0×` 以上，生成耗时占比超过 `60%`。
  - 异常判别信号：若无重复率门槛控制，长回答中会出现局部 4-gram 循环退化但最终数字恰好猜对的情况，导致后续 entropy 骤降、回答长度失控。

---

## 实验二：默认全量配置（两阶段 8K → 16K + 6.1–6.4 全启用）

- **实验目的**：验证分流判分、entropy 带宽 hinge 正则、组通过率峰值加权与溢出样本单步复用协同工作时的整体收益与稳定性。
- **运行命令与配置**：
  ```bash
  # 1. GPU 6-7 启动 judge 服务
  bash scripts/start_judge_vllm.sh
  # 2. 阶段一 8K 训练（500 步）
  bash scripts/run_stage1_8k.sh
  # 3. 阶段二 16K 续训（400 步）
  bash scripts/run_stage2_16k.sh
  ```
- **预期训练过程**：
  - **阶段一（8K）**：`actor/entropy_band_mean` 预期在前 50 步内稳定在 `[0.48, 0.63]` 区间；当偶然越过 `0.65` 上界时，`actor/entropy_band_hinge_loss` 短暂出现正值并在 15–30 步内将 entropy 拉回区间，语种混杂占比预期降至 `1%` 以内。
  - **阶段二（16K）**：上下文放宽且 `loss_scale_factor` 同步从 `8192` 调整为 `16384` 后，梯度范数 `actor/grad_norm` 预期保持平稳（无量级跳变）；`actor/entropy_band_in_range` 预期长期维持在 `0.9` 以上，hinge 惩罚极少触发。
  - **采样与缓冲**：`training/overflow_buffer/oversample_factor` 预期由基线的 `~4.1` 降至 `~2.3`，`training/overflow_buffer/reused_ratio` 预期稳定在 `0.25–0.45`，偶发超过 `0.5` 时 `reuse_paused=1.0` 自动触发并回落；`reused_is_ratio_mean` 预期在 `0.95–1.10` 之间，`reused_is_clipped_frac` 低于 `5%`。
  - 正常判别信号：内部风格评测集准确率平滑上升；`adv/mid_difficulty_group_frac`（通过率 `1/8–3/8`）对应的有效梯度份额显著高于等权基线。
  - 异常判别信号：若 `reused_is_clipped_frac` 持续超过 `20%` 或 `actor/grad_norm` 出现尖峰，说明策略单步变化过大或学习率偏高。

---

## 实验三：核心维度消融

- **实验目的**：检验三项关键超参（entropy 带宽区间、通过率峰值中心与幅度、溢出样本复用步数与截断策略）对最终准确率、语种稳定性及采样效率的影响。
- **运行命令与配置**：基于 `configs/dapo_8xa800_stage1_8k.yaml` 与 `stage2_16k.yaml` 仅替换对应消融字段。

| 消融维度 | 对比配置 | 预期内部风格评测集变化 | 预期过程指标与现象 |
| --- | --- | --- | --- |
| **entropy 带宽区间** | `[0.50, 0.60]`（过窄） | 相对默认 `[0.45, 0.65]` 低 `~0.8` | entropy 在上下界附近高频振荡，hinge 频繁触发干预策略梯度 |
| **entropy 带宽区间** | `[0.45, 0.65]`（默认） | 基准最优 | 第一阶段 entropy 稳定在 `0.48–0.63`，语种混杂降至 `~0.9%` |
| **entropy 带宽区间** | `[0.35, 0.75]`（过宽） | 相对默认低 `~0.6` | 语种混杂占比仍有 `~3.2%`，惩罚约束不足 |
| **通过率峰值位置与幅度** | `center=0.50, amplitude=0.6` | 相对默认 `center=0.25` 低 `~1.0` | 高权过度偏向中等偏易题，难得做对题的信号被稀释 |
| **通过率峰值位置与幅度** | `center=0.25, amplitude=0.3` | 相对默认 `amplitude=0.6` 低 `~1.1` | 中等偏难组（`p∈[1/8, 3/8]`）加权偏弱 |
| **通过率峰值位置与幅度** | `center=0.25, amplitude=1.0` | 相对默认持平或略低 `~0.3` | 梯度范数波动加大，entropy 下降偏快 |
| **溢出样本复用步数** | `max_reuse_steps=3` | 相对默认 `1` 步低 `~0.6` | 三步前旧策略分布偏移明显，IS 截断饱和比例升高、entropy 提前坍缩 |
| **溢出样本复用截断** | `is_ratio_clip=1.2` | 相对默认 `2.0` 低 `~0.4` | 复用样本有效权重被压低，收益缩水 |
| **小问级 judge 校准** | 整题一次判定 vs 逐小问直接判 vs 先抽取再比对 | 假阳性率预期 `~6.3%` → `~3.0%` → `~1.7%` | 整题一次判定易漏判中间小问；先抽取再比对可显著压低假阳性 |

---

## 实验四：技术点逐项对照（累加消融表）

- **实验目的**：量化 6.1–6.4 每一项改进对内部风格评测集准确率与公开数学评测（GAOKAO-Bench 数学）的边际贡献（在 `8 × A800`、`Qwen3-8B` 运行档位下）。
- **运行命令与配置**：以蒸馏起点为基准，依次叠加 6.1 分流判分、6.2 entropy 带宽约束、6.3 通过率峰值加权与去长度归一化、6.4 溢出样本单步复用。

### 预期结果汇总表（Qwen3-8B / 8×A800 档位）

| 配置阶段 | 内部风格评测集准确率（预期） | 相对上一行增量 | GAOKAO-Bench 数学（预期） | 相对上一行增量 | 关键过程指标预期变化 |
| --- | --- | --- | --- | --- | --- |
| 蒸馏起点（Qwen3-8B thinking） | 78.5% | — | 82.0% | — | 零样本起点，多小问末尾遗漏率高 |
| + DAPO 与 6.1 分流判分 | 80.3% | +1.8 | 83.4% | +1.4 | 小问 judge 假阳性率由 `~6.3%` 降至 `~1.7%`；循环退化样本直接被门槛拦截 |
| + 6.2 entropy 带宽约束 | 81.7% | +1.4 | 84.5% | +1.1 | 语种混杂占比由 `~6.8%` 降至 `~0.9%`，entropy 稳定于 `[0.48, 0.63]` |
| + 6.3 通过率峰值加权 | 83.8% | +2.1 | 86.0% | +1.5 | 单项最大增益；中等难度组梯度份额由 `~38%` 升至 `~47%` |
| + 6.4 溢出样本单步复用 | 85.1% | +1.3 | 87.0% | +1.0 | 过采样倍数由 `~4.1` 降至 `~2.3`，生成阶段耗时减少约 `35%` |
| **总提升** | **78.5% → 85.1%** | **+6.6** | **82.0% → 87.0%** | **+5.0** | 逐项增量之和严格等于总提升 |

---

## 偏离预期时的排查顺序

若实际训练或评测曲线偏离上述预期，建议按以下顺序逐层排查：

1. **检查判分门槛与 judge 连通性（6.1）**：核对 `reward/gate_passed` 比例与 `rep_4gram_think` / `rep_4gram_ans` 分布，确认 `<think>...</think>` 模板未被 tokenizer 错误截断；检查 GPU 6–7 上的 judge 服务响应延迟与 `judge_extractions` 输出格式是否稳定。
2. **核查 worker setup hook 与 entropy 计算（6.2）**：在 actor log 中确认 `actor/entropy_band_mean` 与 `actor/entropy_band_hinge_loss` 均有数值输出且 `actor.calculate_entropy=true`，确认 `losses.ppo_loss` 与 `engine_workers.ppo_loss` 均已被 `runtime_hooks.setup_worker_patches` 替换，而非仍走未带 hinge 的默认 `ppo_loss`。
3. **核查 advantage 分组与去长度归一化（6.3）**：检查 `_compute_advantage` 是否按 `uid` 分组且未混入 `is_padding` 合成样本；确认阶段一 `loss_scale_factor=8192`、阶段二同步切到 `16384`，避免切换上下文长度后梯度幅值突变。
4. **核查溢出缓冲 TTL 与重要性比（6.4）**：检查 `training/overflow_buffer/reused_ratio`、`reuse_paused` 与 `reused_is_ratio_raw_max`，确认缓冲样本仅保留 1 步且 `rollout.calculate_log_probs=true` 已开启；若 `reused_is_clipped_frac` 过高，优先降低学习率或收紧 `max_reuse_ratio`。
