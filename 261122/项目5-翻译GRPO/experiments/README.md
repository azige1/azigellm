# 实验设计与预期结果

本文档给出 `项目5-翻译GRPO` 在 **8 × A800 80G** 档位（策略 `Qwen3-8B` + LoRA `rank=64, alpha=128` 占 GPU0–5；在线 judge `Qwen3-32B` vLLM TP=2 占 GPU6–7；每步 32 条源句 × 8 候选、lr `1e-5` 恒定、无 KL、clip `0.2/0.28`、总步数 200 步、每 20 步保存并评测）下的四组预期实验设计与判别准则。所有数值均为当前配置档位下的预期范围与逐项增量结构，用于实验对照与异常排查。

---

## 实验总览

四组实验覆盖：
1. **Baseline**：SFT 起点与单一口径总分 reward GRPO。
2. **默认完整配置**：混合 MQM+Pairwise judge + 7 维 reward + 参考译文质量锚点（λ=0.3）+ hard mask。
3. **核心维度消融**：`λ` 敏感性扫描（`0.1 / 0.3 / 0.6`）与 judge 协议一致率校准实验。
4. **技术点逐项对照（累加消融）**：从 SFT 起点逐项叠加各技术点，核验主指标预期增量之和等于总提升。

统一评测口径：
- **主指标 1**：外部强模型（如 `gpt-5`，`temperature=0.7` 解码）在业务评测集上标注的 **平均 MQM penalty**（越低越好）。
- **主指标 2**：位置交换 Pairwise 对 SFT 起点的**去平局胜率**与符号检验显著性（`p < 0.01`）。
- **辅助监控指标**：每百句术语错误数、5 维独立扣分（`accuracy / terminology / fluency / style / context`）、重复/超长无效样本占比、低于锚点组比例、在线 reward 与外部强模型 penalty 走势一致性。

---

## 第一组：Baseline（SFT 起点与总分 reward GRPO）

### 目的
建立 `Qwen3-8B` LoRA SFT 起点基准，并验证传统「0–100 总分 reward GRPO」在翻译任务上的流畅度偏置与术语停滞问题。

### 运行命令与配置
```bash
bash scripts/start_judge_vllm.sh
# 1) SFT 起点评测：Step 0 checkpoint 直接跑外部强模型评测
bash scripts/eval_checkpoints.sh select outputs/eval_manifest_sft.jsonl
# 2) 总分 reward GRPO：reward_mode=total_score，关闭质量锚点与 hard mask
python -m translation_grpo.train_main trainer.experiment_name=qwen3_8b_total_score_baseline
```
（对应 sidecar 配置 `configs/translation_grpo.yaml` 修改：`reward.reward_mode: total_score`、`advantage.anchor_enabled: false`、`advantage.hard_mask_enabled: false`、`hybrid_pairwise_in_trainer: false`）

### 预期训练过程
- **应观察指标**：`actor/pg_loss`、`actor/entropy`、`reward/direct_score`、外部评测每 20 步的 `strong_mqm_penalty_mean` 与 `terminology_penalty`。
- **预期曲线走向**：前 60 步在线总分平稳上升，但 `fluency_penalty` 下降明显而 `terminology_penalty` 几乎停滞；60–120 步后部分长句出现重复或超长输出，拉低同组均值并造成优势估计抖动。
- **正常信号**：KL/entropy 平滑下降不崩溃，总分从起步约 72 分缓慢抬升至 77–79 分。
- **异常信号**：`actor/entropy` 在 40 步前跌破 0.1（策略坍缩），或重复输出比例突破 6% 且奖励函数继续上升。

---

## 第二组：默认完整配置（7 维 reward + 锚点 λ=0.3 + Hard Mask + 混合 Judge）

### 目的
验证完整方案在 8 × A800 档位下能否同步压低术语错误、抑制重复/超长，并在外部强模型双口径下取得稳健选点。

### 运行命令与配置
```bash
bash scripts/start_judge_vllm.sh
bash scripts/run_train_8xa800.sh
# 每 20 步 checkpoint 在评测集 temperature=0.7 生成后运行选点与一致性诊断
bash scripts/eval_checkpoints.sh consistency outputs/rollout_groups_96.jsonl
bash scripts/eval_checkpoints.sh select outputs/eval_manifest_full.jsonl
```
默认配置为 `configs/grpo_8xa800_qwen3_8b.yaml` + `configs/translation_grpo.yaml` + `configs/judge_eval.yaml`。

### 预期训练过程
- **应观察指标**：`reward/terminology_penalty_mean`、`reward/accuracy_penalty_mean`、`reward/fluency_penalty_mean`、`grpo/invalid_sample_ratio`、`grpo/below_anchor_group_ratio`、`diag/hard_mask_invalid_adv_max`、`strong_mqm_penalty_mean`。
- **预期曲线走向**：
  - Step 0–120：五维扣分同步下行，其中术语维与准确维下降幅度领先总分 baseline；`grpo/invalid_sample_ratio` 从早期约 `3.0%–3.8%` 预期降至 `0.6%–0.9%`；`grpo/below_anchor_group_ratio` 从约 `35%` 逐步降至 `22%–26%`。
  - Step 120–140：外部强模型 `strong_mqm_penalty_mean` 预期达到最低谷，Pairwise 对 SFT 起点去平局胜率预期达到 `56%–59%`（`p < 0.01`）。
  - Step 140–200：在线 reward 预期继续小幅走高，但外部强模型 `strong_mqm_penalty_mean` 预期止降回升，连续 2 个保存点出现反向走势时触发过优化告警，选点脚本应选定 Step 120（而非 Step 200）。
- **正常信号**：`diag/hard_mask_invalid_adv_max == 0.0`、`diag/hard_mask_invalid_loss_mask_sum == 0.0`；含无效样本组与干净组的有效候选 advantage 均值差趋近 0。
- **异常信号**：任一 MQM 维度在验证集上连续 2 次上升（触发 `val-alert/*_consecutive_rise=1`），或 `grpo/skipped_group_count` 单步超过 `10%`。

---

## 第三组：核心参数与 Judge 协议消融

### 目的
1. 验证质量锚点强度 `λ` 在 `0.1 / 0.3 / 0.6` 三档对翻译质量与译文长度/流畅度的权衡。
2. 在人工双人金标集上对比四种 judge 协议与人工一致率、平局率与位置偏差。

### 运行命令与配置
```bash
# λ 扫描（修改 sidecar 配置中的 advantage.anchor_lambda）
TRANSLATION_GRPO_CONFIG=configs/translation_grpo_lam01.yaml bash scripts/run_train_8xa800.sh trainer.experiment_name=ablation_lam01
bash scripts/run_train_8xa800.sh trainer.experiment_name=ablation_lam03
TRANSLATION_GRPO_CONFIG=configs/translation_grpo_lam06.yaml bash scripts/run_train_8xa800.sh trainer.experiment_name=ablation_lam06

# Judge 协议金标校准（5.1）
bash scripts/calibrate_judge.sh data/gold_calibration_sample.jsonl
```

### 预期结果（λ 扫描，基于 Qwen3-8B 档位）
| `anchor_lambda` | 预期平均 MQM penalty（外部强 judge） | 相对 `λ=0` 增量 | 预期长度与流畅度表现 |
|---|---|---|---|
| `0.0`（无锚点，保留 7 维 + hard mask） | `1.294` | `—` | 较差组仍获得较高正更新，约 34% 组最高分低于锚点 |
| `0.1` | `1.280` | `-0.014` | 修正力度偏弱，改善接近评测波动 |
| `0.3`（默认） | `1.246` | `-0.048` | 较差组正 advantage 预期压低约 40%，流畅度无明显回退 |
| `0.6` | `1.272` | `-0.022` | 策略向参考译文风格过度靠拢，平均长度预期拉长约 10%–12%，流畅扣分反升 |

### 预期结果（Judge 协议校准与 5.5 一致性）
| 判分协议 | 预期与人工一致率（去平局） | 预期特征 |
|---|---|---|
| 直接给 0–100 总分 | `55%–58%` | 尺度随 prompt 漂移，同句波动大 |
| Pointwise MQM 错误标注 | `62%–65%` | 扣分可追溯至 5 维，但近分对区分度有限 |
| Pairwise 单次 | `72%–75%` | 存在明显首位置偏好 |
| Pointwise MQM + 位置交换 Pairwise（阈值 2） | `78%–81%` | 平局率约 `8%–10%`（集中在无错译文），显著消除位置偏差 |

在线 `Qwen3-32B` judge 与外部强模型在抽样 96 组（768 条候选）上的预期 Pearson 相关系数为 `0.70–0.75`，组内最优候选分歧率约 `14%–17%`（主要分布在长句与高术语密度句）。

---

## 第四组：技术点逐项对照（累加消融表）

### 目的
在 `Qwen3-8B`（8 × A800）代码档位下，验证 5.2–5.4 各项机制对最终翻译质量的独立贡献；5.1 judge 校准作为各行判分可信的前提，5.5 负责在训练曲线上定位最佳 checkpoint（预期 Step 120）。

### 预期结果表（8 × A800，Qwen3-8B LoRA，外部强模型评测）
| 配置（逐项累加） | 预期平均 MQM penalty（↓） | 相对上一行增量 | 预期 Pairwise 对 SFT 胜率（去平局） | 预期每百句术语错误 |
|---|---|---|---|---|
| 1. SFT 起点（Step 0） | `1.4200` | `—` | `—` | `4.6` |
| 2. + GRPO（总分 reward） | `1.3840` | `-0.0360` | `52.1%` | `4.5` |
| 3. + 改 7 维 MQM reward（5.2） | `1.3260` | `-0.0580` | `54.8%` | `3.5` |
| 4. + 质量锚点修正（5.3，`λ=0.3`） | `1.2780` | `-0.0480` | `56.9%` | `3.3` |
| 5. + Hard mask 与梯度隔离（5.4） | `1.2460` | `-0.0320` | `58.2%`（`p < 0.01`） | `3.2` |

> 逐项增量核对：`-0.0360 + (-0.0580) + (-0.0480) + (-0.0320) = -0.1740`，`1.4200 - 0.1740 = 1.2460`（相对 SFT 起点预期下降约 `12.3%`；7 维 reward 为单项最大贡献，各增量方向与相对占比与规格一致）。

---

## 偏离预期时的排查顺序

1. **检查 Hard Mask 与张量写回**：确认训练日志中 `diag/hard_mask_invalid_adv_max == 0.0` 且 `diag/hard_mask_invalid_loss_mask_sum == 0.0`；若非零，核查 `TranslationGRPOTrainer._compute_advantage` 中 `response_mask` 与 `loss_mask` 是否均通过 `response_to_nested` 写回 TransferQueue。
2. **检查组统计与 Padding 隔离**：核查 `grpo/adv_mean_groups_with_invalid` 与 `grpo/adv_mean_clean_groups` 是否接近；若含无效样本组优势显著偏高，检查 `invalid_mask` 或 `is_padding_mask`（来自 `batch.tags`）是否漏剔除。
3. **检查 Judge 连通性与解析率**：抽查 `outputs/judge_cache/online_judge.jsonl` 是否出现大量空 `errors`（JSON 解析失败会误判为 0 扣分）；若存在，检查 vLLM judge 的 `--max-model-len` 与输出截断，或调高 `judge.timeout_s`。
4. **检查锚点匿名性与 `λ` 幅度**：确认参考译文与候选译文使用完全一致的 `POINTWISE_MQM_PROMPT_TEMPLATE`，且 `grpo/anchor_sigma_batch` 未退化至 `1e-6`；若译文明显变长且流畅度下降，先将 `anchor_lambda` 从 `0.6` 回调至 `0.3`。
5. **检查解码口径与选点信号**：确认选点评测固定 `temperature=0.7`（与 rollout/线上一致），不要用 greedy 结果或在线 reward 最高点（Step 200）替代外部强模型曲线选点。
