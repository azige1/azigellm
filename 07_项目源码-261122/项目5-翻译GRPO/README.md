# 项目5-翻译GRPO

## 项目简介

本项目面向中英双向翻译的后训练强化学习阶段，基于 `verl 0.9.1` V1 同步 GRPO 框架（单轮生成 + LoRA），解决单一 LLM judge 总分导致策略过度迎合流畅度、术语错误难以下降，以及重复/超长失败样本污染组内 advantage 估计的问题。核心机制包括：

1. **Pointwise MQM 与位置交换 Pairwise 混合 judge**：judge 仅输出错误片段、维度与严重度，由程序按 `25 / 5 / 1` 加总扣分；分差小于 `2.0` 的近分译文对转入交换左右位置的双次 Pairwise 比较。
2. **7 维翻译 reward**：五维 MQM（准确、术语、流畅、风格、语境/本地化）独立加权（术语 `1.5`、准确 `1.2`、其余 `1.0`），叠加 4-gram 重复惩罚（`λ_rep = 10`）与长度比惩罚（`λ_len = 5`，10 词以下源句豁免）。
3. **参考译文质量锚点组内 advantage 修正**：每组 8 个策略候选混入 1 条强模型参考译文匿名打分，在组内标准化项上叠加 batch whitening 的 `λ·tanh` 修正项（`λ = 0.3`），参考译文不参与组统计与 loss。
4. **重复与超长输出的 Hard Mask 与梯度隔离**：4-gram 重复率 `> 0.3`、译文长度比 `> 2.5` 或达到 `1024` token 上限的无效样本整条屏蔽 `response_mask / loss_mask`，并从组均值/标准差中剔除；有效样本不足 2 条的组整组跳过。
5. **在线 judge 一致性诊断与外部强模型选点**：抽样 rollout 组对比在线 judge 与外部强模型（如 `gpt-5`）的相关性与分歧分布；每 20 步 checkpoint 在 `temperature=0.7` 下由外部强模型按 MQM 与 Pairwise 双口径判定并检测过优化，训练 reward 不参与选点。

> **与其他项目的衔接**：本项目与项目一至项目四无代码或产物依赖，独立从外部 LoRA SFT checkpoint（默认 `POLICY_SFT_MODEL_PATH`）及中英平行语料 `.jsonl` 起步。

---

## 目录结构

```text
项目5-翻译GRPO/
├── pyproject.toml
├── README.md
├── configs/
│   ├── grpo_8xa800_qwen3_8b.yaml      # 8×A800 运行档位配置（GPU0-5 训练，Qwen3-8B LoRA）
│   ├── grpo_ref_30b_a3b.yaml          # 简历规模参考配置（8×A100 策略 + 4×A100 judge，Qwen3-30B-A3B）
│   ├── translation_grpo.yaml          # 自定义 judge / 7 维权重 / 锚点 λ / hard mask 参数
│   └── judge_eval.yaml                # 5.1 金标校准与 5.5 外部强模型选点配置
├── data/
│   ├── sample_zh2en.jsonl             # 中译英小样例（10 条）
│   ├── sample_en2zh.jsonl             # 英译中小样例（10 条）
│   └── gold_calibration_sample.jsonl  # 金标校准小样例
├── experiments/
│   └── README.md                      # 4 组预期实验设计与排查指南
├── scripts/
│   ├── start_judge_vllm.sh            # 在 GPU6-7 启动 Qwen3-32B vLLM judge 服务
│   ├── prepare_data.sh                # 数据构建脚本（生成平衡 parquet/jsonl）
│   ├── run_train_8xa800.sh            # 启动 8×A800 GRPO 训练
│   ├── calibrate_judge.sh             # 运行 5.1 金标校准
│   └── eval_checkpoints.sh            # 运行 5.5 一致性诊断与选点
├── tests/
│   ├── test_mqm_protocol.py
│   ├── test_reward_scoring.py
│   ├── test_advantage_mask.py
│   └── test_data_and_config.py
└── translation_grpo/
    ├── __init__.py
    ├── judge_client.py                # 异步 OpenAI 兼容 judge 客户端（并发、超时重试、缓存、模板）
    ├── mqm_protocol.py                # 5.1 Pointwise MQM + 位置交换 Pairwise + 8+1 打乱混合打分
    ├── reward_scoring.py              # 5.2 7 维 reward + 5.4 重复/长度检测 + verl custom_reward_function
    ├── advantage_mask.py              # 5.3 锚点 advantage 修正 + 5.4 hard mask 与梯度隔离校验
    ├── trainer.py                     # TranslationGRPOTrainer(PPOTrainerSync) 覆写 _compute_advantage / _validate
    ├── train_main.py                  # @ray.remote TranslationTaskRunner + Hydra 训练入口
    ├── data_prep.py                   # 平行语料平衡与 sample_id 对齐构建
    ├── calibration.py                 # 5.1 金标一致率与偏差统计脚本
    └── checkpoint_eval.py             # 5.5 一致性诊断、符号检验、过优化检测与选点
```

---

## 环境与安装

- Python `>= 3.10`，`PyTorch >= 2.4`，`verl==0.9.1`（`verl[vllm]`），`vllm`，`aiohttp`，`datasets`，`omegaconf`，`hydra-core`。
- 8 × A800 80G 节点建议：GPU0–5 用于策略训练与异步 rollout，GPU6–7 用于 `vLLM` 部署在线 judge（`Qwen3-32B`, TP=2）。

```bash
pip install -e ".[dev]"
```

---

## 运行步骤

1. **准备平衡的中英双向 RL 数据**（按 `sample_id` 对齐参考译文与长度）：
   ```bash
   bash scripts/prepare_data.sh
   ```
2. **启动在线 Judge 服务（GPU6–7）**：
   ```bash
   JUDGE_MODEL_PATH=/path/to/Qwen3-32B bash scripts/start_judge_vllm.sh
   ```
3. **（可选）金标校准 Judge 协议**：
   ```bash
   bash scripts/calibrate_judge.sh data/gold_calibration_sample.jsonl
   ```
4. **启动 GRPO 训练（GPU0–5）**：
   ```bash
   POLICY_SFT_MODEL_PATH=/path/to/qwen3-8b-sft bash scripts/run_train_8xa800.sh
   ```
5. **外部强模型一致性诊断与 Checkpoint 选点**（需自备外部 API key）：
   ```bash
   export STRONG_JUDGE_API_KEY="your-api-key"
   bash scripts/eval_checkpoints.sh consistency outputs/rollout_groups.jsonl
   bash scripts/eval_checkpoints.sh select outputs/eval_manifest.jsonl
   ```

---

## 数据示例

取自 `data/sample_zh2en.jsonl` 与 `data/sample_en2zh.jsonl`：

```json
{"sample_id": "zh2en-0001", "direction": "zh2en", "domain": "legal", "source": "本合同自双方签字盖章之日起生效，有效期三年，任何一方不得单方面变更或解除。", "reference": "This contract shall take effect upon signature and seal by both parties and remain valid for three years, during which neither party may unilaterally amend or terminate it."}
{"sample_id": "en2zh-0002", "direction": "en2zh", "domain": "tech", "source": "Gradient checkpointing trades extra recomputation during backpropagation for a substantial reduction in activation memory footprint.", "reference": "梯度检查点通过在反向传播期间增加重计算开销，大幅降低了激活值的显存占用。"}
```

---

## 技术点与代码落点

| 技术点 | 核心机制 | 代码落点 |
|---|---|---|
| **5.1** 混合 MQM 与位置交换 Pairwise judge | 仅标错误片段/维度/严重度，按 `25/5/1` 计扣分；分差 `< 2.0` 转位置交换 Pairwise，两次一致才计胜负；金标校准一致率与偏差 | `translation_grpo/judge_client.py`、`translation_grpo/mqm_protocol.py`、`translation_grpo/calibration.py` |
| **5.2** 7 维翻译 reward | 五维权重（术语 `1.5`、准确 `1.2`、其余 `1.0`）+ 4-gram 重复惩罚（`10`）+ 长度比惩罚（`5·max(0, ratio-1.5)`，10 词以下豁免）；分维度监控与连续两次上升报警 | `translation_grpo/reward_scoring.py`、`translation_grpo/trainer.py` |
| **5.3** 参考译文质量锚点组内修正 | 8 候选 + 1 匿名参考译文混入打分，组内标准化后叠加 `λ·tanh((r_i - r_anc)/σ_B)`（默认 `λ=0.3`），锚点不计入组统计与 loss | `translation_grpo/mqm_protocol.py`、`translation_grpo/advantage_mask.py`、`translation_grpo/trainer.py` |
| **5.4** 重复与超长 Hard Mask 与梯度隔离 | `rep_rate > 0.3` 或 `len_ratio > 2.5` 或 `response_len >= 1024` 置零 `response_mask / loss_mask` 并剔除组统计；有效不足 2 条整组跳过；定期梯度隔离校验 | `translation_grpo/reward_scoring.py`、`translation_grpo/advantage_mask.py`、`translation_grpo/trainer.py` |
| **5.5** Judge 一致性诊断与强模型选点 | 抽样对比在线 judge 与外部强模型 Pearson 相关及 top-1 分歧；`temperature=0.7` 下每 20 步 checkpoint 做 MQM + Pairwise 符号检验与过优化告警，不看训练 reward 选点 | `translation_grpo/checkpoint_eval.py`、`configs/judge_eval.yaml` |

---

## 配置项

- `configs/grpo_8xa800_qwen3_8b.yaml`（运行档位）：6 卡训练/rollout，32 源句 × 8 候选，LoRA `rank=64, alpha=128`，`lr=1e-5` constant，无 KL，clip `0.2 / 0.28`，200 步，每 20 步保存与验证。
- `configs/grpo_ref_30b_a3b.yaml`（简历规模参考配置，仅作集群参数参考）：`Qwen3-30B-A3B` LoRA，8×A100 80G 策略 + 4×A100 judge，每步 64 源句 × 8 候选。
- `configs/translation_grpo.yaml`：judge 接口、7 维权重、锚点 `λ=0.3`、hard mask 阈值与检查频率。
- `configs/judge_eval.yaml`：外部强模型名称（默认 `gpt-5`）、环境变量名（`STRONG_JUDGE_API_KEY`）与选点参数。

---

## 评测指标与预期结果

详见 [experiments/README.md](experiments/README.md)。核心评测口径为外部强模型在评测集上计算的**平均 MQM penalty**（越低越好）及对 SFT 起点的 **Pairwise 去平局胜率**（含符号检验 `z` 与 `p` 值）。

---

## 测试

无需 GPU 与外网即可运行全部单元测试：

```bash
pytest -q
```
