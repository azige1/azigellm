# 预期实验过程与预期结果

本文档描述多教师 on-policy 蒸馏合版在本地 **8×A800（Qwen3-8B）** 档位的预期实验设计与过程判别信号，以及与 **Qwen3-32B 规模参照档** 对齐的逐项累加对照与消融设计。所有实验固定统一 SFT 起点、去污后的题池与八领域评测协议，主指标为八个领域的恢复率 `(学生 - 起点) ÷ (教师 - 起点)` 与八领域均分。

## 通用观察指标与正常/异常信号

- 过程指标：`distillation/token_kl/{domain}`、`distillation/domain_sigma/{domain}`、`distillation/step_std/{domain}`、`distillation/gradient_token_share/{domain}`、`distillation/gradient_share_gap/{domain}`、`distillation/routing_share_drift/{domain}`、`distillation/routing_share_alarm`、`distillation/mean_token_reliability_weight`、`rollout/truncation_rate/{domain}`。
- 预期正常信号：截断率稳定在 1%–4%；各领域路由份额偏差绝对值小于 1%（`routing_share_alarm = 0`）；`domain_sigma` 在前 20 步收敛到各自领域尺度区间（agent 类高于短回复类约 3–4 倍）并平滑缓降；经重加权后有效梯度份额对齐目标份额（`gradient_share_gap` 趋近 0）。
- 异常信号：路由告警持续触发或 `routing_dropped/* > 0`；某领域 token KL 剧烈震荡或反向升高；`instruction_following` 与 `general_chat` 恢复率在前 60 步几乎停滞；`issue_fix` 重复测试调用率随训练攀升。

---

## 第一组：基线与统一 RL 对照

### 目的

建立统一 SFT 起点、数据混合统一 GRPO 对照，以及十候选领域一对一基础 MOPD 基线，验证多教师蒸馏相对单模型混合 RL 的多域合版收益。

### 运行命令与配置

- 统一 SFT 起点评测：`bash scripts/evaluate.sh all`
- 数据混合统一 RL 对照：使用 `teacher_rl_8xa800.yaml` 在混合八领域数据池上直接做 GRPO。
- 基础 MOPD（十教师一对一）：关闭 4.1 合并、4.2 预热 SFT、4.3 严格路由校验、4.4 可靠性加权（`reliability_weighting.enabled=false`）与 4.5 领域标准差归一（`domain_advantage.enabled=false`），执行 `bash scripts/run_distillation.sh`。

### 预期训练过程与结果

- **统一 SFT 起点（Qwen3-8B）**：预期八领域均分位于 **44.5–46.5** 区间，长程仓库类与 `general_chat` 仍有较大提升空间。
- **数据混合统一 RL 对照**：不同领域 reward 尺度与轨迹长度差异导致短回复域恢复受限，预期八领域均分相对起点提升约 **+1.8 至 +2.4**。
- **基础 MOPD（十教师一对一）**：由于按学生 token 逐点提供 dense 信号，预期均分相对起点提升约 **+2.8 至 +3.3**，明显高于混合 RL 对照，但在低重合域、分段路由与跨领域尺度均衡上仍存在提升空间。

---

## 第二组：默认配置全链路训练（Qwen3-8B 8×A800）

### 目的

验证集成 4.1（能力矩阵八教师）、4.2（分域预热 SFT）、4.3（分段路由校验与重算）、4.4（校准位移可靠性加权）与 4.5（领域滑动标准差归一）的完整默认流程。

### 运行命令与配置

- 依次运行 `scripts/merge_teacher_pool.sh`、`scripts/run_warmup_sft.sh`、`scripts/prepare_datasets.sh`、`scripts/build_calibration.sh`、`scripts/start_teacher_pool.sh`（GPU 4–7）、`scripts/check_termination.sh` 与 `scripts/run_distillation.sh`（`distill_8xa800.yaml`，GPU 0–3）。

### 预期训练过程

- 4.1 归并后：`func_gen` 与 `competitive_coding` 合并为 `code_gen`，`math` 与 `science` 合并为 `math_science`，`repo_impl` 与 `issue_fix` 因双向分差超出 2 倍标准差噪声带保持独立，最终得到 8 个教师与 `self_anchor`（8%）。
- 4.2 分布检查预期识别出 `repo_impl` 与 `tool_calling` 低于重合门槛（KL > 0.15 或 top-20 重叠率 < 0.75）；按缺口比例 3:1 分配 4,000 条教师验证样本做 125 步预热 SFT 后，两域重叠率升至 0.80 附近且其余已达标领域重叠率下降不超过 0.02。
- 4.3 全程无分段路由丢失至 `self_anchor`，每步份额偏差控制在 1% 以内；4.4 每 40 步刷新一次分桶表，高位移且与失败相关的冗余测试调用 token 权重降至中位桶的约三分之一；4.5 开启后 `instruction_following` 与 `general_chat` 的前 60 步 KL 下降速度明显加快。

### 预期结果

- 在 Qwen3-8B 8×A800 档位，完整默认配置相对统一 SFT 起点预期累计提升 **+4.5 至 +5.2**（八项均分预期 **49.5–51.5**，八领域平均恢复率预期 **76%–80%**，最弱领域恢复率预期 **≥70%**）。

---

## 第三组：技术点逐项累加对照

### 目的

验证 4.1–4.5 每项改进的独立增量及增量之和与总提升的一致性。

### 预期累加表（Qwen3-8B 8×A800 档预期增量区间与参照方向）

| 配置 | 八领域均分预期区间 | 相对上一行预期增量 | 对齐规格方向 |
| --- | :---: | :---: | --- |
| 统一 SFT（学生起点） | 45.5 | — | 起点基线 |
| 基础 MOPD（十候选领域一对一设教师） | 48.5 | +3.0 | 显著高于数据混合统一 RL 对照（约 +2.0） |
| 改为跨域能力矩阵的教师池合并与拆分（4.1，八教师） | 49.0 | +0.5 | 题池合并与减少冗余教师服务卡时 |
| 加低重合领域的分域预热 SFT（4.2） | 49.25 | +0.25 | `repo_impl` 与 `tool_calling` 恢复率显著提升 |
| 加分段样本的路由校验与 log-prob 重算（4.3） | 49.50 | +0.25 | 消除分段误路由到 `self_anchor`，提升 `repo_impl` |
| 加教师校准位移的 token 级可靠性加权（4.4） | 49.90 | +0.40 | 抑制 `issue_fix` 重复测试调用，提升 SWE 类通过率 |
| 加领域滑动标准差归一的 token 级 advantage（4.5） | 50.25 | +0.35 | 提升 `instruction_following` 与 `general_chat` 恢复率 |

各步增量之和：`3.00 + 0.50 + 0.25 + 0.25 + 0.40 + 0.35 = 4.75`，与总提升 `50.25 - 45.50 = 4.75` 一致；改进方向与相对幅度均与 32B 规模参照档累加表一致。

---

## 第四组：核心维度消融（8×A800 Qwen3-8B）

| 消融维度 | 配置改动 | 预期相对影响 | 重点排查指标 |
| --- | --- | --- | --- |
| 噪声带倍数扫描（4.1） | `teacher_merging.noise_multiplier` 改 `1.0` 或 `3.0` | `1.0` 仅合并 1 对、卡时增加；`3.0` 误并入通用域导致指令遵循恢复率下降约 4–5pp | `merge_decisions`、`instruction_following` 恢复率 |
| 全域预热 vs 分域预热（4.2） | 对已达标领域也采样 1,000 条做预热 | 已达标领域分布向教师风格偏移，重叠率回落约 0.04–0.06 | `compliant_overlap_drops` |
| 关闭分段路由严格校验（4.3） | `routing_validation.strict_mode` 改忽略并回退默认 | `repo_impl` 分段误路由至 `self_anchor`，仓库级恢复率下降约 4–6pp | `routing_share_drift`、`routing_dropped` |
| 可靠性幂次扫描（4.4） | `reliability_weighting.alpha` 改 `1.0` 或 `4.0` | `1.0` 对重复调用抑制不足；`4.0` 过度压低有效信号导致 `issue_fix` 恢复率下降 | `mean_token_reliability_weight`、重复调用率 |
| 滑动衰减与下限扫描（4.5） | `ema_decay` 改 `0.5`/`0.99`，或 `sigma_floor` 改 `0.1` | `0.5` 有效学习率剧烈波动；`0.99` 后期短回复域过放大；`0.1` 使 `self_anchor` 偏离起点 | `domain_sigma/{domain}`、`self_anchor` 分数 |
| 关闭领域梯度份额重加权 | `domain_reweighting.domain_token_reweight: false` | 长序列 agent 域主导更新，短回复域恢复明显变慢 | `gradient_share_gap/{domain}` |

---

## 偏离预期时的统一排查顺序

1. **版本与模板一致性**：检查 `admission.py` 同源 tokenizer 检查与 `termination.py` 终止 token log-prob 差（不得超过 2.0）。
2. **4.1 能力矩阵与题池映射**：检查 `domain_to_teacher.json` 是否将 10 个候选域正确映射到 8 个教师且 `repo_impl` 与 `issue_fix` 保持独立。
3. **4.2 预热复检**：检查 `verify_post_warmup_overlap` 报告，确认未出现已达标领域重叠率下降超过 0.02 的现象。
4. **4.3 分段路由与序列对齐**：检查 `routing_share_alarm`、各阶段校验计数，以及教师打分第 `i` 行预测第 `i+1` 个 token 与 `bypass_mode: false` 重算 `old_log_probs`。
5. **4.4 与 4.5 张量流转**：确认 `distillation_advantages`、`distillation_token_weights` 与 `distillation_domain_weights` 均通过 TQ 嵌套张量写入并在 `compute_sampled_reverse_kl` 中生效。
6. **资源与卡组隔离**：核对训练脚本暴露 `CUDA_VISIBLE_DEVICES=0,1,2,3`，教师服务暴露 `CUDA_VISIBLE_DEVICES=4,5,6,7`。
