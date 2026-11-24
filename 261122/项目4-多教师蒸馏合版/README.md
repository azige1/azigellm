# 多领域代码专家的多教师 on-policy 蒸馏合版

## 项目简介

本项目面向 Qwen3-8B 同族模型（规模参照档为 Qwen3-32B），将分散在函数级代码、仓库级实现、issue 修复、测试生成、数学与科学推理、指令遵循、工具调用及通用对话等领域的专项教师能力，通过 on-policy 蒸馏合并回同一学生模型。十个候选领域经跨域能力矩阵归并为八个常驻教师，另设 `self_anchor`（8% 采样份额）路由至冻结学生起点以抑制通用能力回退。学生每题采样 1 条轨迹，按领域路由到对应教师在完全一致的 token 序列上重算 log-prob，通过 `sampled_reverse_kl`（策略梯度路径）完成更新。

## 与其他项目的衔接

- **学生起点（统一 SFT）**：读取统一 SFT 产出的 HuggingFace checkpoint（`global_step_N/huggingface`），该模型在仓库级轨迹（来自项目二轨迹数据）及其余领域 SFT 数据上完成统一微调。
- **仓库级实现教师（`repo_impl`）**：直接读取项目三长程 RL 产出的 HuggingFace checkpoint 作为常驻教师，不再单独训练；该教师经 `admission.py` 校验同源 tokenizer 后进入教师池。
- **仓库级 rollout 与沙箱适配（独立适配模块）**：`multi_teacher_distillation/domains/project3_adapter.py` 通过配置项 `project3_adapter.agent_loop_path` 与 `project3_adapter.sandbox_pool_path` 加载项目三的多轮 agent loop 与沙箱容器池，默认值为 `long_horizon_rl.agent_loop:CodingCompactionAgentLoop` 与 `long_horizon_rl.sandbox_pool:SandboxPool`，可通过环境变量 `PROJECT3_AGENT_LOOP_PATH`、`PROJECT3_SANDBOX_POOL_PATH` 或 Hydra 覆盖；每个 rollout worker 进程共用一个容器池。

## 目录结构

```text
multi_teacher_distillation/
  __init__.py                  十候选域、八合并教师与 self_anchor 份额常量
  main.py                      Ray TrainingRunner 与 Hydra 入口
  trainer.py                   DistillationTrainer：路由器、TQ 张量回写与状态保存
  losses.py                    sampled_reverse_kl 注册与领域梯度份额重加权
  teacher_merging.py           4.1 跨域能力矩阵教师合并与子类拆分
  warmup_sft.py                4.2 低重合领域检查、按缺口采样与整段渲染 SFT 数据集
  routing.py                   4.3 分段元数据传递、四级路由校验与版本重算
  reliability_weighting.py     4.4 教师校准位移 10 桶提升倍数与可靠性权重
  domain_advantage.py          4.5 领域 token 加权滑动标准差归一（±5 -> ±3）
  admission.py                 同源 tokenizer、增益、通用回归与 bootstrap 下界检查
  fixed_sampling.py            跨步长整数配额固定领域采样器
  rollout_masking.py           截断整条掩码、重复率统计与诊断十分组
  termination.py               模板后缀、终止 token 与固定 prompt 概率核对
  metrics.py                   恢复率与八领域汇总指标
  evaluate.py                  文本/代码评测与八领域恢复率汇总
  patches.py                   verl 0.9.1 外部教师与空掩码安全补丁
  worker_setup.py              Ray worker setup hook（注册损失与运行时补丁）
  domains/
    project3_adapter.py        项目三 agent loop 与沙箱独立适配模块及 issue 容器配置
    coding_loops.py            单轮与连续 token 多轮领域 rollout loop
    verifiers.py               八领域 verifier（含交换位置成对比较 judge）
    entrypoints.py             agent loop 注册导出
  teacher/
    schemas.py                 ScoreRequest / PositionScore / ScoreResponse 协议
    resident_service.py        每卡两 Qwen3-8B 教师的常驻服务与分块 log-softmax
    pool_requests.py           异步教师池 HTTP 客户端与版本校验
    pool_service.py            FastAPI 教师池服务入口
  data/
    build.py                   十域转八教师标准化与 Parquet 题池构建
    decontaminate.py           13-gram 与模板/实体去污
experiments/
  configs/
    distill_8xa800.yaml        8×A800 默认蒸馏配置（GPU0-3 训练与 rollout）
    distill_32x_h100_reference.yaml  Qwen3-32B 32+32+16 H100 规模参考配置
    distill_smoke_8xa800.yaml  2 步冒烟配置
    teacher_rl_8xa800.yaml     7 个非仓库领域教师的 GRPO 训练配置
    warmup_sft_8xa800.yaml     4.2 分域预热 SFT 配置（sft_trainer_engine）
    agent_loop.yaml            九路 agent loop 目标配置
    issue_fix_container.yaml   issue 修复容器参考配置
    evaluation.yaml            八领域评测基准与指标配置
  README.md                    四组预期实验设计与预期结果
scripts/                       合并、预热 SFT、教师 RL、准入、教师池、蒸馏与评测脚本
tests/                         无 GPU/网络依赖的 pytest 契约测试
data/examples/                 输入样本与能力矩阵示例
```

## 环境与安装

固定版本：Python 3.12、`verl[vllm]==0.9.1`、vLLM 0.24.0、transformers 5.9.0、torch 2.11。学生、八个教师与 `self_anchor` 必须共用同一 Qwen3 tokenizer，全流程设置 `enable_thinking=False`。

```bash
pip install -e .
# 仓库级实现领域复用项目一至三的环境、判分与 agent loop：
pip install -e ../项目1-函数级任务合成 -e ../项目2-长程轨迹增强蒸馏 -e ../项目3-编码Agent长程RL
```

## 运行步骤

1. **训练候选教师并执行能力矩阵归并（4.1）**：除 `repo_impl` 直接使用项目三 RL checkpoint 外，其余领域从统一 SFT 起步做 GRPO；评测 10 × 10 能力矩阵后执行合并与拆分：
   ```bash
   ABILITY_MATRIX_JSON=data/examples/ability_matrix.json TEACHER_MAP_OUTPUT=outputs/domain_to_teacher.json bash scripts/merge_teacher_pool.sh
   ```
2. **准入检查与分域预热 SFT（4.2）**：检查各领域学生起点 rollout 的 reverse KL 与 top-20 重叠率；对不达标领域按缺口采样做 1 epoch SFT（`lr=5e-6`、batch 32）：
   ```bash
   python -m multi_teacher_distillation.warmup_sft check --input data/overlap_rollouts.jsonl --output outputs/overlap_report.json
   python -m multi_teacher_distillation.warmup_sft prepare --overlap-report outputs/overlap_report.json --teacher-samples data/teacher_verified.jsonl --output data/warmup_sft.parquet
   MODEL_WEIGHTS_PATH=/path/to/unified_sft WARMUP_SFT_DATA=data/warmup_sft.parquet bash scripts/run_warmup_sft.sh
   ```
3. **构建蒸馏题池与校准表（4.4）**：构建标准化 Parquet 题池并基于 512 条/域校准 rollout 生成可靠性分桶表：
   ```bash
   INPUT_MANIFEST=data/manifest.json OUTPUT_DATA=data/distill.parquet POOL_STATS=outputs/pool_stats.json DOMAIN_MAP_PATH=outputs/domain_to_teacher.json bash scripts/prepare_datasets.sh
   CALIBRATION_ROLLOUTS=data/calibration_rollouts.jsonl CALIBRATION_TABLES_OUTPUT=outputs/calibration_tables.json bash scripts/build_calibration.sh
   ```
4. **启动教师常驻池（GPU 4–7）与模板核对**：
   ```bash
   TEACHER_GPUS=4,5,6,7 bash scripts/start_teacher_pool.sh
   MODEL_WEIGHTS_PATH=/path/to/warmed_sft FIXED_PROMPTS=data/examples/fixed_prompts.jsonl TERMINATION_REPORT=outputs/termination.json bash scripts/check_termination.sh
   ```
5. **启动多教师 on-policy 蒸馏（GPU 0–3 暴露给 Ray）**：
   ```bash
   TRAINING_DATA_FILE=data/distill.parquet VALIDATION_DATA_FILE=data/distill_val.parquet JUDGE_BASE_URL=http://127.0.0.1:8001/v1 \
   TERMINATION_REPORT=outputs/termination.json bash scripts/run_distillation.sh
   ```
6. **八领域评测**：`bash scripts/evaluate.sh all`。

## 数据示例

取自 [data/examples/prompts.jsonl](data/examples/prompts.jsonl) 与 [data/examples/ability_matrix.json](data/examples/ability_matrix.json)：

```json
{"prompt_id":"issue-00001","domain":"issue_fix","task_id":"issue-07731","instruction":"修复 parse() 在缺少 sep 参数时抛出 KeyError 的问题。"}
```

## 技术点与代码落点

- **4.1 跨域能力矩阵的教师池合并与拆分**：`multi_teacher_distillation/teacher_merging.py`（`build_teacher_pool_from_matrix`、`_pair_within_noise_band`）与 `scripts/merge_teacher_pool.sh`。按 reward 来源类型相同且双向分差均小于 2 倍标准差噪声带合并；若子类低于起点则拆分，输出教师清单与 `domain_to_teacher` 映射。
- **4.2 低重合领域的分域预热 SFT**：`multi_teacher_distillation/warmup_sft.py`（`evaluate_domain_overlap`、`allocate_warmup_samples`、`filter_verified_teacher_samples`、`DomainWarmupSFTDataset`、`verify_post_warmup_overlap`）、`experiments/configs/warmup_sft_8xa800.yaml`（`defaults: [sft_trainer_engine, _self_]`）与 `scripts/run_warmup_sft.sh`。检查 reverse KL `<= 0.15` 且 top-20 重叠率 `>= 0.75`，仅对不达标领域按缺口比例采样并在 `DomainWarmupSFTDataset` 中整段渲染多轮对话、仅对 assistant token 计 loss。
- **4.3 分段样本的路由校验与 log-prob 重算**：`multi_teacher_distillation/routing.py`（`propagate_segment_metadata`、`validate_routing_stage`、`filter_cached_logprobs_by_version`、`check_routing_share_drift`）、`multi_teacher_distillation/domains/project3_adapter.py`、`multi_teacher_distillation/trainer.py` 与 `multi_teacher_distillation/teacher/resident_service.py`。轨迹分段时显式传递 `domain`、`teacher`、`teacher_version` 与 `segment_index`；在数据构建、rollout 输出、教师请求、loss 前四级校验路由一致，版本不一致时丢弃缓存并在与学生完全一致的 token 序列上重算（按第 `i` 行隐状态预测第 `i+1` 个 token 对齐）。
- **4.4 教师校准位移的 token 级可靠性加权**：`multi_teacher_distillation/reliability_weighting.py`（`fit_calibration_table`、`score_token_reliability_weights`、`build_batch_reliability_weights`、`CalibrationManager`）、`multi_teacher_distillation/admission.py`、`multi_teacher_distillation/trainer.py` 与 `multi_teacher_distillation/losses.py`。按相对冻结学生起点的位移分 10 桶计算提升倍数与平方可靠性权重，域内均值归一到 1 并乘准入下界系数 `min(1, LB_T / 2)`，写入 TQ 张量字段 `distillation_token_weights` 进入 loss，每 40 步刷新。
- **4.5 领域滑动标准差归一的 token 级 advantage**：`multi_teacher_distillation/domain_advantage.py`（`DomainAdvantageNormalizer`）、`multi_teacher_distillation/trainer.py` 与 `multi_teacher_distillation/losses.py`。原始 log-prob 差裁剪到 `±5`，按领域（含 `self_anchor`）维护 token 数加权滑动标准差（衰减 `0.9`、下限 `0.2`），归一后裁剪到 `±3` 并乘 4.4 权重，再按实际梯度 token 量重加权对齐静态目标份额；只对学生生成 token 计 loss，走 `use_policy_gradient: true`、`policy_loss_mode: vanilla` 路径。

## 配置项

- `8×A800` 默认配置（[distill_8xa800.yaml](experiments/configs/distill_8xa800.yaml)）：学生 Qwen3-8B 使用 GPU 0–3（`CUDA_VISIBLE_DEVICES=0,1,2,3`），GPU 4–7 常驻 8 个 Qwen3-8B 教师（每卡 2 个）；每步 64 题 × 1 条、`lr=1e-6`、warmup 10 步；关闭 `algorithm.rollout_correction.bypass_mode` 以便对分段轨迹重算 `old_log_probs`。
- `32+32+16 H100` 规模参考配置（[distill_32x_h100_reference.yaml](experiments/configs/distill_32x_h100_reference.yaml)）：Qwen3-32B 学生与 8 个教师，learner 32×H100、rollout 32×H100、教师池 16×H100，每步 512 题 × 1 条、160 步、每 20 步评测。
- `issue_fix` 容器参考配置（[issue_fix_container.yaml](experiments/configs/issue_fix_container.yaml)）：预热池大小 10、写时复制快照复位、单轮超时 300s、最多 120 轮与 4 段压缩。

## 评测指标与预期结果

评测八个领域（`LiveCodeBench v6`、96 道 held-out 仓库任务、`SWE-bench Verified`、held-out 变异杀死率、`AIME 2025`、`IFEval`、`BFCL v3`、`Arena-Hard v2` 使用 `GLM-5`）。主指标为各领域恢复率 `(学生 - 起点) / (教师 - 起点)` 与八领域均分，完整对照设计与预期结果见 [experiments/README.md](experiments/README.md)。

## 测试

```bash
pytest -q
```
