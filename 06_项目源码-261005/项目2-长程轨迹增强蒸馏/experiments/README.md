# 实验说明：预期过程与预期结果

## 规模与公共前置

实验按两档规模组织：

| 规模 | 任务 | rollout | 入训样本 | 训练步数 | 用途 |
| --- | --- | --- | --- | --- | --- |
| 流程验证 | `seed_repos.yaml` 默认 8 个（6 训练 / 2 评测） | 每任务 24 次，共 144 条 | 约 100 条 | 2 epoch 约 13 步 | 确认全链路跑通、loss 正常下降，行为指标不作结论 |
| 完整对照 | 按同样字段把清单扩到约 30 个（24 训练 / 6 评测） | 每任务 16 次，共约 384 条 | 约 290 条 | 2 epoch 约 36 步 | 下文各预期结果表按此规模给出 |

学生为 Qwen3-8B 全参 SFT，上下文 32K。此档下推理压缩阈值取 24K（teacher 采集 100K），压缩恢复切分阈值取 16K，切分后 B 段才放得进 32K。训练步数少，`--save_steps` 与 `--eval_steps` 需要调小：流程验证取 5，完整对照取 10，经 `sft_qwen3_8b.sh` 的第三个参数起传入。

公共前置（四组实验共用）：

```bash
python -m task_builder build --repos task_builder/seed_repos.yaml --out tasks/
python -m task_builder prepare-host-env --tasks tasks/
export OPENAI_BASE_URL=<teacher 接口地址> OPENAI_API_KEY=<key> TEACHER=<模型名>
python -m agent_rollout run --tasks tasks/ --split train --runs_per_task 16 \
  --max_turns 150 --model $TEACHER --out runs/ --workers 8
python -m trajectory_tools export --runs runs/ --out exported/
python -m trajectory_tools attribute --exported exported/full --tasks tasks/test.jsonl \
  --reports runs/ --out exported/attribution_report.json
python -m trajectory_tools select --index exported/index.jsonl \
  --attribution exported/attribution_report.json --out exported/selected.jsonl
python -m trajectory_tools inject --exported exported/full --selected exported/selected.jsonl \
  --out augmented/injected --injection_model $TEACHER
python -m trajectory_tools compact --exported exported/full --selected exported/selected.jsonl \
  --out augmented/compacted --model $TEACHER --cut_threshold 16000
```

预期的中间产物（完整对照规模）：

| 环节 | 预期数量 | 判别信号 |
| --- | --- | --- |
| rollout | 约 384 条，`result` 为 success 的约八成 | `error_max_turns` 超过三成时先查 `--max_turns` 与接口限流 |
| 配对校验剔除 | 约 3% | 超过 10% 时查工具调用 id 生成与日志中断 |
| 合并后 assistant 消息数 | 比按记录逐条转换减少三到四成 | 没有减少说明 message id 未正确共享 |
| 成功轨迹（selected） | 约 215 条 | 低于 100 条时查 `pass_rate` 阈值与判分环境 |
| 注入版样本 | 约 55 条 | 质量门通过率约 45%，过低时查提示词与词数区间 |
| 压缩恢复 B 段 | 约 40 条 | 切分阈值 16K 下长轨迹约两成可切 |

评测命令（四组共用，`<ckpt>` 换成各组的 checkpoint）：

```bash
bash evaluation/serve_student.sh <ckpt> 8000
python -m evaluation run --tasks tasks/ --split eval --model <ckpt> \
  --base_url http://127.0.0.1:8000/v1 --repeats 5 --out eval_results/<组名>
```

## 实验一：初版数据基线

- **目的**：给出不做筛选、不做增强时的出发点，同时作为后续逐项增量的起点。
- **命令与配置**：

```bash
python training/build_dataset.py --baseline_raw --raw_index exported/index.jsonl \
  --tokenizer Qwen/Qwen3-8B --out data/train_baseline.jsonl
bash training/sft_qwen3_8b.sh data/train_baseline.jsonl output/baseline --save_steps 10 --eval_steps 10
```

  数据只按隐藏测试通过率不低于 0.9 取成功轨迹，不经归因筛选与盲改过滤，含答案获取类轨迹。
- **预期训练过程**：train loss 从约 0.85 降到约 0.55，val loss 降到约 0.6 后走平。train loss 不降时先查 assistant 消息是否进入 loss（`--loss_scale default`）与 tool_calls 是否被模板正确渲染；val loss 持续上升时查切分比例与样本是否集中在极少数任务上。
- **预期结果**：

| 指标 | 预期 |
| --- | --- |
| 每条轨迹盲改次数 | 约 5.5 |
| 最长盲改 | 约 8 至 10 |
| 宣告完成前未执行测试的轨迹占比 | 约 25% |
| 隐藏测试平均通过率 | 作为后续对照的基准值 |

- **偏离时的排查顺序**：① 查 `data/train_baseline.jsonl` 的条数与丢弃计数；② 查超长样本比例，超过一半被丢弃时放宽或改用头部截断；③ 查学生评测是否关闭了 thinking（`--enable_thinking` 默认关闭）；④ 查 vLLM 是否启用 hermes 工具解析。

## 实验二：默认配置（增强合计 25%）

- **目的**：验证导出校验、归因筛选、过程监督注入与压缩恢复样本的完整链路。
- **命令与配置**：

```bash
python training/build_dataset.py --selected exported/selected.jsonl \
  --injected augmented/injected --compacted augmented/compacted \
  --mix 0.25 --inject_to_compact 3:1 --tokenizer Qwen/Qwen3-8B --out data/train_mix_0.25.jsonl
bash training/sft_qwen3_8b.sh data/train_mix_0.25.jsonl output/mix-0.25 --save_steps 10 --eval_steps 10
```

- **预期训练过程**：loss 形态与基线接近，起点略高（注入文本是模型未见过的风格），train loss 约 0.9 降到约 0.55。约 25 步后 val loss 走平，继续训练时盯住行为指标，不只看 loss。第二个 epoch 后段若 val loss 回升且学生复述训练任务中的具体文件名，取较早的 checkpoint。
- **预期结果**：打包日志显示实际增强占比约 25%（约 54 条注入版加 18 条压缩段），入训约 287 条。

| 指标 | 基线 | 默认配置 | 相对变化 |
| --- | --- | --- | --- |
| 每条轨迹盲改次数 | 约 5.5 | 约 3.4 | 下降约 38% |
| 最长盲改 | 约 8 至 10 | 约 4 至 6 | 约减半 |
| 接口签名错位率 | 基准值 | 下降约五到六成 | |
| 工具报错率 | 基准值 | 约减半 | |
| 宣告完成前未执行测试的轨迹占比 | 约 25% | 约 10% | |
| 压缩后首个动作为 Read 或测试的比例 | 约 40% | 约 75% | |
| 隐藏测试平均通过率 | 基准值 | 差异在多次评测波动范围内 | |

- **偏离时的排查顺序**：① 查实际增强占比与两类样本数量；② 抽查注入版，确认回顾写在 assistant content 前且 tool_calls 未变；③ 抽查 B 段，确认摘要后第一个动作是 Read 或测试；④ 查评测压缩阈值是否为 24K，确认评测中确实触发了压缩。

## 实验三：核心消融（逐项累加）

- **目的**：分离各环节的贡献，逐项增量之和等于总提升。
- **命令与配置**：其余超参与实验二一致，只改打包参数。

```bash
# 第 2 行：只用筛选后的成功轨迹，不加增强
python training/build_dataset.py --selected exported/selected.jsonl --mix 0 \
  --tokenizer Qwen/Qwen3-8B --out data/train_selected.jsonl
# 第 3 行：加注入版（不给 --compacted，配比全部由注入版填充）
python training/build_dataset.py --selected exported/selected.jsonl --injected augmented/injected \
  --mix 0.2 --tokenizer Qwen/Qwen3-8B --out data/train_injected.jsonl
# 第 4 行：实验二的数据；第 1 行：实验一的数据
```

- **预期训练过程**：四组 loss 曲线差异很小，区分度来自评测阶段的行为指标。
- **预期结果**：

| 配置 | 每条轨迹盲改次数 | 增量 |
| --- | --- | --- |
| 初版数据：成功轨迹原样打包 | 约 5.5 | — |
| 加导出校验与归因筛选 | 约 4.8 | −0.7 |
| 加过程监督注入 | 约 4.2 | −0.6 |
| 加压缩恢复样本（增强合计 25%） | 约 3.4 | −0.8 |

  三个增量之和为 −2.1，等于总提升 5.5 − 3.4。
- **偏离时的排查顺序**：某一项增量明显偏小时，先查该步的产出数量与质量门通过率，再查评测任务数是否足以压住波动（5 次重复的标准差约 0.3 至 0.4，小于该量级的差异不作结论）。

## 实验四：技术点逐项对照

- **目的**：每个技术点单独给出可判别的信号。
- **命令与配置**：配比扫描与两个对照由脚本串行完成：

```bash
bash training/run_mix_sweep.sh exported/selected.jsonl augmented/injected augmented/compacted exported/index.jsonl
```

- **预期过程与结果**：

| 技术点 | 对照方式 | 预期 | 判别信号 |
| --- | --- | --- | --- |
| 消息聚合与配对校验 | 对比 `export` 的统计输出 | 剔除约 3%，assistant 消息数减少三到四成 | 训练样本配对错位为 0，tokenize 阶段无模板报错 |
| 失败归因 | 看 `attribution_report.json` 的两列统计 | 自测与判分不一致在低分轨迹中占绝大多数；未完成类命中少但均分最低 | 答案获取类在全量上均分高于其余轨迹，在低分子集上看不出来 |
| 过程监督注入 | 实验三第 2、3 行对比 | 宣告完成前未测占比约 25% 降到约 15% | 学生在宣告完成前列出已实现模块并执行测试 |
| 压缩恢复样本 | 实验三第 3、4 行对比 | 压缩后首动作为 Read 或测试的比例由约 40% 升到约 75% | 对未触发压缩的任务没有可测量影响 |
| 增强配比 | 四档 15% / 25% / 35% / 45% | 盲改约 3.9 / 3.4 / 3.2 / 3.3 | 35% 以上约一成评测轨迹相邻轮次在回顾与直接调用间交替 |
| 数据量对照 | `--repeat_success` | 盲改约 4.6 | 下降主要来自增强样本的内容，不来自样本数量 |

- **偏离时的排查顺序**：① 查每档打包日志的实际占比；② 35% 以上未出现风格交替时，抽查注入文本是否被质量门大量过滤；③ 配比曲线不单调时，增加评测重复次数后再判断。

## 整体排查顺序

1. 采集侧：`events.jsonl` 末条 `result` 的分布，以及 `score.json` 中 `error` 的占比。
2. 判分侧：`score.json` 中 `collection_error` 占比，高时查任务虚拟环境与 `requirements.lock`；用参考实现的 `reference_pass_rate` 确认任务本身可通过。
3. 导出侧：配对校验剔除率与失败类型。
4. 归因侧：九类模式的命中分布，`answer_access` 异常高时查工作区隔离与系统提示。
5. 训练侧：先查数据（样本数、增强占比、超长丢弃数），再查 loss，最后查行为指标。
