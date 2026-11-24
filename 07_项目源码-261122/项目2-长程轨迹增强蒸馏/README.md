# 仓库级编码 Agent 长程轨迹的质量筛选与增强蒸馏

从公开小型 Python 仓库构建「整库重建」任务，用 teacher 模型采集长程编码轨迹，
经本机判分与失败归因后，通过过程监督注入、压缩恢复样本合成与配比扫描，
把高质量轨迹蒸馏进 Qwen3-8B（全参 SFT，32K 上下文），并用行为指标评测学生。

## 目录结构

```
rebuild_spec/        任务环境规格、补丁评测与通过率汇总（含 HOST 宿主机后端）
task_builder/        从种子仓库构建本地任务（骨架提交、隐藏测试、规格文档、基准 venv）
agent_rollout/       teacher/student rollout 采集（7 工具 harness + 推理时压缩 + 自动判分）
trajectory_tools/    导出校验、失败归因、筛选、过程监督注入、压缩恢复、行为指标
training/            配比打包（ms-swift 4.6.0 OpenAI agent 格式）与 SFT 脚本
evaluation/          学生 rollout 评测与行为指标报告
tests/               不依赖 GPU/网络/Docker 的单元测试
tasks/               构建出的任务目录（tasks/test.jsonl 即数据集）
runs/                采集的轨迹（events.jsonl + score.json）
experiments/         实验说明
```

## 环境

- 训练与学生评测：4×A800 80G，无 Docker（判分走宿主机后端，每运行在临时目录建独立 venv）。
- 轨迹采集：CPU 即可；teacher 走 OpenAI 兼容 API，自备 key，用 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 与 `--model` 指定。
- 需要预装 `uv` 与 `git`；rollout 工作区与判分各自在临时目录中建虚拟环境，同一任务可以并发。
- 训练框架：ms-swift==4.6.0（`training/requirements.txt`），4 卡 DeepSpeed ZeRO-3。
- 学生服务：vLLM（`evaluation/serve_student.sh`）。
- 本机 Python >= 3.10；依赖见 `pyproject.toml`（uv 管理）。

## 运行步骤

```bash
# 1. 构建任务（8 个仓库：6 训练 / 2 评测，含 hidden_tests 与 spec.md）
python -m task_builder build --repos task_builder/seed_repos.yaml --out tasks/
python -m task_builder prepare-host-env --tasks tasks/

# 2. teacher rollout 采集（自动判分，events.jsonl 可断点查看）
python -m agent_rollout run --tasks tasks/ --split train --runs_per_task 24 \
  --max_turns 150 --model $TEACHER --out runs/ --workers 4

# 3. 导出与配对校验
python -m trajectory_tools export --runs runs/ --out exported/

# 4. 失败归因与筛选
python -m trajectory_tools attribute --exported exported/full --tasks tasks/test.jsonl \
  --reports runs/ --out exported/attribution_report.json
python -m trajectory_tools select --index exported/index.jsonl \
  --attribution exported/attribution_report.json --out exported/selected.jsonl

# 5. 增强：过程监督注入 + 压缩恢复样本
python -m trajectory_tools inject --exported exported/full --selected exported/selected.jsonl \
  --out augmented/injected --injection_model $TEACHER
python -m trajectory_tools compact --exported exported/full --selected exported/selected.jsonl \
  --out augmented/compacted --model $TEACHER --cut_threshold 16000

# 6. 配比打包与 SFT（Qwen3-8B 全参，32K）
python training/build_dataset.py --selected exported/selected.jsonl \
  --injected augmented/injected --compacted augmented/compacted \
  --mix 0.25 --inject_to_compact 3:1 --tokenizer Qwen/Qwen3-8B --out data/train.jsonl
bash training/sft_qwen3_8b.sh data/train.jsonl output/qwen3-8b-sft --save_steps 5 --eval_steps 5

# 7. 学生评测（held-out 任务，5 次重复，32K 档压缩阈值 24K）
bash evaluation/serve_student.sh <checkpoint> 8000
python -m evaluation run --tasks tasks/ --split eval --model <checkpoint> \
  --base_url http://127.0.0.1:8000/v1 --repeats 5
```

默认 8 个种子仓库用于验证全链路；要得到稳定的对照结果，按 `task_builder/seed_repos.yaml` 的字段把清单扩到约 30 个仓库，见 `experiments/README.md`。

配比四档扫描与两个对照见 `training/run_mix_sweep.sh`；
Qwen3-32B + YaRN 128K 参考配置见 `training/sft_qwen3_32b_yarn128k_reference.sh`（需 16×H100 量级）。

## 数据示例

一条导出后的轨迹（`exported/full/{id}.json`）片段：

```json
{
  "id": "tinydb__ab12cd34ef56",
  "instance_id": "tinydb",
  "messages": [
    {"g": 0, "role": "system", "content": "你在一个工作区里完成编码任务……"},
    {"g": 1, "role": "user", "content": "# 任务规格\n……"},
    {
      "g": 12,
      "role": "assistant",
      "content": "先实现 storage 模块。",
      "tool_calls": [
        {"id": "c7", "type": "function",
         "function": {"name": "write_file", "arguments": "{\"path\": \"tinydb/storages.py\", …}"}}
      ]
    },
    {"g": 13, "role": "tool", "tool_call_id": "c7", "content": "已写入 ……", "is_error": false}
  ],
  "meta": {
    "hidden_pass_rate": 0.97,
    "max_blind": 4,
    "blind_segments": 1,
    "has_compact_boundary": true,
    "injections": [{"g": 40, "type": "completion_claim"}],
    "segment": {"part": "B", "of": 2, "cut_token": 50112}
  }
}
```

训练样本为 ms-swift 4.6.0 OpenAI agent 格式：`{"tools": [...], "messages": [...]}`，
assistant 保留 `tool_calls`，工具结果为 `{"role": "tool", "tool_call_id", "content"}`。

## 技术点与代码落点

1. **stream-json 轨迹的消息聚合与配对校验**：同 message id 内容块合并、全局索引 g、
   四类配对失败剔除 —— `trajectory_tools/export.py`、`trajectory_tools/pair_validation.py`。
2. **失败断言到出错动作的三层归因**：判分层 file:line → 轨迹层最后写者 g →
   九类声明式规则 —— `trajectory_tools/attribution.py`。
3. **关键决策点的过程监督注入**：九主两备触发点、调和平均预算、纯规则质量门、
   只改 content 不动 tool_calls —— `trajectory_tools/injection.py`。
4. **九段摘要模板的压缩恢复样本合成**：轮边界切点、九段有损摘要、恢复引导、自指词质量门 ——
   `trajectory_tools/compaction.py`、`trajectory_tools/prompts/nine_section_summary.py`。
5. **增强样本配比扫描与长序列蒸馏 SFT**：四档 mix、注入:压缩 3:1、32K 截断保配对、
   SP=2 的 4 卡 ZeRO-3 —— `training/build_dataset.py`、`training/run_mix_sweep.sh`。

## 配置项

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `--compact_at` | teacher 100K / 学生 24K | 推理压缩阈值（token 估算，字符数/3） |
| `--cut_threshold` | 32K 学生档用 16K | 压缩恢复切分阈值，B 段需放得进学生上下文 |
| 规格文档上限 | 2.4 万字符 | 约 8 千 token，`task_builder/spec_document.py` 的 `MAX_SPEC_CHARS`；学生上下文更大时可调高 |
| `--mix` | 0.25 | 增强样本占比 |
| `--inject_to_compact` | 3:1 | 注入版与压缩段配比 |
| 成功阈值 | pass_rate ≥ 0.9 且 max_blind ≤ 8 | 成功轨迹判定（可配） |
| `--max_tokens` | 32768 | 打包截断上限（Qwen3 tokenizer 实计） |
| 注入上限 | 10 | 单轨迹注入点数 |

## 评测指标与预期

见 `experiments/README.md`：盲改段数、最长盲改、接口签名错位率、工具报错率、
完成宣告前未测占比、压缩后首动作比例，以及隐藏通过率按 answer_access 分组的报告。

## 测试

```bash
pytest tests/
```

只覆盖不需要 GPU、网络与 Docker 的部分（HOST 后端测试仅需 git 与 bash）。
