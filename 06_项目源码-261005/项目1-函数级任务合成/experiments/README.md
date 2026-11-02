# 实验说明：预期过程与预期结果

本文档以「预期」口吻描述四组实验的目的、命令、观察信号与预期结果。规模按代码配置：
20 个种子仓库（`data/seed_repos.json`）、约 3,000 个候选函数、
Qwen/Qwen3-Coder-30B-A3B-Instruct 合成、Qwen/Qwen3-8B 可解性检查。
通用环境变量：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
bash scripts/serve_models.sh
# 公共前置：抓取、建环境、抽取与真实调用采集只做一次，四组实验共用
EXP=exp1
codeharvest fetch --repo_urls_file data/seed_repos.json --run_pycg_for_repos
codeharvest build --local
codeharvest harvest -e $EXP --overwrite_extracted
codeharvest capture -e $EXP
```

四组实验共用 `${EXP}_extracted.json` 与 `$EXP` 的采集记录，结果分别写在 `${EXP}_baseline`、`${EXP}_default` 等实验 ID 下。

## 1. 基线（full + literal，单轮）

目的：建立对照基线，复现「字面期望值」写法的质量上限。

```bash
EXP=exp1 bash scripts/run_ablation.sh 1
```

预期过程：约 3,000 个候选函数进入 `agent-loop`；单轮执行后大部分函数有测试结果。
正常信号：`{WORKSPACE}/execution/exp1_baseline_out.json` 中多数函数带 `run_tests_logs`。
异常信号：若 runnable 比例明显低于 80%，优先查 `build --local` 是否为各仓库装好依赖、
仓库 venv 内是否有 coverage。

预期结果：

| 指标 | 预期 |
| --- | --- |
| 一次执行通过率 | 约 52%–58% |
| 覆盖率达标率（首轮） | 约 40% |

## 2. 默认配置（sliced + equivalence + 采集 + 3 轮反馈 + 规格提炼）

目的：产出最终任务包，全流程跑通。

```bash
EXP=exp1 bash scripts/run_ablation.sh 2   # 含 refine-specs、check-solvability、export-tasks
```

预期过程：agent-loop 分 3 轮收敛；`{WORKSPACE}/reports/exp1_default_rounds.json`
逐轮记录处理数与累计达标率。refine-specs 只处理 verdict=passed 且分支覆盖率 ≥0.8 的函数。
异常信号：自等价拦截率若超过 30%，检查提示词是否混入字面断言；
可解性退回率若超过 20%，检查规格是否过短或示例不足。

预期结果：

| 指标 | 预期 |
| --- | --- |
| 一次执行通过率 | 约 70% |
| 覆盖率达标率（第 1/2/3 轮） | 约 42% / 67% / 83% |
| 可解性检查首次退回率 | 约 8%–12% |
| 任务包导出后训练 / 评测条数 | 约 2,000 / 200（按仓库切分） |

## 3. 核心消融（累加表）

目的：分离各技术点的增量贡献，逐项之和应等于总提升。

```bash
EXP=exp1 bash scripts/run_ablation.sh 3
```

| 配置 | 一次执行通过率（预期） | 增量 |
| --- | --- | --- |
| 整仓上下文 + 字面期望值 | 约 55% | — |
| + 等价性断言与自等价校验 | 约 68% | +13pp |
| + 依赖切片 3 层 | 约 68% | +0pp，上下文缩短约六成 |
| + 运行时插桩采集 | 约 70% | +2pp |

预期过程：equivalence 组相对 baseline 的提升应主要来自 `test_defect` 拦截减少
（观察 rounds 报告的 self_equivalence_intercepts 与最终 valid 比例）；
sliced 组通过率与整仓组持平、上下文 token 均值应降至约四成
（脚本末尾的 `codeharvest slice-report -e exp1` 给出各深度的 token 均值）；
capture 组提升集中在「签名信息不足、docstring 含糊」的函数上。

偏离预期时的排查顺序：① 比对两组的 `run_tests_errors` 分布是否同源；
② 检查采集文件 `{WORKSPACE}/captured_calls/{exp}/*.json` 是否覆盖过半函数；
③ 用 `--limit 100` 小规模重跑排除样本波动。

## 4. 技术点逐项对照

目的：每个技术点关闭后，对应指标应出现可判别的回退。

```bash
EXP=exp1 bash scripts/run_ablation.sh 4
```

| 关闭项 | 配置 | 预期变化 | 判别信号 |
| --- | --- | --- | --- |
| 多轮反馈 | `--max_rounds 1` | 覆盖率达标率从约 83% 回落至约 42% | rounds 报告只有第 1 轮 |
| 真实调用采集 | `--disable_captured_calls` | 一次执行通过率回落约 2pp | 提示词中无 Observed real calls 消息 |
| 依赖切片 | `--context_type full` | 通过率基本持平，上下文 token 均值约为切片档的 2.5 倍 | slice-report 对照 |

## 附：切片深度扫描

```bash
codeharvest slice-report -e exp1 --depths 1,2,3,5,full
```

预期无缺失符号占比随深度：约 70% / 88% / 96% / 97% / 97%；`full`（整仓）上下文
token 均值显著高于各切片档，作为切片收益的对照。
