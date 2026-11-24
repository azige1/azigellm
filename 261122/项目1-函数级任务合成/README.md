# codeharvest：函数级任务合成与等价性判分

`codeharvest` 从真实 Python 仓库自动合成**函数级编程任务**：抽取函数 → 构建依赖切片上下文 →
用大模型合成与参考实现对照的等价性测试 → 在沙箱中执行判分 → 提炼规格说明 → 可解性检查 →
打包为按仓库切分的训练 / 评测数据集。判分测试对使用者隐藏，任务只暴露签名与规格。

## 运行环境

- AutoDL 4×A800 80G，Linux，**Python 3.11**（PyCG 在 3.12+ 可能不可用；不可用时调用图缺失，
  依赖切片退化为单文件切片，跨文件补全失效、无缺失符号占比下降），无 Docker。
- 主路径为 `--local` 本地模式：每个仓库用 `uv venv` 建独立虚拟环境（`codeharvest build --local`，
  环境放在 `{WORKSPACE}/repo_venvs/{仓库}`，不放进仓库目录），执行测试与采集时用仓库自己的解释器，
  仓库之间依赖互不污染。需要预装 `uv`。Docker 路径保留（`codeharvest serve` 在容器内起 rpyc 执行服务），但不是默认。
- 合成模型：`Qwen/Qwen3-Coder-30B-A3B-Instruct`（vLLM 部署，2 卡 TP，端口 8000）；
  可解性检查模型：`Qwen/Qwen3-8B`（1 卡，端口 8001）。两者都走 OpenAI 兼容接口，
  合成相关子命令读 `OPENAI_BASE_URL`，`check-solvability` 默认连 8001（`--base_url` 可改），
  规格退回重写时连 8000 上的合成模型：

```bash
pip install uv && pip install -e .   # 安装 uv 与 codeharvest
bash scripts/serve_models.sh         # 起两个 vLLM 服务
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

## 目录结构

```
src/codeharvest/
  extraction/    抓取仓库、过滤抽取（含不确定行为过滤器）、PyCG 调用图、本地 venv 隔离
  analysis/      依赖切片（语句级依赖图 + 调用图补全）、调用图构建、运行时插桩采集
  synthesis/     上下文构建、测试合成与多轮反馈、规格提炼、可解性检查、任务导出
  execution/     执行沙箱（fut_module + reference 副本）、子进程 coverage 判分
  llm/           OpenAI 兼容客户端与模型注册表
  evaluation/    生成测试质量统计
  cli.py         全部子命令入口
scripts/         一键流水线与消融实验脚本（LF 换行）
data/            种子仓库清单与公开基准黑名单
tests/           unittest 测试
experiments/     实验说明与预期结果
```

## 流程

```bash
codeharvest fetch --repo_urls_file data/seed_repos.json --run_pycg_for_repos
codeharvest build --local            # 每仓库 uv venv + 安装依赖 + coverage
codeharvest harvest -e exp1 --overwrite_extracted
codeharvest capture -e exp1          # pytest/examples 真实调用采集
codeharvest agent-loop -e exp1 --local
codeharvest refine-specs -e exp1
codeharvest check-solvability -e exp1 --local
codeharvest export-tasks -e exp1 --eval_ratio 0.1 --blocklist data/benchmark_repo_blocklist.txt
```

`bash scripts/run_pipeline.sh` 串起全部步骤。

## 等价性判分

- 测试不写字面期望值：对每个输入同时调用候选与 `reference_{name}`（AST 复制的参考副本），
  用 `compare_calls` / `assert_same_outcome` 比较返回值、异常类型与消息；浮点按 rel_tol=1e-9 递归比较。
- 每个测试在独立子进程中以 coverage 分支模式运行；行号与被测函数在原文件中的 1 起行号一致。
- 新测试先在参考实现上自等价校验（候选即参考），未通过标记 `test_defect` 拦截；
  `SyntaxWarning` 等初始化噪声计入 `env_noise`，不算失败。
- 参数无法 deepcopy 的测试标记 `test_defect`（`uncopyable_argument`）。
- 运行仓库自带测试 / examples 时采集的真实调用（入参、返回值或异常、调用者，每函数最多 3 条）
  随合成请求附上；有副作用的函数不采集。

## 任务包格式

```json
{
  "task_id": "tomlkit-dumps-1",
  "signature": "def dumps(obj, ... ) -> str: ...",
  "spec": "Serialize the given object to a TOML string. ...",
  "source_ref": {"repo_id": "sdispater___tomlkit", "file_path": "tomlkit/api.py", "commit": "..."},
  "test_source": "# hidden equivalence tests",
  "coverage": 0.87,
  "verdict": "passed",
  "solve_rate": 0.75,
  "split": "train"
}
```

## 技术点与代码落点

1. **参考实现作 oracle 的等价性判分**：`execution/local_exec_service.py`（沙箱构建、codegen 模式替换）、
   `execution/run_one_test.py`（子进程 coverage 判分）、`execution/equivalence_helpers.py`（对照比较）。
2. **语句依赖图与调用图的依赖切片**：`analysis/slicing/`（`slicer.py` 深度控制，
   `handlers/definitions.py` 跨文件补全与动态引用，`missing_symbols.py` 符号完整性检查）。
3. **运行时插桩的真实调用样本采集**：`analysis/instrument/standalone_capture.py`（自包含插件）、
   `side_effect_filter.py`（副作用静态判定）、`capture_runner.py`。
4. **执行报错与未覆盖分支的多轮反馈**：`synthesis/tests/loop.py`（按 `fut.id` 对齐、
   3 轮反馈）、`feedback.py`（结构化报错与 `# UNCOVERED BRANCH` 标注）。
5. **docstring 与已通过测试的规格提炼**：`synthesis/specs/`（`prompts.py` 3–8 句规格、
   `solvability_check.py` 采样 4 次全失败退回重写）。

## 配置项（常用）

| 配置 | 默认 | 说明 |
| --- | --- | --- |
| `--context_type` | sliced | 上下文类型：sliced / full / naive |
| `--test_style` | equivalence | 测试写法：equivalence 对照 / literal 字面期望值 |
| `--max_context_size` | 32000 | 上下文 token 上限 |
| `--temperature` / `--max_tokens` | 0.7 / 4096 | 测试合成的采样温度与输出上限（规格提炼默认 0.3，可解性检查默认 0.8、n=4） |
| `--capture_exp_id` | 同 exp_id | 读取哪个实验的真实调用采集记录 |
| `--max_rounds` / `--min_cov` / `--min_valid` | 3 / 0.8 / 0.8 | 多轮反馈终止条件 |
| `--disable_captured_calls` | 关 | 关闭真实调用采集附加（消融） |
| `--disable_nondeterministic_filter` | 关 | 关闭不确定行为函数过滤 |
| `--eval_ratio` | 0.1 | 任务包按仓库切分的评测集比例 |

## 评测指标与预期

| 指标 | 预期 |
| --- | --- |
| 测试一次执行通过率（基线：full + literal 单轮） | 约 52%–58% |
| 一次执行通过率（+ 等价性断言与自等价校验） | 约 68%（+13pp） |
| 一次执行通过率（+ 切片 / + 采集） | 约 68% / 约 70% |
| 覆盖率达标率（第 1/2/3 轮） | 约 42% / 67% / 83% |
| 切片深度 1/2/3/5/full 无缺失符号占比 | 约 70% / 88% / 96% / 97% / 97% |
| 可解性检查首次退回率（对照直接用 docstring 约 30%） | 约 8%–12% |
| 抽检缺陷率（人工抽 100 条，literal vs equivalence） | 约 25%–30% vs 2%–3% |

详细实验设计、命令与预期过程见 [experiments/README.md](experiments/README.md)。
