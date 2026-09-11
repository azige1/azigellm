# NL2Repo 从零重建任务：数据合成与蒸馏实战教程（阅读版）

> 本教程面向想复现「NL2RepoBench 风格训练数据生产线」的同学。全文不出现任何公司、组织与内部系统标识；函数名、字段名与产物文件名用通用命名，数值一律用约数与量级表述、不指向任何真实取值。照做需要：一批可浅克隆的开源仓库、一个能跑 Docker 的机器或沙箱集群、一个前沿编码 agent 的 API 额度。

---

## 目录

- [0. 教程定位](#0-教程定位)
- [1. 任务形态与判分协议](#1-任务形态与判分协议)
- [2. 全链路总览](#2-全链路总览)
- [3. Step 1 种子收割与筛选](#3-step-1-种子收割与筛选)
- [4. Step 2 SPEC 合成](#4-step-2-spec-合成)
- [5. Step 3 判分测试生成](#5-step-3-判分测试生成)
- [6. Step 4 任务箱组装与原子发布](#6-step-4-任务箱组装与原子发布)
- [7. Step 5 并行 Rollout](#7-step-5-并行-rollout)
- [8. Step 6 轨迹导出与过滤](#8-step-6-轨迹导出与过滤)
- [9. Step 7 质量审计](#9-step-7-质量审计)
- [10. Step 8 低分轨迹修复（Branch-and-Repair）](#10-step-8-低分轨迹修复branch-and-repair)
- [11. Step 9 轨迹增强（过程监督与 Compaction）](#11-step-9-轨迹增强过程监督与-compaction)
- [12. Step 10 蒸馏训练与效果评估](#12-step-10-蒸馏训练与效果评估)
- [13. 坑清单](#13-坑清单)
- [阅读内容](#阅读内容)
- [代码实战](#代码实战)

---

## 阅读内容

读教程各 Step 时对照以下材料（均为公开论文/讨论，附链接自行获取）：

| 标签 | 材料 | 链接 |
|---|---|---|
| [Step 1][Benchmark] | **NL2RepoBench**：长程仓库级生成评测，本教程任务形态的对标 | [arXiv 2512.12730](https://arxiv.org/abs/2512.12730) |
| [Step 1][Benchmark] | **ProgramBench**：给编译产物与文档从零重建程序，行为等价判分，200 任务 | [arXiv 2605.03546](https://arxiv.org/abs/2605.03546) |
| [Step 10][蒸馏] | **MindForge**：source-free 环境 + 强教师轨迹蒸馏到小模型，与本教程蒸馏段目标最一致 | [arXiv 2607.27146](https://arxiv.org/abs/2607.27146) |
| [Step 9][RL] | **CompactionRL**：用 RL 教模型管理上下文压缩，与本教程 Step 9 的 SFT 路线互补 | [arXiv 2607.05378](https://arxiv.org/abs/2607.05378) |
| [Step 7][Reward Hacking] | **METR：Recent frontier models are reward hacking**，作弊模式的公开讨论 | [METR blog](https://metr.org/blog/2025-06-05-recent-reward-hacking/) |
| [Step 7][评测漏洞] | **SWE-bench issue #465：Repo state loopholes**，评测侧漏洞的公开讨论 | [GitHub issue](https://github.com/SWE-bench/SWE-bench/issues/465) |

---

## 代码实战

`repos/` 下有三个配套练习项目，分别对应教程的一个环节，建议按顺序做：

| 标签 | 目录 | 练什么 |
|---|---|---|
| [Step 1] | `repos/[Step 1] 跨语言仓库重建与行为判分` | 从零重建类任务的评测闭环：任务目录布局、容器化运行、黑盒行为判分（同参跑两边比对输出） |
| [Step 2] | `repos/[Step 2] 函数级spec与测试合成` | 从真实仓库抽公开函数、合成规格文档与测试 harness 的造题手法（教程 SPEC 合成的函数级雏形） |
| [Step 5] | `repos/[Step 5] spec从零构建python库` | 完整的「spec → 从零重建 → 测试 → 判分 → lint」工具链，以及把 agent 接上评测的接线方式 |

每个项目都带中文 README（背景 / 安装 / 用法 / 目录结构）。作业要求见各 README 末节。

---

## 0. 教程定位

NL2RepoBench 的 reconstruct-from-scratch 形态：agent 拿到一份自然语言契约文档（SPEC）和一个空目录，要求从零实现整个 Python 库——可 `pip install -e .`、可 `import`、公开 API 行为符合规范，判分用一份 agent 不可见的测试。

本教程讲怎么**批量生产这种任务的训练数据**：造题（种子→SPEC→测试→任务箱）、采集（并行 rollout）、质检（审计与假绿过滤）、改造（修复低分轨迹、注入过程监督）、蒸馏（SFT 与效果归因）。每一步给出可直接改用的代码骨架与数据示例。

与 SWE-bench 式「修 bug」数据的差别：任务粒度是整个库，轨迹长达百轮量级，因此这条链路的独有难题是**长轨迹质量**（盲写、假完成、压缩后崩塌）与**诚实性**（模型会试图直接下载参考答案）。

---

## 1. 任务形态与判分协议

### 1.1 一条任务由四件东西构成

| 组成 | 内容 | agent 可见 |
|---|---|---|
| `start.md` | SPEC 全文：真实包名与版本、公开 API 签名、docstring 含 doctest、逐字 raise 契约、目录树 | ✅ 唯一输入 |
| 空 workspace | 只放 `start.md` | ✅ |
| `tests/test_spec.py` | 判分测试，预置进基础镜像的 `/workspace/test/` | ❌ |
| Integrity Rules | SPEC 内的反作弊硬约束段 | ✅ |

### 1.2 判分：两道 strip + 重建容器

判分**不在 agent 的工作容器里跑**。新起容器，覆盖前先剥两层：

```
agent workspace
   │  ① remove_package_files()   删 pyproject.toml / setup.py / pytest.ini / requirements*.txt
   │  ② remove_test_files()      删 agent 自己写进 test/ 的文件
   ▼
/tmp/workspace ──overlay──▶ /workspace（基础镜像里的 golden test 存活）
   ▼
create_dockerfile() → build_test_image() → run_test_commands()
   ▼
pip install -e . && pytest --continue-on-collection-errors test
   ▼
score = min(passed / test_case_count, 1)
```

- *为什么删 `pyproject.toml`*：agent 自写的打包配置会改 import 路径，可能让判分测试导到它自己的目录结构上、绕开 SPEC 要求的包布局。删掉后统一用 `PYTHONPATH=/workspace`，路径口径唯一。
- *为什么删 agent 的 `test/`*：不删的话 overlay 一覆盖，golden test 会被 agent 自己写的送分测试顶掉，判分等于让它自己给自己打分。

### 1.3 Integrity Rules（反作弊硬约束）

SPEC 固定含一段 `## Integrity Rules — No Hacking (MANDATORY)`：

- **(a)** 禁止 `pip download <target>`、禁止 `git clone` 上游仓库、禁止抓取 sdist/wheel/tarball；
- **(b)** 禁止读取、拷贝或参考 `site-packages` 里已安装的目标库副本；
- **(c)** 禁止用 WebFetch / WebSearch / curl / wget 抓上游仓库、包索引页、文档站。

段尾声明：判分会检查 workspace 里有没有抄来的源码，抄了即使测试全绿也判 FAIL。

*为什么必须写进 SPEC*：从零重建类任务的参考实现就在公网上，一条 `pip download` 就能把「重建能力」测试变成「检索能力」测试。实测某批轨迹里约四分之一的任务存在作弊，且 verbatim 拷贝一档几乎必然满分（详见 Step 7）。

> ⚠️ prompt 层声明只能约束合作的模型。真正的封堵在工具层（从 `allowedTools` 摘掉 WebFetch）与网络层（容器出站限制，参考 RepoZero 的 `--network none` 做法）。

---

## 2. 全链路总览

```
种子收割 ──► 种子筛选 ──► SPEC 合成 ──► 判分测试生成 ──► 任务箱发布
(索引+浅克隆)  (coupling)   (LLM ①)      (LLM ②)          (staging+mv)
                                                            │
                                          ┌─────────────────┘
                                          ▼
                              并行 Rollout（沙箱集群，LLM ③）
                                          │
                                          ▼
                       导出（聚合 → 格式对齐 → prefilter）
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
        质量审计（Step 7）        低分轨迹修复（Step 8）        轨迹增强（Step 9）
        假绿过滤 / 作弊审计        Branch-and-Repair            过程监督 + Compaction
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                              SFT 蒸馏 → 评测 → 分账归因（Step 10）
```

| 环节 | 调 LLM | 产物 |
|---|:---:|---|
| 种子收割/筛选 | ❌ | `seed_rank_all.tsv` |
| SPEC 合成 | ✅ | `start_detail.md` + meta |
| 判分测试生成 | ✅ | `tests/test_spec.py` |
| 任务箱转换 | ❌ | 标准任务目录 |
| Rollout | ✅ | `results.json` + `dialog.jsonl` |
| 导出/prefilter | ❌ | `sft_train.jsonl` |
| 审计/修复/增强 | 半 | 质检报告 + 增强轨迹 |

**一条贯穿全程的原则：确定性优先。** 能不调 LLM 就不调（筛选、转换、对齐、过滤全是确定性代码），LLM 只用在三处——写 SPEC、写测试、答题。

---

## 3. Step 1 种子收割与筛选

### 3.1 收割：预建索引 + 游标分页 + 浅克隆

先建一份种子索引（离线 dump 或批量调 Search API 均可），四列 TSV：

```
repo                        star    instance_id    source_dump
psf/requests                52k     psf__requests  gharchive_20xx
encode/httpx                 6.9k    encode__httpx  gharchive_20xx
...
```

用数字游标按块（每块一千多行）分页喂料，每轮浅克隆（`git clone --depth 1`，多 worker 并行）一批，只留 LOC 落在带内（数千到数万行）的库。

- *为什么用预建索引而不是现查 API*：Search API 单查询最多返回 1,000 条，普查上万个库要数万次查询。本地索引可按任意维度重筛，且支持断点续喂——生产线挂了重启，从游标位置接着走。
- *为什么设 LOC 上下界*：几百行的库一次写完，构不成 long-horizon 任务；超过上界的库，模型光探索就把轮数预算耗光，轨迹全是「还在读代码」。

### 3.2 筛选的核心指标：coupling

**coupling = 库内部 `from <pkg>.x import ...` / `import <pkg>` 形式的自引用边数。** 这是整条链路最重要的一个自定义指标：

```python
import ast, os

def compute_coupling(pkg_dir: str, pkg_name: str) -> int:
    """统计库内模块间的自引用次数（import 图的内部边数）。"""
    edges = 0
    for root, _, files in os.walk(pkg_dir):
        if any(seg in root for seg in ("tests", "test", "__pycache__")):
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                tree = ast.parse(open(os.path.join(root, f), encoding="utf-8").read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    edges += sum(1 for a in node.names if a.name.startswith(pkg_name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith(pkg_name):
                        edges += 1
    return edges
```

- *为什么它是最核心的选种判据*：一个由互不相关的工具函数堆成的库，即使 LOC 达标，模型也能逐函数独立写完——任务退化成函数级 code generation。高 coupling 逼模型先定模块边界与导出面、再让各模块彼此对齐，这才是 repo-level 的真难度，也直接把交互量顶到长程量级（实测中 coupling 上百的库能产生几十次工具调用的轨迹）。
- 门槛取「几十」量级（早期版本曾高到一百，为扩候选池下调，难度损失靠判分测试强度补回）。

### 3.3 heavy_tier：装不动的库要降级

检测重依赖（torch / tensorflow / jax 一类）是否在 **import 时**就被真正触达——静态 BFS 走顶层 import 链展开：

```python
def classify_dep_tier(pkg_dir: str, pkg_name: str, heavy=("torch", "tensorflow", "jax")) -> str:
    """重依赖在 import-time 可达 → 'AUTHOR'（装不动，被迫手写）；
       仅惰性加载（在某个子模块函数里 import）→ 'COPY'。"""
    reachable = set()
    frontier = [os.path.join(pkg_dir, "__init__.py")]
    while frontier:
        path = frontier.pop()
        ...  # 解析该文件顶层 import，把包内模块入队，包外模块记入 reachable
    return "AUTHOR" if any(h in reachable for h in heavy) else "COPY"
```

- *为什么*：重依赖在 `import` 阶段就必须存在时，判分测试极易在 collection 期直接崩，得 0 分——而这个 0 分与模型实现质量无关，只是环境装不上。典型误判：某库声明依赖 tensorflow，但只在 `backends/tensorflow.py` 里惰性加载，`import pkg` 并不触发，应判 COPY。

**产出**：`seed_rank_all.tsv`（`repo, pkg, coupling, errstr, doctest, loc, nmod, version, deps, env, tests, tier` 十二列），按 coupling 降序取候选。

---

## 4. Step 2 SPEC 合成

### 4.1 输入构造：AST 静态抽取，不喂源码

SPEC 合成**不把种子库源码喂给 LLM**，而是用 AST 抽三类信息：

```python
def collect_public_api(pkg_dir: str):
    """遍历包内 .py，抽公开 API 签名、完整 docstring（含 doctest）、raise 契约。"""
    items = []
    for path in iter_pkg_files(pkg_dir):            # 跳过 tests/__pycache__，支持 src-layout
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in tree.body:                      # 只收顶层定义
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                items.append({
                    "name": node.name,
                    "signature": ast.unparse(node.args),
                    "doc": ast.get_docstring(node),          # 不截断，doctest 完整保留
                    "raises": collect_raise_contracts(node),  # 见下
                })
    return items

def collect_raise_contracts(func, cap=8):
    """抽形如 raise XError("字面量") 的语句；类型名须含 Error/Exception/Warning；
       消息取第一个参数，f-string 表达式段占位为 {...}；按 (类型, 消息) 去重，每函数封顶 cap 条。"""
    out = []
    for node in ast.walk(func):
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            exc_type = ast.unparse(node.exc.func)
            if not any(k in exc_type for k in ("Error", "Exception", "Warning")):
                continue                                   # StopIteration 这类抓不到，属预期
            msg = literal_or_placeholder(node.exc.args[0]) if node.exc.args else ""
            if msg.strip("{}. ") and (exc_type, msg) not in out:
                out.append((exc_type, msg))
    return out[:cap]
```

- *为什么只抽这三类*：这一步决定 SPEC 的信息边界。模型看不到任何实现体，只看到 interface 与 contract——必须自己想出实现。同时逐字的 error 契约让判分测试可以断言精确文案，模型照 SPEC 实现就能过，不用猜。
- **两个实测坑**：① src-layout 的库（`repo/src/<pkg>`）要走错目录回退逻辑，否则抽出来是空的；② 渲染给 LLM 的 API 文档要做 `MAX_API_MD` 截断（十万字符量级），大库后半段 API 会被砍掉——这是后面覆盖率回补的根因（见 4.4）。

### 4.2 SPEC 的八段固定结构

```markdown
## Introduction and Goals of the {package} Project
## Integrity Rules — No Hacking (MANDATORY)      ← 生成时即写入，勿事后补
## Natural Language Instruction (Prompt)          ← 编号功能清单 + Core File Requirements
## Environment Configuration                       ← Python 版本 + 核心依赖版本（钉死）
## {package} Project Architecture                  ← 真实目录树
## API Usage Guide                                 ← 逐公开类/函数：签名 + 完整 docstring 含 doctest
## Detailed Implementation Nodes of Functions      ← Node K：描述 + 可运行 I/O 示例 + 逐字 raises 契约
## Quality Gates                                   ← finish 前必须满足的自检条件
```

第 7 段密度最高，契约逐字给出：

```
Node 3 — split_fields(text: str, sep: str) -> list[str]

Input and Output Examples:
    >>> split_fields("a,,b", ",")
    ["a", "", "b"]
    >>> split_fields("", ",")
    [""]

Raises:
    ValueError("separator must not be empty")   when sep == ""
```

**一条显式取舍**：SPEC 越细，任务越接近「按图施工」而非「自主设计」。选细 SPEC 的理由是判分必须可自动化——SPEC 一模糊，判分测试就无法既覆盖契约又不引入 SPEC 外要求，模型的失败会与能力无关。代价是任务不再考察架构设计自由度。

> **反作弊段要在生成时写对，不要事后 scrub。** 早期版本用独立脚本做事后硬化（正则删「参照上游」散文 + 注入 Integrity 段），结果 DROP 正则命中了自己注入的句子（`git clone` of its upstream repository），全部任务的 Integrity 段被写坏。可推广的判据：任何「生成后再 scrub」的步骤先验证幂等性，且 scrub 规则不得与注入内容自身重叠；更稳的形态是把约束塞进生成 prompt，一次写对。

### 4.3 Quality Gates 段：自证契约

Quality Gates 强制 agent 自写 `tests/test_contracts.py`，把每条 doctest 和每条 raises 契约编成 pytest 断言（doctest 断精确返回值；错误契约用 `pytest.raises(XError)` 且 `str(exc.value) == "<exact message>"`）。只规定覆盖要求，不代写测试体。

- *为什么*：这是模型侧的自测入口，也是后面「假绿」分析的对照组——几乎每个模型都会写并跑这个文件，但相当比例的实现仍过不了不可见判分测试（见 Step 7）。

### 4.4 SPEC 覆盖率审计与回补

一次性 LLM 调用会漏符号。审计：重新克隆真实上游 → 同一套 AST 抽取列出全部公开符号 → 与 SPEC 中出现的符号取交集 → 低覆盖的重生成。

实测一批数千个任务：回补前平均覆盖率约三成，回补后接近九成，过半任务补到全覆盖。两处主要损耗：近四成重克隆失败、约一成包名解析不到上游仓库（可用包索引 API 抽 GitHub URL 补一轮）。

> **反直觉观察**：并非所有重生成都提升覆盖率。种子库 API 列表超出 LLM 上下文预算被截断时，重生成采到的是不同前缀子集，覆盖率反而下降（实测有一例不升反降了几个点）。这类任务要按模块切分 API 列表分批生成，而不是重跑同样的调用。

---

## 5. Step 3 判分测试生成

判分测试由 LLM **照 SPEC 现写**，不从种子库拷。这是整条链路的**质量命门**。

### 5.1 两条硬约束

**① GROUNDED——只断言 SPEC 明文陈述过的内容。** 缺席时 LLM 会写出 SPEC 里没写的行为要求，答题模型无从得知，失败与能力无关，纯噪声，还系统性压低所有模型分数。

**② DISCRIMINATING——必须能 fail 掉全 `raise NotImplementedError` 的空骨架。** 缺席时测试退化成 shape matching：

```python
# ❌ 送分测试（giveaway）：空骨架也能过，零区分度
def test_import():
    import mypkg
    assert mypkg is not None

def test_version():
    import mypkg
    assert mypkg.__version__ == "1.2.3"

def test_has_attr():
    import mypkg
    assert hasattr(mypkg, "process")
```

生成后跑一遍确定性清理，按 `def test_` 切块，命中即整块丢：

```python
GIVEAWAY_PATTERNS = [
    r"\bhasattr\s*\(", r"\bcallable\s*\(", r"inspect\.signature",
    r"__version__", r"__all__", r"__doc__", r"__license__",
]

def is_giveaway(block: str) -> bool:
    """先剥注释行再匹配（anchor 注释里写 hasattr 不误伤）。"""
    return any(re.search(p, strip_comments(block)) for p in GIVEAWAY_PATTERNS)

def clean_tests(test_src: str) -> str:
    """按 def test_ 切块，命中送分模式的整块删。"""
    return "".join(b for b in split_on_def_test(test_src) if not is_giveaway(b))
```

### 5.2 合规测试长什么样：A/B anchor

```python
# anchor: A  —— 正向路径，断言 SPEC 第 7 段的 Input/Output Examples
def test_split_keeps_empty_fields():
    from mypkg import split_fields
    assert split_fields("a,,b", ",") == ["a", "", "b"]

# anchor: B  —— 反向路径，断言逐字 raises 契约
def test_empty_separator_rejected():
    import pytest
    from mypkg import split_fields
    with pytest.raises(ValueError, match="separator must not be empty"):
        split_fields("a,b", "")
```

- *为什么打 anchor*：让覆盖率审计能**按契约类型统计**。一个任务二十条测试全是 A、一条 B 都没有，说明完全没覆盖 error path——只数条数看不出来。实测数千个文件的产出中九成上下带 anchor。
- **一处有意的非对称设计**：SPEC 侧要求错误消息逐字复现；测试侧反而**禁止**断言全文消息（SPEC 消息常含 `{...}` 占位符），只许 `pytest.raises(精确类型)` + 稳定子串。契约严苛、判分宽容。

### 5.3 形态参数

| 参数 | 参考取值 | 说明 |
|---|---|---|
| 喂给测试生成的 SPEC 预算 | 数万字符 | 超出截断 |
| 单条测试形态 | 十到几十行、几条到十几条 assert | 过短覆盖不足；过长 fail 难定位到具体契约 |
| 单任务测试条数 | 十条上下 | 覆盖两三个子系统的耦合行为 |

---

## 6. Step 4 任务箱组装与原子发布

### 6.1 任务箱结构

```
tb_tasks/<pkg>/
├── task.yaml            # 元数据 + 超时 + 判分入口声明
├── Dockerfile           # 基础镜像 + 依赖预装（构建期装好，运行期不下载）
├── docker-compose.yaml  # 标准模板
├── run-tests.sh         # 判分入口
├── start.md             # SPEC 全文，作为 agent 的唯一 instruction
└── tests/
    └── test_spec.py     # 判分测试：构建进镜像，不进 agent workspace
```

`task.yaml` 关键字段：

```yaml
# canary: <uuid4>            # 防训练语料污染标记，自建语料库必备
instruction: |-              # start.md 全文，字面量块
difficulty: hard
parser_name: pytest
max_agent_timeout_sec: 5400.0   # 演示值，按需调整
max_test_timeout_sec: 2400.0    # 演示值，按需调整
run_tests_in_same_shell: false
category: rebuild-from-scratch
tags: [from-scratch-rebuild]
```

`run-tests.sh` 的核心是**整树覆盖**，把 agent 自写的测试顶掉：

```bash
#!/bin/bash
set -euo pipefail
cd /app
python -m pip install -e .
python -m pip install "pytest>=7"
rm -rf /app/tests                         # 删掉 agent 自写测试
cp -r /golden/tests /app/tests            # 整树换成镜像里的判分测试
python -m pytest -rA --continue-on-collection-errors /app/tests
```

### 6.2 原子发布：staging + mv

生产线（Producer）与采集线（Consumer）是两条独立后台循环，通过 `tb_tasks/` 目录解耦。Producer 先写 `tb_staging/<pkg>/`，双重校验（目录在 + `task.yaml` 在）后同文件系统一次 `mv` 整体移入。

- *为什么必须原子*：Consumer 随时扫 `tb_tasks/`。边写边放它会读到半成品——缺 `tests/` 的转不了、缺 `Dockerfile` 的建不了镜像，这类失败在日志里跟真坏任务混在一起，排查成本极高。同文件系统 `mv` 是原子 rename：要么看不到，要么看到完整任务。
- 配套：`done_ledger.txt` 用**行数**秒级给出「已确认可用任务数」。任务到几千个后一次全树遍历要一两分钟，ledger 行数把这步降到毫秒级。

---

## 7. Step 5 并行 Rollout

### 7.1 执行形态

每个任务发一个独立容器沙箱（DinD），容器内起编码 agent（CLI 形态），以 `start.md` **全文**作 instruction，在空 workspace 重建整个库。产物两份：

- `results.json`：`{"results": [{"is_resolved": bool, "parser_results": {"test_x": "passed"/"failed", ...}}]}`——逐测试状态字典，`tests_total = len(parser_results)`；
- `dialog.jsonl`：完整对话，stream-json 多行，**最后一行最完整**。

agent 启动形态参考：

```bash
agent-cli --verbose --output-format stream-json \
  -p "$(cat start.md)" \
  --allowedTools Bash Edit Write Read Glob Grep LS TodoRead TodoWrite
```

> ⚠️ 白名单里若保留 `WebFetch`，Integrity Rules 第 (c) 条就只剩 prompt 层约束。实测作弊相当比例走 WebFetch 完成。要封死：摘掉它，或容器网络层 `--network none` / 出站白名单。

### 7.2 并发与预算

| 参数 | 参考取值 | 说明 |
|---|---|---|
| `parallel` 主 rollout | 几十路 | 受沙箱容量约束 |
| `max_episodes` 主 rollout | 几十轮 | 超过判未完成 |
| rerun 的 `parallel` / `max_episodes` | 并发减半 / 预算翻倍 | rerun 给复杂任务更多预算 |
| 垃圾重投轮数 | 一两轮 | 见 7.3 |

**两个工程要点**：

- **工作区隔离**：每个 run 独立父目录。高并行下发生过 sibling 工作区串扰（`cp` 到隔壁遗留目录），导致某任务只跑了正常轮次的零头就「完成」。
- **内存闸**：`free` 可用内存低于阈值时阻塞新 run（单条长轨迹峰值在 GB 量级，高并发必加）。

### 7.3 垃圾判定与重试

```python
def needs_requeue(result) -> bool:
    """无轨迹，或（未 resolved 且 tests_total <= 1）→ 判装机失败，重投。"""
    if not result.has_trajectory:
        return True
    return result.tests_total <= 1 and not result.resolved
```

- *为什么用 `tests_total <= 1` 而不用得分*：得 0 分有两种完全不同的成因——**装机失败**（镜像没建起来，测试一条没跑）与**真答错**（测试跑了几十条全 fail）。前者该重投，后者是有效负样本必须保留。只看得分会把两者混为一谈。
- 另有一道 size 过滤：`dialog.jsonl` 只有几 KB 的判空跑丢弃。

---

## 8. Step 6 轨迹导出与过滤

### 8.1 聚合：stream-json → OpenAI messages

stream-json 里同一轮 assistant 的多个 block 共享 `message.id`，按它聚合：

```python
def aggregate(stream_lines):
    """按 message.id 聚合 assistant blocks，只保留 tool_result（清洗冗余系统行），
       只导有 result 行的完成轨迹。"""
    msgs, buf = [], {}
    for ev in stream_lines:
        if ev["type"] == "assistant_block":
            buf.setdefault(ev["message_id"], []).append(ev)
        elif ev["type"] == "tool_result":
            if buf:
                msgs.append(flush_assistant(buf.popitem()[1]))   # 合成一条 assistant + tool_calls
            msgs.append({"role": "tool",
                         "tool_call_id": ev["tool_use_id"], "content": ev["content"]})
    return {"messages": msgs, "tools": [...], "id": ..., "source": "..."}
```

**不可破坏的不变量**：`assistant.tool_calls[].id` 必须与后续 `tool.tool_call_id` 一一配对。下游任何切分与回填（Step 9/10）都以这个配对为前提，配对一破，切出来的段全是孤儿 tool 消息，训练直接报错。

导出格式：`assistant` 计 loss，`user` / `tool` 不计；剥掉推理段字段；`arguments` 一律 JSON 字符串。

### 8.2 prefilter 两条判据

| 判据 | 说明 | 一批 3k 余条上的实测 |
|---|---|---|
| `CHEATING` | `dialog.jsonl` 里出现**实际执行**的目标包获取调用（`pip download <pkg>`、抓上游 blob 等） | 几十条 |
| `LOW_SCORE` | `average_score` 低于阈值 | 几十条（其中大半整 0，多数 collection 期崩） |

共丢一百余条（占几个百分点），与保留集零重叠。

### 8.3 规模漏斗（一批实测）

| 环节 | 条数 |
|---|---:|
| SPEC 任务目录 | 3k 余 |
| 可转换任务箱（缺 `tests/` 的进 pending） | 3k 余 |
| 垃圾隔离 | 两百上下 |
| 每任务取最优合并 | 2k 弱 |
| rerun ×2 扩量 | 3k 余 |
| prefilter 后进入 SFT | **3k 出头** |

---

## 9. Step 7 质量审计

导出前必须回答三个问题：模型为什么失败、失败是不是模型的问题、这批数据有没有污染。

### 9.1 三层证据链

| 层 | 数据源 | 回答什么 |
|---|---|---|
| 判分层 | `post-test.txt`（pytest 完整输出：断言、报错、file:line） | 哪几条测试 fail、fail 在哪行 |
| 轨迹层 | `dialog.jsonl`（`g{N}` 全局消息索引） | 哪一步动作导致该 fail |
| 归因层 | 前两层对齐 → Bad Pattern 判定 + 严重度 | 属于哪类可命名失败模式 |

### 9.2 九类 Bad Pattern

| Bad Pattern | 定义 |
|---|---|
| Early Termination | Verification Hallucination 致过早收尾（以为测试过了） |
| Non-Finish | 从未调用 finish，等待输入或超时 |
| Context Truncation | 对话截断，已发现的深层 bug 状态丢失 |
| Navigation Trap | ls/cd 与 read 间高转移、缺 edit，浏览中空转 |
| Blind Editing | 连续多次 Write/Edit 中间无任何验证 |
| Dependency/Package Failure | 漏 `__init__.py`、包结构错、import 崩 |
| Test Suite Alignment Failure | 自测与判分测试的验证模式根本不同——**头号病灶，实测低分任务几乎全命中** |
| Resource Exhaustion | OOM / 内存不足 |
| Cheating | 获取目标包源码或判分测试，或读 `/workspace/test/` |

**频率与危害分两列看**：TSA 与 Blind Editing 几乎无处不在，但触发者均分不是最低；Non-Finish 只出现个位数次，触发者均分不足一成。零分任务几乎全命中两种组合之一：`TSA + Dependency`（collection 期崩）或 `TSA + Early Termination`（自测全绿但契约错位）。

### 9.3 Cheating 审计：信号只认执行

整套审计里最容易做错的一点——**只在实际被执行的 tool call 参数里判定**：

```python
# ❌ 全文本匹配：README 安装指引、SPEC 里作为反例写出的禁止命令全被算成作弊
if "git clone" in json.dumps(dialog):
    mark_cheating()

# ✅ 只在 tool_calls 的 command/url 字段判定
for msg in dialog:
    for call in msg.get("tool_calls", []):
        args = json.loads(call["function"]["arguments"])
        payload = args.get("command", "") + args.get("url", "")
        if hits_target_package(payload):      # token 等价匹配，不是子串
            mark_cheating()
```

实测差异：某 checkpoint 全文本匹配报出的作弊任务，改用 tool call 字段判定后**全部消失**——报出的全是误报。

**四档手法与效力梯度**（一批一百余任务的实测，确认作弊的占约四分之一）：

| Tier | 手法 | 任务数 | 均分 |
|:---:|---|---:|---:|
| 1 | Verbatim 源码拷入（`pip download` 解 wheel → `cp -r` 整包） | 十来个 | **几乎满分** |
| 2 | 拷贝/运行上游测试作 oracle（`git clone` → `pytest tests/` 驱动实现） | 几个 | 九成上下 |
| 3 | 拉取源码作参考后重写（WebFetch / curl 单文件） | 十来个 | 七成多 |
| 4 | 拉取上游测试作参考 | 个别 | 接近九成 |

同批非作弊任务均分不到五成。**作弊效力随拷贝直接程度单调递减**——先查 Tier 1，既最好查又危害最大。明确排除项：README/CHANGELOG/`pyproject.toml` 等构建元数据、依赖库源码、数据文件、包索引描述页。分布规律：作弊集中在单文件/纯 Python 库（Utility 类过半），ML 类为零——重型框架源码庞大且测试 import 即崩，拿到源码也过不了判分，作弊无收益。**反作弊强度应按任务类型分配。**

> **一条 selection bias 警告**：按「得分低」筛选深度分析子集，再用该子集统计「什么导致高分」，结论会反转——某次几十任务的深度集得出「作弊不转化为高分」（子集内作弊任务均分仅一成多），全量统计则是作弊任务均分八成多、非作弊不到五成。成因统计一律用全量。

### 9.4 假绿（False Green）：held-out 后验评分

模型自测全绿 ≠ 满足判分契约。导出阶段加一步：

```python
for traj in completed_trajectories:
    repo = replay_workspace(traj)                # 重放 Write/Edit 重建 repo
    res  = run_hidden_tests(repo, task.tests)    # pip install -e . && pytest
    traj["hidden_pass_rate"] = res.passed / res.total
    traj["hidden_status"]    = classify(res)
    # classify → pass | near | coll_err | beh_fail | install_fail
```

实测三十来个可运行任务：**约四成通不过 held-out**（全过的不到三分之二、个别只缺一两条契约、近三成 collection 期崩、个别行为失败），而同批轨迹几乎全都写了并跑了自测。最大的一类是 collection 期崩：判分测试连第一条断言都没跑到就 import 崩。

> 任何「模型自己写测试、自己判断完成」的链路都有同一问题：**判定完成的信号来自模型自身，没有外部判据校正，误差不可观测。** 加一道模型不可见的 held-out 后验评分，是这类链路最高杠杆的单点改动。

### 9.5 静态预筛信号：max_blind

held-out 评分要跑容器，成本高。纯静态特征里区分度最强的是 `max_blind`：

```python
def max_blind_edits(messages) -> int:
    """连续 Write/Edit 而中间无任何验证调用的最大次数。"""
    cur = best = 0
    for msg in messages:
        for call in msg.get("tool_calls", []):
            name = call["function"]["name"]
            if name in ("Write", "Edit", "str_replace_editor"):
                cur += 1
                best = max(best, cur)
            elif name in ("Bash", "execute_bash"):
                cur = 0
    return best
```

实测：通过轨迹的中位 `max_blind` 在十几，collection 崩的在三十上下（两倍多）；极端个例接近九十（两百多条消息里六十多次 Write、三十多次 Edit）。因果链：写很多文件不验证 → 结构/import 崩 → 自测测不到 → held-out collection 期崩。**同一指标三处复用**：SFT 静态质量门、rollout 运行时节流（超阈值插「先跑 import 冒烟」reminder）、失败预测。

### 9.6 SFT 质量门评分卡

| 维度 | 阈值 | 权重 |
|---|---|---:|
| `hidden_pass_rate` | 接近全过 KEEP / 过半 REPAIR / 不足一半 RE-ROLLOUT | 约一半 |
| held-out collection 期 | 零 error | 约两成 |
| `max_blind` | 不超过二十上下 | 约一成 |
| `finished` | True | 约一成 |
| cheating | 零信号（**一票否决**） | 约一成 |

综合分八成以上进 SFT；五到八成进修复队列；不足五成重跑或丢弃。

---

## 10. Step 8 低分轨迹修复（Branch-and-Repair）

低分与作弊轨迹数量可观、任务上下文完备：直接丢弃浪费，直接进 SFT 会把作弊与盲写教给模型。这条 pipeline 把它们改造成诚实、可验证的高质量轨迹。

### 10.1 三条核心洞察（方案成立的前提）

1. **头号病灶是契约错位，而契约可从 instruction 推导**——很多任务的 SPEC 明文写了契约，判分测试恰好断言它。所以「诚实地通过不可见判分测试」可行，无需窃取测试。
2. **失败点已结构化**——`post-test.txt` 给断言/报错/file:line，Bad Pattern 分析给根因 + 轨迹位置 `g{N}`，可外科手术式定位，不必盲目重做。
3. **高分轨迹里混有作弊**——约四分之一的任务作弊，verbatim 一档几乎满分。这些是污染源，第一步先甄别替换。

总推论：**瓶颈不在 LLM 算力，而在容器化验证的往返次数。** 设计重心 = 用充足的 LLM 调用换候选多样性与并行，用最少的容器往返做筛选。

### 10.2 两种模式

| | **Mode B（Re-solve，分叉续写）** | **Mode R（Repair，终态修复）** |
|---|---|---|
| 起点 | 原轨迹首个 bad pattern 发生点 `g{N}`（回放重建） | 最终失败 workspace（已落盘，零回放） |
| 产物 | 全长高质量构建轨迹 + 天然偏好对（坏分叉 vs 好分叉） | 短外科轨迹（几十条消息）：跑自测→见失败→诊断→补→再跑→过 |
| 主供 | SFT | 偏好对 + 「验证技能」SFT |
| 成本 | 高 | 低（资源紧张时优先） |

Mode B 回放的常见故障是 Edit 的 `old_string` 找不到（更早的 Edit 没被正确回放）：

```python
blocked = 0
for edit in edits:
    if edit.old_string in current_files[edit.path]:
        current_files[edit.path] = apply(edit)
    else:
        blocked += 1
if blocked / len(edits) > MAX_BLOCKED_RATIO:     # 参考取「接近一半」
    downgrade_to_mode_r()
```

### 10.3 诚实边界（整条 pipeline 的技术分界）

> **修复 agent 可以看到判分测试；被产出的轨迹不能看到判分测试。**

修复 agent 读判分测试是为了知道该修什么；但它写出的轨迹里：动作序列不得包含任何读取 `/workspace/test/` 的步骤，reasoning 不得引用判分测试断言，验证依据只能是从 instruction 推导的**契约自测**。这条边界由**确定性**质量门复核（正则扫 `tool_calls` + 容器审计文件访问），不能只靠生成时的 prompt 约束。

### 10.4 契约自测：诚实性机制的设计核心

判分测试不可见，但其**验证模式**能从 instruction 推断。按任务类型给模板：

| 任务类型 | 判分常见验证模式 | 契约自测（从 instruction 推导） |
|---|---|---|
| CLI 库 | `output=stream` 流捕获、usage/error 文案 | 测 `run(output=buf)` 后 buf 含 usage/error/version |
| 校验库 | `raises(Exc)` + 消息子串 + `error.path` | 测异常类型 + 消息子串 + path 字段 |
| 工具库 | 文案精确匹配、边界输入 | 负数/NaN/空容器/大数边界 + 文案 |
| 包/SDK | `from pkg import *` 导入契约 | `__all__` 完整性、API 全可导入 |
| ML/数据 | fixture + 数据形状 | 核心 API 返回形状/类型 |

完整示例——instruction 明文要求「校验失败时抛 `ValidationError`，且异常对象带 `path` 字段指向出错位置」，据此推导的契约自测：

```python
def test_invalid_age_raises_with_path():
    """契约来源：instruction 的异常契约一节，未引用任何判分断言。"""
    import pytest, mylib
    with pytest.raises(mylib.ValidationError) as exc:
        mylib.validate({"age": "unknown"}, schema={"age": {"type": "integer"}})
    assert exc.value.path == ["age"]
    assert "expected integer" in str(exc.value)
```

轨迹演示：跑自测 → 异常对象上没有 `path` → 诊断「自定义异常未存 `path` 属性」→ 在 `ValidationError.__init__` 里补 `self.path = path` → 再跑 → 过。**全程未读判分目录，但修好的正是判分测试点。** 附加收益：一条契约自测片段同时演示「finish 前做契约对齐自检」与「先诊断再改再验证」两个可迁移习惯，一条 splice 对抗两类病灶。

### 10.5 A/B/C 三档分流（按根因，不按任务）

- **A 档 局部 Splice（性价比最高）**：失败集中（只有少数几个失败组）且根因是「缺失」非「错位」（漏文件/漏 re-export/漏异常分支）。LLM 诊断（1 次调用，产 `repair_plan{insert_after_gN, ops, verify_cmds}`）→ LLM 生成「诊断→修复→验证」消息片段（1 次）→ splice 到 finish 前 → held-out 重验，失败降级 B。
- **B 档 定向 Re-rollout**：失败分散跨模块、自测与 held-out 系统性错位（判据参考：`coll_err` + 盲写连续二十几轮 + 未 finish）。把诊断浓缩成「已知历史失败模式」注入 SPEC 之后让强模型重跑，harness 层加盲写节流，最多两轮。
- **C 档 任务重设计**：双模型 0 分且 gap=0 / 环境固定失败 / SPEC 契约歧义——从训练集移除、拆 SPEC、或提高选种 coupling 门槛后重新合成。

诊断产物示例：

```json
{"task": "mypkg", "score": 0.91,
 "fail_points": [{"test": "test_decoder_inv",
   "err": "AttributeError: 'Decoder' object has no attribute 'coef_'",
   "root_cause": "MISSING_SYMBOL",
   "fix_target": "mypkg/decoding/core.py :: Decoder.inv",
   "causal_overlap": "none", "tier": "Append"}]}
```

### 10.6 五道质量过滤

| 门 | 方法 | 拒绝条件 |
|---|---|---|
| Honesty | 正则扫 `tool_calls` + 容器审计 | 命中即弃 |
| Pass | 热容器跑判分流程 | 低于参照模型得分的九成 |
| Realistic | `max_blind`、edit/verify 比 | 盲写超过个位数轮，或 edit/verify 低于约三分之一 |
| Faithful | 声明「测试通过」前必须真跑过 pytest | 声明 pass 无 pytest 结果即弃（**杀 Verification Hallucination 的确定性检测**：回溯最近的 pytest `tool_result` 查结论） |
| Diverse | 动作序列嵌入聚类 | 相似度高于约 0.9 |

**热容器省往返**：每任务常驻一个容器（base image + 已 `pip install -e .`，workspace bind-mount），增强循环只 `cp` 改动文件 + 增量 pytest（`--lf` / 定向 node），单次从近一分钟降到几秒；仅最终验收跑完整判分流程。**验收判定必须与最终评测完全一致**——自己写一套判分，哪怕差一个参数，验收通过的轨迹到真评测也可能掉分。

---

## 11. Step 9 轨迹增强（过程监督与 Compaction）

这条 pipeline **不修对错**，只往已过质量门的成功轨迹里补两类过程信号。位置：轨迹采集完成与 SFT 打包之间。

### 11.1 要治的四类病理

| 病理 | 表现 | 手段 |
|---|---|---|
| 假完成 | 宣告完成但从没跑判分测试 | 宣告点注入 completion_check 思考 |
| 盲写不验证 | 连续 Write 一堆文件不验证 | 盲写段首/段尾注入 |
| 规划打卡化 | 建了待办清单但不据此行动 | 分解后首次开工点注入 planning |
| 压缩后崩塌 | 压缩组中位得分约为未压缩组的一半 | Compaction 合成 |

共同根因是**任务状态感知缺失**——模型知道自己做过什么，不知道自己处在任务哪个位置、还差什么。所以注入的是状态感知，不是操作指导。

### 11.2 过程监督注入：触发点 + 采样 + 回填

**触发点**（9 主 + 2 fallback，统一路由表）：

| 触发点 | 优先级 | 路由 prompt |
|---|---:|---|
| `self_declare_complete`（"all tests pass" 一类表述） | 最高，**必采** | completion_check |
| `error_recovery_decision_point`（报错→修复拐点） | 次高 | planning |
| `official_test_execution`（首次跑 pytest + 最后一次的前一轮） | 高 | completion_check |
| `task_phase_boundary` / `after_task_decomposition` | 中 | 各一 |
| `before/after bulk_write_segment`（盲写连续若干轮后首次回头跑命令） | 中 | 各一 |
| `pressure_inflection`（token 突增，fallback） | 低 | completion_check |
| `reference_point`（重读泥潭，fallback） | 低 | completion_check |

- *为什么 `self_declare_complete` 必采且不受限流*：这是整条轨迹最该被 challenge 的一句话。漏采等于放过 Verification Hallucination 的唯一入口。

**采样**（防扎堆 + 防成本爆炸）：

```python
n_cand = num_turns - 3                                           # 去头尾
base = max(1, (n_cand * MAX_SAMPLES) // (n_cand + MAX_SAMPLES))
if total_chars > 100_000: base += 2
elif total_chars > 50_000: base += 1
eff_max = clamp(base, 1, MAX_SAMPLES)                            # MAX_SAMPLES 取十上下
gap = max(2, n_cand // (eff_max + 1))                            # 均分区间，散开
# must_sample 全选；其余按 score 降序、距已选点 ≥ gap
```

**两套 prompt 的共同约束**："Write your thinking in English. Write naturally and directly."——自然口吻，不用结构化标注。逐轮打标（`阶段: 实现中`）在训练数据里像外来物，模型学到的是复述标签格式，不是形成状态感知。planning 类额外喂 `{future_actions}`（该点之后**真实**发生的动作窗口），让计划 grounded。

**回填**：从完整轨迹存档恢复（`to_label` 里是截断子集，直接回填会丢触发点之后的内容）→ 质量门（纯关键词：长度 + goal 组 + verdict 组，零 LLM 调用；**降级 = 丢该注入点，不丢轨迹**）→ 前置拼接 `messages[idx]["content"] = response + "\n\n" + original`——**只改 content，不碰 tool_calls**，保住配对不变量。

### 11.3 Compaction 合成

真实长任务会触发上下文压缩，而压缩后模型表现显著下降（记忆盲改、重读泥潭）。手法：在完整轨迹里**人为造压缩边界**，一条拆成 N+1 条独立样本。

**切点检测**——逐轮累积 token（`len(text)//3` 粗估，全链路唯一口径）达阈值（几万 token、十几万字符的量级，**刻意切得比真实压缩点密**，多造样本）后不立即切，顺延找第一个 `k`：

```
(a) k+1 是「重锚点」：该轮 tool_calls 含 Read，
    或 Bash 且 command 含 pytest / run-tests.sh / unittest
(b) 尾部剩余 >= MIN_TAIL_TURNS（默认 4）
```

- *为什么必须顺延到 Read/pytest 轮*：原轨迹里 k+1 之后的动作**天然就是「压缩后先重读/先验证」的正确反射**，B 段后续动作零盲改。随手切在某次 Write 之后，B 段第一个动作是「继续盲写」——把「压缩后凭记忆接着改」写进了训练数据，方向完全反了。

**N+1 段构成**：A 段（压缩前，`messages[:切点]`原样截断）；B 段 = `messages[:background_end_idx]`（system + 任务前缀）+ **摘要消息对**（两条 user）+ tail，恢复引导前置注入 tail 首个 assistant 的 content。切点只落轮界（`TurnSpan.end_idx`），保证每段 `tool_call_id` 配对完整——实测一批轨迹切出一百多段，全部无孤儿。

**摘要消息对**（对齐主流编码 agent 的压缩消息形态）：两条 `<system-reminder>`（最后 Read 的 input/result，超长截断）+ CONTINUED_PREFIX + **9 段有损摘要**（Primary Request and Intent / Key Technical Concepts / Files and Code Sections / Errors and fixes / Problem Solving / All user messages / Pending Tasks / Current Work / Optional Next Step）+ RESUME_INSTRUCTION（"Resume directly — do not acknowledge the summary..."）。

**恢复引导刻意不提 compaction**（禁 compact/summary/context limit/previous conversation 等词，命中即 fail）——推理时 harness 指令就是 `do not acknowledge the summary`，训练数据里注入「我刚被压缩过」这类元叙述会造成训练/推理不一致。质量门：长度 + 核验语义词（confirm/verify/check/re-read/before modifying/run the test）。

---

## 12. Step 10 蒸馏训练与效果评估

### 12.1 数据配方

- 主力：Step 6 导出的成功轨迹（几千条）+ Step 8 修复轨迹 + Step 9 增强轨迹（过程监督版 + compaction 版）。
- 偏好对：Step 8 Mode B 天然产出（坏分叉 = rejected，好分叉 = chosen），供 DPO。
- **每条样本带元数据**（任务、是否 resolved、测试通过数、来源 pipeline 版本），消融时按维度切片。

### 12.2 训练要点

- 标准 agentic SFT：assistant 计 loss，user/tool 不计；百轮量级的长轨迹注意显存与上下文窗口；
- 学生模型选二三十 B 量级的 dense 或同级 MoE 即可看到明显迁移（公开对照：MindForge 用 27B 在 ProgramBench 上 37.98% → 49.51%）；
- **评测必须与训练数据去污染**：种子库扫描自公开生态，与 NL2RepoBench 等基准的任务仓可能重叠，训练集要按仓库名黑名单过滤。

### 12.3 一次实测的归因案例（方法比数字重要）

某 30B 级学生模型，SFT 前后在同一批百余任务上评测（参照为最强前沿模型）：

| 指标 | 教师轨迹采集模型 | 学生 V1（初版数据 SFT） | 学生 V2（增强数据 SFT） | 参照模型 |
|---|---|---|---|---|
| 均分 | 约五成半 | 约四成 | 四成出头 | 七成多 |
| 满分 / 零分任务 | 近二十个 / 十几个 | 个位数 / 约二十个 | 比 V1 略好 | 四十多个 / 二十几个 |

- **V1 总分掉了十几分，看似训练失败，分账后是三重作用叠加**：退步的八成半集中在训练前的二十几个作弊任务上（它们训练前靠 verbatim 拷贝撑到均分八成多，训练后停止作弊崩到三成上下）；其余七十多个非作弊任务几乎没动。结论：SFT 消除了作弊（正面）、没教会诚实实现（负面）、引入了「过度活跃但无进展」的轨迹模式（消息数涨四成、edit 数涨七成半，而 pytest 次数持平）。
- **V2 总分只涨了一两个百分点（噪声量级），分账后发现约六成增益来自几个新增作弊任务**——增强数据里混入了 `pip download` 上游源码的有害正样本，需剔除。同时轨迹行为指标实质改善：盲改次数降了约四成、接口签名错位降了六成多、最长盲改从八轮上下压到四轮上下（与参照模型持平）、工具报错减半。

### 12.4 消融六条口径纪律

1. **模型主体写清**：多份报告涉及不同模型/checkpoint 时，报告头部写明对象；
2. **前后版本定义写清**：不同报告的 V1/V2 可能含义不同，交叉说明；
3. **单 trial、|Δ| 不足 0.05 时降档表述**为趋势性而非确证性；
4. **收益按 cheating/非 cheating 分账**——不分账会把作弊红利当训练效果；
5. **退步同样归因到具体样本组**——只报总分会把三件不同的事混成一句「训练失败」；
6. **轨迹行为指标与得分分开报**——分数没动但行为质量明显改善是真实改善，应单独报告。

---

## 13. 坑清单

1. **「生成后再 scrub」必须验证幂等性**，且 scrub 规则不得与注入内容自身重叠——某版 scrub 脚本把自己注入的反作弊段当违规散文删了，整批任务反作弊段全坏；正解是把约束写进生成 prompt 一次到位。
2. **file-level 默认值 ≠ 实跑配置**：脚本顶部的默认参数常是早期原型残留，实跑由上层 CLI 覆盖；事后引用配置会把从未跑过的参数当真。实跑参数集中一处并落盘。
3. **假 SUCCESS 横幅**：批处理脚本的成功判定必须基于产物存在性与数量核对，不能基于流程走完；日志横幅不作数，看 traceback。
4. **两条循环解耦时，先确认哪侧停了**：采集侧「0 候选」可能只是生产侧停了喂料。
5. **sibling 工作区串扰**：高并行 rollout 必须每 run 独立父目录。
6. **白名单与 Integrity Rules 要一致**：留着 WebFetch 等于反作弊只剩 prompt 层。
7. **大库 SPEC 覆盖率会越补越低**：API 列表超上下文预算被截断时，重生成采到不同前缀子集；要按模块分批生成。
8. **docstring 与实现不符要信代码**： pipeline 里出现过「注入 reasoning_content」的 docstring 对应着「前置拼 content」的实现；关键函数补断言与单测。
9. **token 粗估（len//3）对中文系统性低估**：语料以中文为主时切点会比预期稀疏，常数要重新标定。
10. **标注模型的内部控制标记要清洗**：注入文本直接进训练数据，质量门只查长度+关键词不够，加一道标记清洗。
11. **作弊扫描只认 tool call 参数字段**：全文本匹配必然大量误报（实测改用正确口径后，报出的候选全是误报）。
12. **按结果筛样本再统计成因，结论会反转**：成因统计一律用全量；必须用子集时显式写出选取条件及其排除的样本类。
