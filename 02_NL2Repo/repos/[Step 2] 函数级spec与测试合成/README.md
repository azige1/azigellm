# codeharvest：从真实仓库到可执行的函数级编程任务

> 作业代码包 · 函数级 spec 与测试合成（Step 2）

`codeharvest` 是一个练习用框架：输入任意 Python 仓库，自动完成
**抽取函数/方法 → 构建上下文 → 用大模型合成规格说明（docstring）与等价性测试 → 在沙箱中执行判分**
的完整流水线。最终产物是一批「带参考实现 + 自动生成测试」的函数级编程题环境，
可以用来评测代码生成模型或编程智能体。

## 背景：什么是等价性测试（equivalence test）

传统做法让大模型「预测测试的期望输出」，容易错。这里的思路不同：

- 仓库里的**原函数本身就是参考实现**（ground truth）；
- 大模型只需要写一段测试代码，断言「候选实现」与「参考实现」在相同输入下输出一致；
- 测试在沙箱里真实执行，用**分支覆盖率**过滤掉质量差的测试。

这样得到的测试 harness 自带完整的环境准备（文件、依赖、类实例），
可以直接用作编程 agent 的交互式评测环境。

## 安装

```bash
cd "[Step 2] 函数级spec与测试合成"
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

> 说明：
> - 全流程需要 Docker（用于沙箱执行）；没有 Docker 时可用 `--local` 在本机跑。
> - 沙箱执行依赖一个配套的执行服务包 `repo-exec-server`（作业环境中已预装；
>   若自行安装请联系助教获取）。
> - 合成步骤需要 OpenAI 兼容接口的 API Key（`OPENAI_API_KEY`）。
> - 本框架面向 Linux / macOS；Windows 下部分依赖（如 `resource` 模块）不可用。

## 整体流程

```bash
# 0. 选定一个实验 id（全流程复用），例如 quickstart

# 1. 抓取仓库（GitHub URL / 本地目录 / 清单文件均可）
codeharvest fetch -r https://github.com/<some-small-repo>

# 2. 抽取函数与方法（输出 {exp_id}_extracted.json）
codeharvest harvest -e quickstart --overwrite_extracted

# 3. 构建执行镜像（也可 --local 跳过 Docker）
codeharvest build -e quickstart

# 4. 生成并执行等价测试：k 轮「生成 → 执行 → 反馈改写」循环，
#    直到足够多的函数达到目标分支覆盖率（默认 k=3, min_cov=0.8, min_valid=0.8）
codeharvest agent-loop -e quickstart --save_chat

# 也可以把第 4 步拆成两步单独跑：
codeharvest synthesize -e quickstart     # 只生成测试
codeharvest run-tests -e quickstart      # 只执行测试

# 5. 查看结果
codeharvest list-targets -e quickstart --limit 10
codeharvest show -e quickstart -f <函数名> --code --test --result
codeharvest show -e quickstart --summary
```

产物目录（由 `src/codeharvest/settings.yml` 配置，默认在家目录下）：

| 目录 | 内容 |
|------|------|
| `~/buckets/local_repos/repos` | 克隆下来的待分析仓库 |
| `~/buckets/codeharvest_bucket/extracted_data` | 抽取出的函数/方法 |
| `~/buckets/codeharvest_bucket/testgen` | 生成的测试 |
| `~/buckets/codeharvest_bucket/execution` | 执行结果与覆盖率 |

## 目录结构

```
src/codeharvest/
├── cli.py                # 命令行入口（click）
├── workspace.py          # 路径与配置（读取 settings.yml）
├── parallel.py           # 多进程任务执行（带超时）
├── core/                 # 数据模型：仓库/文件/函数/方法/测试目标/生成任务
├── analysis/             # 静态分析工具箱
│   ├── ast_tools/        #   AST 解析、查找、变换、反解析
│   ├── callgraph/        #   调用图构建（PyCG）、清洗、查询
│   ├── imports/          #   import 解析与改写
│   ├── modules/          #   模块运行时检查
│   ├── instrument/       #   调用插桩：捕获真实入参/返回值
│   └── slicing/          #   依赖切片：最小可运行上下文
├── extraction/           # 仓库抓取与函数/方法抽取（含 Dockerfile 渲染）
├── synthesis/            # 合成子系统
│   ├── context/          #   上下文构建（完整仓库 / 依赖切片两种策略）
│   ├── specs/            #   规格合成：用大模型改写 docstring（规格说明）
│   └── tests/            #   测试合成与「生成-执行」多轮循环
├── execution/            # 沙箱执行：Docker 容器、执行服务、等价判定
├── evaluation/           # 结果统计：通过率、覆盖率、错误分布
├── llm/                  # 大模型接入：OpenAI / vLLM runner、磁盘缓存
└── utils/                # JSON 读写与模型查找工具
tests/                    # 单元测试（unittest / pytest 均可跑）
```

## 示例：跑通最小流程

```bash
# 抽取一个小仓库
codeharvest fetch -r https://github.com/google-research/python-graphs
codeharvest harvest -e demo --overwrite_extracted
codeharvest list-targets -e demo -d      # 看看抽到了哪些函数

# 生成 + 执行（本机模式，不需要 Docker）
codeharvest agent-loop -e demo --local --save_chat
codeharvest show -e demo --summary
```

## 作业提示

1. 先读 `core/` 下的数据模型，弄清 `FunctionTestTarget` / `TestRunHistory` 的字段；
2. 再顺着 `extraction/pipeline.py` → `synthesis/tests/generator.py` →
   `execution/equivalence.py` 把主流水线走一遍；
3. `analysis/slicing/` 是最有技术含量的部分：理解「为什么切片上下文比完整文件上下文更省 token 且更准」；
4. 尝试修改 `synthesis/tests/prompts.py` 中的提示词，观察覆盖率变化。
