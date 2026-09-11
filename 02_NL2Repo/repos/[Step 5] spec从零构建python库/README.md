# rebuild-spec：按契约文档从零重建 Python 库

## 背景

这份作业的核心任务是：**给你一个真实开源 Python 库的"契约"（规格说明文档 +
整套单元测试 + 函数签名骨架），在看不到原始实现的前提下，从零把库重新实现出来**，
最终用测试通过率来衡量完成度。

本仓库提供的是配套的流程工具链 `rebuild-spec`，负责把上述任务跑起来：

1. **setup** —— 按数据集记录把目标仓库克隆到本地，并切出工作分支；
2. **build** —— 为每个仓库构建带完整依赖的 Docker 镜像（环境里只有骨架代码）；
3. **get-tests** —— 查看某个仓库要跑哪些测试用例；
4. **test** —— 把你写的实现打成补丁，送进隔离环境跑指定测试；
5. **evaluate** —— 对整个仓库分组批量评估，输出每个库的通过率与耗时；
6. **lint** —— 用 ruff + pyright 检查你产出的代码风格；
7. **save** —— 把成果分支推送到你自己的 GitHub 账号下留档。

你的实现工作发生在 `setup` 克隆出来的仓库里：直接在 `rebuild` 分支（或自建分支）
上写代码、提交，然后用 `test` / `evaluate` 验证。

## 安装

```bash
pip install -e .
```

依赖：Python ≥ 3.10；本地跑测试需要安装并启动 Docker。
云后端（modal / e2b）需要另行配置对应平台的凭证。

安装后得到 `rebuild-spec` 命令；如果 PATH 里没有它，可以用
`python -m rebuild_spec` 等价替代。

## 命令用法

典型流程（先 `cd` 到一个空的工作目录）：

```bash
# 1. 克隆仓库分组（lite 是小分组，适合先跑通；all 是完整分组）
rebuild-spec setup lite --dataset-name <your-dataset>

# 2. 构建镜像（需要 Docker）
rebuild-spec build --num-workers 8

# 3. 查看某个库的测试用例
rebuild-spec get-tests tinydb

# 4. 在 repos/<repo> 里写代码并 git commit 到你的分支，然后跑测试
rebuild-spec test tinydb "test/test_tinydb.py" --branch my-solution --backend local

# 5. 批量评估整个分组
rebuild-spec evaluate --branch my-solution --backend local

# 6. 风格检查（可选指定文件）
rebuild-spec lint tinydb

# 7. 推送到自己的 GitHub 留档（需要 GITHUB_TOKEN）
rebuild-spec save <your-github-user> my-solution
```

每个命令都支持 `-h` 查看完整参数。`setup` 成功后会在当前目录写入
`.rebuild-spec.yaml` 状态文件，后续命令从这里读取数据集与目录配置。

`--reference` 选项表示对"参考实现"提交跑测试，用于确认环境与测试本身是正常的。

## 目录结构

```
rebuild_spec/
├── __main__.py        # python -m rebuild_spec 入口
├── cli.py             # 七个子命令的定义（typer）
├── core/              # 基础设施
│   ├── constants.py   #   数据记录模型、仓库分组、日志常量
│   ├── gitops.py      #   git 克隆 / 分支 / diff 与日志工具
│   ├── specs.py       #   环境规格（安装脚本、评测脚本）与 Dockerfile 模板
│   ├── images.py      #   Docker 镜像构建
│   ├── containers.py  #   容器内文件拷贝、限时执行、清理
│   └── backends.py    #   local (Docker) / modal / e2b 三种执行后端
├── steps/             # 流水线步骤，与命令一一对应
│   ├── clone.py       #   setup
│   ├── build_images.py#   build
│   ├── list_tests.py  #   get-tests
│   ├── run_tests.py   #   test
│   ├── evaluate.py    #   evaluate
│   ├── lint.py        #   lint
│   └── publish.py     #   save
└── data/test_ids/     # 各仓库测试用例 ID 的离线数据（bz2 文本）
```

运行过程中生成的日志在 `logs/` 下，克隆的仓库在 `repos/` 下。

## 数据自备说明

工具链本身**不包含任务数据集**，需要自己准备一份 Hugging Face 数据集，
并把 `--dataset-name <your-dataset>` 换成你的名字（`setup` 之后它会被记入
`.rebuild-spec.yaml`，后续命令自动沿用）。

整库重建类数据集每条记录需要的字段：

| 字段 | 含义 |
| --- | --- |
| `instance_id` | 实例唯一标识 |
| `repo` | `owner/name` 形式的 GitHub 仓库 |
| `base_commit` | 骨架代码所在的提交（只有签名与文档，没有实现） |
| `reference_commit` | 用于安装环境的参考提交 |
| `setup` | 环境安装配置（`python` 版本、`pre_install`、`packages`、`pip_packages`、`install`） |
| `test` | 测试配置（`test_cmd`、`test_dir`） |
| `src_dir` | 源码目录 |

数据集名里也支持 swe 风格实例（字段含 `test.patch` / `test_patch`）与
函数补全类数据（`prompt` / `canonical_solution` / `test`，每条记录一道小题）。
另外约定：**数据集名中包含 `rebuild` 字样时**，命令行会按内置仓库分组
（`lite` / `all` / 单库名，见 `core/constants.py`）做合法性校验。

镜像推送与 modal 后端用到的镜像仓库前缀由环境变量 `REBUILD_SPEC_REGISTRY`
指定（默认是占位值 `your-registry`，推送/云端运行前请改成自己的）。

## 关于接入编程 agent

本工具链只负责"环境 + 评测"闭环，**不绑定任何具体的编程 agent**。
你可以用任意方式产出实现——手写、 Copilot、或自己搭的 agent——只要在克隆
出来的仓库里提交到某个分支，`test` / `evaluate` 就会基于该分支与
`base_commit` 之间的 diff 生成补丁并评测。推荐的协作方式是：让 agent 在
`repos/<repo>` 目录内工作、按文件逐个提交，然后周期性跑 `rebuild-spec test`
获取测试反馈。
