# 测试套件

全部测试可离线运行：网络调用、容器运行时与训练框架（ray / skyrl）
均已打桩或 mock。

## 运行

```bash
uv sync --extra dev
pytest                     # 全部
pytest tests/test_llm_gateway.py     # 单个文件
pytest tests/test_rl_env.py::TestTerminalTaskEnvStep -v   # 单个类
```

## 结构

```
tests/
├── conftest.py               # 共享夹具
├── test_llm_gateway.py       # LLM 网关接入层 + 动作解析 + pass@k
├── test_harbor_convert.py    # SIF → Harbor 任务格式转换
├── test_cap_tasks.py         # CAP 领域模板化任务生成
├── test_rl_env.py            # RL 环境与训练配置（skyrl 全打桩）
└── test_task_audit.py        # 任务质量审计（静态 / 解题结果 / 批次异常）
```

注意：`task_audit.container_checks.check_task` 需要真实 Docker，
未纳入自动测试；其聚合逻辑（`batch_anomalies`）用构造数据覆盖。
