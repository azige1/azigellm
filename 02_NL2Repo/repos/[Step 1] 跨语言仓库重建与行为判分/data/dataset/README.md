# 数据集目录（需自行准备）

本作业包只附带**判分用例**（`data/testcases/`），不附带被翻译的库源码。
请把数据包按下述布局解压到本目录：

```
data/dataset/
├── py2node/
│   └── sample_lib_a/        # 一个 Python 库，内含 test1.py ... testN.py
│       ├── test1.py
│       ├── test1_executable  # 可选：参考程序的预编译产物（容器模式用）
│       └── ...
└── cpp2rust/
    └── sample_lib_a/        # 一个 C++ 库，内含 tests/testN.cpp 及编译产物
        └── tests/
            ├── test1.cpp
            └── test1         # 参考二进制（可用 docker_env/compile_cpp_tests.sh 重新编译）
```

约定：
- 目录第一层即「库名」，判分用例里的文件名（如 `sample_lib_a/test1.py`）以此为前缀；
- 参考可执行文件命名：py2node 为 `<模块名>_executable`，与源文件同目录；
  cpp2rust 为去掉 `.cpp` 后缀的同名文件（或与 `tests/` 同级的 `<模块名>_executable`）；
- 补充新库时，记得在 `data/testcases/<task>/` 放置对应的用例 jsonl，
  并在 `harness/tasks/<task>.py` 的 `FORBIDDEN_HINTS` 里登记该库禁止参照的现成实现。
