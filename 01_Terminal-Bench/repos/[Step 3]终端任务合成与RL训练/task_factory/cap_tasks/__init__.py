"""CAP 任务合成子包。

面向 CAP（CDS 数据建模）领域的模板化任务生成：任务模板采样、容器
环境模板、初始/终态测试拼装。
"""

from .templates import CAP_TASK_CATEGORIES, build_cap_task
from .container import cap_container_def
from .test_builders import cap_initial_test_code, cap_final_test_code

__all__ = [
    "CAP_TASK_CATEGORIES",
    "build_cap_task",
    "cap_container_def",
    "cap_initial_test_code",
    "cap_final_test_code",
]
