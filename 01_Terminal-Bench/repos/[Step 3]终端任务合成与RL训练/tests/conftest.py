"""pytest 共享夹具。"""
import pytest


@pytest.fixture
def sample_task_data():
    """通用任务元数据样例。"""
    return {
        "task_id": "001",
        "task_description": "Sample task for unit testing",
        "category": "test",
        "complexity": "simple",
        "required_files": ["test.py"],
        "success_criteria": ["File exists", "Code runs"],
    }


@pytest.fixture
def cap_task_data():
    """CAP 任务元数据样例。"""
    return {
        "task_id": "cap_001",
        "task_description": "Create a CDS model for a bookshop",
        "category": "data_modeling",
        "complexity": "simple",
        "domain": "bookshop",
        "required_files": ["db/schema.cds"],
        "entities": ["Books", "Authors"],
        "success_criteria": [
            "db/schema.cds exists",
            "Entities are defined",
            "CDS compiles",
        ],
    }
