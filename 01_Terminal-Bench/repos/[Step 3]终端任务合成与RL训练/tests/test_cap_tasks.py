"""CAP 任务合成的单元测试。

覆盖 cap_tasks 子包：任务模板采样、初始/终态测试拼装、容器模板。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_factory.cap_tasks.templates import (
    CAP_TASK_CATEGORIES,
    build_cap_task,
    list_cap_templates,
    DATA_MODELING_TEMPLATES,
    SERVICE_DEFINITION_TEMPLATES,
    DATABASE_OPERATIONS_TEMPLATES,
)
from task_factory.cap_tasks.test_builders import (
    cap_initial_test_code,
    cap_final_test_code,
    cap_test_suite,
)
from task_factory.cap_tasks.container import (
    cap_container_def,
    cap_dockerfile,
)


class TestCapTaskCategories:
    """任务类别定义。"""

    def test_categories_defined(self):
        assert len(CAP_TASK_CATEGORIES) > 0
        assert "data_modeling" in CAP_TASK_CATEGORIES
        assert "service_definition" in CAP_TASK_CATEGORIES
        assert "database_operations" in CAP_TASK_CATEGORIES

    def test_templates_available(self):
        templates = list_cap_templates()
        assert len(templates) > 0

        categories_covered = {t.category for t in templates}
        assert "data_modeling" in categories_covered


class TestBuildCapTask:
    """任务采样。"""

    def test_generate_data_modeling_simple(self):
        task = build_cap_task("data_modeling", complexity="simple")

        assert task["category"] == "data_modeling"
        assert task["complexity"] == "simple"
        assert "task_description" in task
        assert len(task["task_description"]) > 0
        assert "required_files" in task
        assert "db/schema.cds" in task["required_files"]

    def test_generate_data_modeling_medium(self):
        task = build_cap_task("data_modeling", complexity="medium")

        assert task["category"] == "data_modeling"
        assert task["complexity"] == "medium"
        assert "entities" in task
        assert len(task["entities"]) >= 2
        assert "relationship" in task["task_description"].lower()

    def test_generate_data_modeling_complex(self):
        task = build_cap_task("data_modeling", complexity="complex")

        assert task["category"] == "data_modeling"
        assert task["complexity"] == "complex"
        assert "managed" in task["task_description"].lower() or "annotation" in task["task_description"].lower()

    def test_generate_service_definition(self):
        task = build_cap_task("service_definition", complexity="simple")

        assert task["category"] == "service_definition"
        assert "srv/catalog-service.cds" in task["required_files"]
        assert "service" in task["task_description"].lower()

    def test_generate_database_operations(self):
        task = build_cap_task("database_operations", complexity="medium")

        assert task["category"] == "database_operations"
        assert "database" in task["task_description"].lower() or "data" in task["task_description"].lower()
        assert "success_criteria" in task
        assert len(task["success_criteria"]) > 0

    def test_generate_with_specific_domain(self):
        task = build_cap_task("data_modeling", complexity="simple", domain="bookshop")

        assert task["domain"] == "bookshop"
        assert "bookshop" in task["task_description"].lower()

    def test_generate_invalid_category(self):
        with pytest.raises(ValueError, match="Unknown category"):
            build_cap_task("invalid_category")

    def test_task_has_required_fields(self):
        task = build_cap_task("data_modeling", complexity="simple")

        required_fields = [
            "task_description",
            "category",
            "complexity",
            "domain",
            "required_files",
            "success_criteria",
            "hints",
            "focus_areas",
        ]

        for field in required_fields:
            assert field in task, f"Missing required field: {field}"

    def test_task_description_not_empty(self):
        for category in ["data_modeling", "service_definition", "database_operations"]:
            for complexity in ["simple", "medium", "complex"]:
                try:
                    task = build_cap_task(category, complexity=complexity)
                    assert len(task["task_description"]) > 50, \
                        f"Task description too short for {category}/{complexity}"
                except ValueError:
                    # 该类别/复杂度组合没有模板时跳过
                    pass


class TestCapInitialTest:
    """初始状态测试拼装。"""

    def test_generate_initial_test(self):
        task_data = {
            "category": "data_modeling",
            "complexity": "simple",
        }

        test_code = cap_initial_test_code(task_data)

        assert "def test_node_installed" in test_code
        assert "def test_npm_installed" in test_code
        assert "def test_cds_installed" in test_code
        assert "def test_sqlite3_available" in test_code
        assert "subprocess.run" in test_code

    def test_initial_test_checks_node_version(self):
        test_code = cap_initial_test_code({})

        assert "major_version >= 20" in test_code
        assert "Node.js version must be" in test_code

    def test_initial_test_checks_working_directory(self):
        test_code = cap_initial_test_code({})

        assert "/home/user" in test_code
        assert "def test_working_directory" in test_code


class TestCapFinalTest:
    """终态测试拼装。"""

    def test_generate_final_test_basic(self):
        task_data = {
            "category": "data_modeling",
            "required_files": ["db/schema.cds"],
            "entities": ["Books", "Authors"],
        }

        test_code = cap_final_test_code(task_data)

        assert "def test_db_schema_cds_exists" in test_code
        assert "def test_cds_schema_compiles" in test_code
        assert "def test_entity_books_defined" in test_code
        assert "def test_entity_authors_defined" in test_code

    def test_generate_final_test_service(self):
        task_data = {
            "category": "service_definition",
            "required_files": ["srv/catalog-service.cds", "db/schema.cds"],
            "entities": [],
        }

        test_code = cap_final_test_code(task_data)

        assert "def test_service_compiles" in test_code
        assert "def test_service_metadata_generation" in test_code
        assert "edmx" in test_code.lower()

    def test_generate_final_test_database(self):
        task_data = {
            "category": "database_operations",
            "required_files": ["sqlite.db", "db/schema.cds"],
            "entities": [],
        }

        test_code = cap_final_test_code(task_data)

        assert "def test_database_exists" in test_code
        assert "def test_database_schema" in test_code
        assert "sqlite3" in test_code

    def test_final_test_no_duplicates(self):
        task_data = {
            "category": "data_modeling",
            "required_files": ["db/schema.cds", "srv/service.cds"],
            "entities": ["Books", "Books"],  # 重复实体
        }

        test_code = cap_final_test_code(task_data)

        lines = test_code.split("\n")
        test_functions = [line for line in lines if line.startswith("def test_")]

        function_names = [line.split("(")[0] for line in test_functions]
        assert len(function_names) == len(set(function_names)), \
            f"Duplicate test functions found: {function_names}"


class TestCapTestSuite:
    """整套测试生成。"""

    def test_generate_complete_suite(self):
        task_data = {
            "category": "data_modeling",
            "complexity": "medium",
            "required_files": ["db/schema.cds"],
            "entities": ["Books", "Authors"],
        }

        initial_test, final_test = cap_test_suite(task_data)

        assert len(initial_test) > 0
        assert len(final_test) > 0
        assert "def test_" in initial_test
        assert "def test_" in final_test
        assert initial_test != final_test


class TestCapContainerTemplates:
    """容器环境模板。"""

    def test_get_container_def(self):
        container_def = cap_container_def()

        assert "Bootstrap: docker" in container_def
        assert "From: ubuntu:22.04" in container_def
        assert "nodejs" in container_def.lower()
        assert "@sap/cds-dk" in container_def
        assert "useradd" in container_def

    def test_container_def_has_node_20(self):
        container_def = cap_container_def()

        assert "setup_20.x" in container_def

    def test_get_dockerfile(self):
        dockerfile = cap_dockerfile()

        assert "FROM ubuntu:22.04" in dockerfile
        assert "nodejs" in dockerfile.lower()
        assert "@sap/cds-dk" in dockerfile
        assert "WORKDIR /home/user" in dockerfile
        assert "USER user" in dockerfile

    def test_dockerfile_multi_stage_not_required(self):
        dockerfile = cap_dockerfile()

        from_count = dockerfile.count("FROM ")
        assert from_count == 1


class TestEndToEnd:
    """完整流程：采样任务 + 生成测试 + 取容器模板。"""

    def test_end_to_end_task_generation(self):
        task = build_cap_task("data_modeling", complexity="simple", domain="bookshop")

        initial_test, final_test = cap_test_suite(task)

        container_def = cap_container_def()

        assert len(task["task_description"]) > 0
        assert len(initial_test) > 0
        assert len(final_test) > 0
        assert len(container_def) > 0

        for required_file in task["required_files"]:
            # 终态测试应覆盖每个验收文件
            assert required_file.replace("/", "_").replace(".", "_") in final_test

    def test_generate_multiple_tasks_different_domains(self):
        domains = ["bookshop", "inventory", "order management"]
        tasks = []

        for domain in domains:
            task = build_cap_task("data_modeling", complexity="medium", domain=domain)
            tasks.append(task)

        descriptions = [t["task_description"] for t in tasks]
        assert len(set(descriptions)) == len(descriptions)

        for task in tasks:
            assert task["domain"] in domains
            assert len(task["required_files"]) > 0
            assert len(task["entities"]) > 0

    def test_complexity_progression(self):
        simple = build_cap_task("data_modeling", complexity="simple", domain="bookshop")
        medium = build_cap_task("data_modeling", complexity="medium", domain="bookshop")
        complex_task = build_cap_task("data_modeling", complexity="complex", domain="bookshop")

        assert len(simple["task_description"]) <= len(medium["task_description"])
        assert len(simple["success_criteria"]) <= len(complex_task["success_criteria"])

        complex_desc = complex_task["task_description"].lower()
        assert any(keyword in complex_desc for keyword in ["managed", "annotation", "many-to-many"])


class TestEdgeCases:
    """边界与异常输入。"""

    def test_empty_task_data(self):
        initial_test = cap_initial_test_code({})
        assert len(initial_test) > 0

        final_test = cap_final_test_code({})
        assert len(final_test) > 0

    def test_task_with_no_entities(self):
        task_data = {
            "category": "file_management",
            "required_files": ["package.json"],
            "entities": [],
        }

        test_code = cap_final_test_code(task_data)
        assert "def test_" in test_code

    def test_task_with_many_files(self):
        task_data = {
            "category": "file_management",
            "required_files": [
                "db/schema.cds",
                "srv/service1.cds",
                "srv/service2.cds",
                "package.json",
                "README.md",
            ],
            "entities": [],
        }

        test_code = cap_final_test_code(task_data)

        for filepath in task_data["required_files"]:
            test_name = filepath.replace("/", "_").replace(".", "_")
            assert f"def test_{test_name}_exists" in test_code
