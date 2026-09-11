"""Harbor 任务格式转换的单元测试。

覆盖 task_bundle 模块：task.json 读取、instruction.md 渲染、任务目录
发现与单任务端到端转换。LLM 转换调用均 mock 掉。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from task_factory.harbor_convert.task_bundle import (
    read_task_json,
    render_instruction_md,
    convert_task,
    find_task_dirs,
)


class TestReadTaskJson:
    """task.json 读取。"""

    def test_load_valid_task_json(self, tmp_path):
        task_dir = tmp_path / "task_001"
        task_dir.mkdir()

        task_data = {
            "task_id": "001",
            "task_description": "Test task description",
            "category": "test",
        }

        task_json = task_dir / "task.json"
        task_json.write_text(json.dumps(task_data))

        result = read_task_json(task_dir)
        assert result is not None
        assert result["task_id"] == "001"
        assert result["task_description"] == "Test task description"

    def test_load_missing_task_json(self, tmp_path):
        task_dir = tmp_path / "task_002"
        task_dir.mkdir()

        result = read_task_json(task_dir)
        assert result is None

    def test_load_invalid_json(self, tmp_path):
        task_dir = tmp_path / "task_003"
        task_dir.mkdir()

        task_json = task_dir / "task.json"
        task_json.write_text("{ invalid json }")

        result = read_task_json(task_dir)
        assert result is None


class TestRenderInstructionMd:
    """instruction.md 渲染。"""

    def test_create_instruction_with_description(self):
        task_data = {
            "task_description": "Create a Python script that sorts numbers.",
            "task_id": "001",
        }

        result = render_instruction_md(task_data)

        assert "# Task" in result
        assert "Create a Python script that sorts numbers." in result
        assert "<!-- Task ID: 001 -->" in result

    def test_create_instruction_without_task_id(self):
        task_data = {
            "task_description": "Write a bash script.",
        }

        result = render_instruction_md(task_data)

        assert "# Task" in result
        assert "Write a bash script." in result
        assert "<!-- Task ID" not in result

    def test_create_instruction_with_fallback_description(self):
        task_data = {
            "description": "Alternative description field",
        }

        result = render_instruction_md(task_data)

        assert "Alternative description field" in result

    def test_create_instruction_no_description(self):
        task_data = {}

        result = render_instruction_md(task_data)

        assert "No description available" in result


class TestFindTaskDirs:
    """任务目录发现。"""

    def test_get_task_directories_valid(self, tmp_path):
        for i in range(3):
            task_dir = tmp_path / f"task_{i:03d}_hash"
            task_dir.mkdir()
            (task_dir / "task.json").write_text("{}")

        invalid_dir = tmp_path / "task_invalid"
        invalid_dir.mkdir()

        other_dir = tmp_path / "other_dir"
        other_dir.mkdir()
        (other_dir / "task.json").write_text("{}")

        result = find_task_dirs(tmp_path)

        assert len(result) == 3
        assert all("task" in d.name for d in result)

    def test_get_task_directories_empty(self, tmp_path):
        result = find_task_dirs(tmp_path)
        assert len(result) == 0

    def test_get_task_directories_nonexistent(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        result = find_task_dirs(nonexistent)
        assert len(result) == 0


class TestConvertTask:
    """单任务端到端转换。"""

    @pytest.fixture
    def sample_task_dir(self, tmp_path):
        """构造一个最小 SIF 任务目录。"""
        task_dir = tmp_path / "source" / "task_001_abcd"
        task_dir.mkdir(parents=True)

        task_data = {
            "task_id": "001",
            "task_description": "Test task for unit testing",
        }
        (task_dir / "task.json").write_text(json.dumps(task_data))

        test_code = "def test_example():\n    assert True\n"
        (task_dir / "test_final_state.py").write_text(test_code)

        container_def = "Bootstrap: docker\nFrom: ubuntu:22.04\n"
        (task_dir / "container.def").write_text(container_def)

        return task_dir

    @pytest.fixture
    def output_dir(self, tmp_path):
        output = tmp_path / "output"
        output.mkdir()
        return output

    def test_convert_task_success(self, sample_task_dir, output_dir):
        with mock.patch("task_factory.harbor_convert.task_bundle.def_to_dockerfile") as mock_convert:
            mock_convert.return_value = "FROM ubuntu:22.04\nRUN echo test"

            result = convert_task(
                sample_task_dir,
                output_dir,
                reuse_dockerfile=False,
            )

            assert result["success"] is True
            assert result["error"] is None

            harbor_dir = output_dir / sample_task_dir.name
            assert harbor_dir.exists()

            assert (harbor_dir / "instruction.md").exists()

            assert (harbor_dir / "environment" / "Dockerfile").exists()

            assert (harbor_dir / "tests" / "test_final_state.py").exists()
            assert (harbor_dir / "tests" / "test.sh").exists()

            test_sh = harbor_dir / "tests" / "test.sh"
            assert test_sh.stat().st_mode & 0o111

    def test_convert_task_missing_files(self, tmp_path, output_dir):
        task_dir = tmp_path / "incomplete_task"
        task_dir.mkdir()

        task_data = {"task_description": "Incomplete task"}
        (task_dir / "task.json").write_text(json.dumps(task_data))

        result = convert_task(task_dir, output_dir)

        assert result["success"] is False
        assert "Missing required files" in result["error"]

    def test_convert_task_reuse_dockerfile(self, sample_task_dir, output_dir):
        harbor_dir = output_dir / sample_task_dir.name
        harbor_dir.mkdir()
        (harbor_dir / "environment").mkdir()
        existing_dockerfile = harbor_dir / "environment" / "Dockerfile"
        existing_dockerfile.write_text("FROM ubuntu:22.04\nRUN echo existing")

        with mock.patch("task_factory.harbor_convert.task_bundle.def_to_dockerfile") as mock_convert:
            result = convert_task(
                sample_task_dir,
                output_dir,
                reuse_dockerfile=True,
            )

            # 已有 Dockerfile 时不应再调 LLM
            mock_convert.assert_not_called()
            assert result["success"] is True


class TestBatchConversion:
    """批量转换流程。"""

    def test_batch_conversion(self, tmp_path):
        source_dir = tmp_path / "tasks"
        source_dir.mkdir()

        for i in range(5):
            task_dir = source_dir / f"task_{i:03d}_hash"
            task_dir.mkdir()

            task_data = {"task_description": f"Task {i}"}
            (task_dir / "task.json").write_text(json.dumps(task_data))
            (task_dir / "test_final_state.py").write_text("def test(): pass")
            (task_dir / "container.def").write_text("Bootstrap: docker\nFrom: ubuntu:22.04")

        tasks = find_task_dirs(source_dir)
        assert len(tasks) == 5

        output_dir = tmp_path / "harbor"
        output_dir.mkdir()

        with mock.patch("task_factory.harbor_convert.task_bundle.def_to_dockerfile") as mock_convert:
            mock_convert.return_value = "FROM ubuntu:22.04"

            results = []
            for task in tasks:
                result = convert_task(task, output_dir)
                results.append(result)

            assert all(r["success"] for r in results)
            assert len(list(output_dir.iterdir())) == 5
