"""CAP 任务的测试代码生成：初始状态校验与终态校验。

与 LLM 生成路线不同，这里是模板化拼装：根据任务元数据（需要的文件、
实体、类别）直接拼出 pytest 代码。
"""
from __future__ import annotations

from typing import Dict, Any


def cap_initial_test_code(task_data: Dict[str, Any]) -> str:
    """生成初始状态测试：校验 Node.js / CDS / SQLite 等工具链就绪。"""
    return """import subprocess
import sys
from pathlib import Path


def test_node_installed():
    \"\"\"Verify Node.js is installed.\"\"\"
    result = subprocess.run(["node", "--version"], capture_output=True, text=True)
    assert result.returncode == 0, "Node.js is not installed"
    version = result.stdout.strip()
    major_version = int(version.lstrip('v').split('.')[0])
    assert major_version >= 20, f"Node.js version must be >= 20, got {version}"


def test_npm_installed():
    \"\"\"Verify npm is installed.\"\"\"
    result = subprocess.run(["npm", "--version"], capture_output=True, text=True)
    assert result.returncode == 0, "npm is not installed"


def test_cds_installed():
    \"\"\"Verify CDS tools are installed.\"\"\"
    result = subprocess.run(["cds", "version"], capture_output=True, text=True)
    assert result.returncode == 0, "CDS tools are not installed"
    assert "@sap/cds" in result.stdout, "CDS installation seems incomplete"


def test_sqlite3_available():
    \"\"\"Verify SQLite3 is available.\"\"\"
    result = subprocess.run(["sqlite3", "--version"], capture_output=True, text=True)
    assert result.returncode == 0, "SQLite3 is not installed"


def test_working_directory():
    \"\"\"Verify we are in /home/user directory.\"\"\"
    cwd = Path.cwd()
    assert cwd == Path("/home/user"), f"Working directory should be /home/user, got {cwd}"
"""


def cap_final_test_code(task_data: Dict[str, Any]) -> str:
    """生成终态测试：按任务元数据拼装文件存在性、CDS 编译等检查。"""
    category = task_data.get("category", "data_modeling")
    required_files = task_data.get("required_files", ["db/schema.cds"])
    entities = list(dict.fromkeys(task_data.get("entities", [])))

    test_code = """import subprocess
import sys
from pathlib import Path


"""

    # 文件存在性检查
    for filepath in required_files:
        test_name = f"test_{filepath.replace('/', '_').replace('.', '_')}_exists"
        test_code += f"""def {test_name}():
    \"\"\"Verify {filepath} exists.\"\"\"
    assert Path("{filepath}").exists(), "{filepath} does not exist"


"""

    # schema.cds 相关检查
    if any("schema.cds" in f for f in required_files):
        test_code += """def test_cds_schema_compiles():
    \"\"\"Verify CDS schema compiles without errors.\"\"\"
    result = subprocess.run(
        ["cds", "compile", "db/schema.cds"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"CDS compilation failed: {result.stderr}"


"""

        if entities:
            for entity in entities:
                test_code += f"""def test_entity_{entity.lower()}_defined():
    \"\"\"Verify {entity} entity is defined in schema.\"\"\"
    schema_content = Path("db/schema.cds").read_text()
    assert "entity {entity}" in schema_content or "entity {entity.lower()}" in schema_content.lower(), \\
        f"{entity} entity not found in schema"


"""

    # 服务相关检查
    if any("service.cds" in f for f in required_files):
        test_code += """def test_service_compiles():
    \"\"\"Verify CDS service compiles without errors.\"\"\"
    result = subprocess.run(
        ["cds", "compile", "srv"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Service compilation failed: {result.stderr}"


def test_service_metadata_generation():
    \"\"\"Verify service can generate OData metadata.\"\"\"
    result = subprocess.run(
        ["cds", "compile", "srv", "--to", "edmx"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Metadata generation failed: {result.stderr}"
    assert "<edmx:Edmx" in result.stdout, "Invalid OData metadata generated"


"""

    # 数据库相关检查
    if category == "database_operations" or "sqlite.db" in required_files:
        test_code += """def test_database_exists():
    \"\"\"Verify SQLite database file exists.\"\"\"
    assert Path("sqlite.db").exists(), "Database file sqlite.db does not exist"


def test_database_schema():
    \"\"\"Verify database has tables from schema.\"\"\"
    result = subprocess.run(
        ["sqlite3", "sqlite.db", ".tables"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, "Failed to query database tables"
    # At least one table should exist
    assert len(result.stdout.strip()) > 0, "Database has no tables"


"""

    return test_code


def cap_test_suite(task_data: Dict[str, Any]) -> tuple[str, str]:
    """生成整套测试，返回 ``(初始测试代码, 终态测试代码)``。"""
    initial_test = cap_initial_test_code(task_data)
    final_test = cap_final_test_code(task_data)
    return initial_test, final_test
