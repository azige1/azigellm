"""CAP（CDS）任务模板库与采样逻辑。

覆盖数据建模、服务定义、数据库操作三类任务，每类按复杂度分档；
模板中的领域、实体、字段等占位符在采样时随机填充。
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import random


CAP_TASK_CATEGORIES = [
    "data_modeling",
    "service_definition",
    "database_operations",
    "handler_implementation",
    "file_management",
]


@dataclass
class CapTaskTemplate:
    """CAP 任务模板：描述模板 + 验收文件 + 通过标准。"""

    category: str
    complexity: str  # simple / medium / complex
    description_template: str
    required_files: List[str]
    success_criteria: List[str]
    hints: List[str]
    focus_areas: List[str]


# 数据建模模板
DATA_MODELING_TEMPLATES = [
    CapTaskTemplate(
        category="data_modeling",
        complexity="simple",
        description_template=(
            "Create a CDS data model for a {domain} application.\n\n"
            "Requirements:\n"
            "- Define a {entity1} entity with fields: {fields1}\n"
            "- All fields should have appropriate types (String, Integer, Decimal, etc.)\n"
            "- Add a primary key field named 'ID' of type UUID\n"
            "- Save the model in db/schema.cds"
        ),
        required_files=["db/schema.cds"],
        success_criteria=[
            "db/schema.cds exists",
            "Entity is properly defined with all required fields",
            "CDS file compiles without errors",
        ],
        hints=[
            "Use 'cds init' to create project structure",
            "CDS entities use syntax: entity Name { ... }",
            "UUID type is available in CDS",
        ],
        focus_areas=["entity definition", "field types", "primary keys"],
    ),
    CapTaskTemplate(
        category="data_modeling",
        complexity="medium",
        description_template=(
            "Create a CDS data model for a {domain} application with relationships.\n\n"
            "Requirements:\n"
            "- Define {entity1} entity with fields: {fields1}\n"
            "- Define {entity2} entity with fields: {fields2}\n"
            "- Create a {relationship_type} relationship from {entity1} to {entity2}\n"
            "- Use appropriate association/composition keywords\n"
            "- Save the model in db/schema.cds"
        ),
        required_files=["db/schema.cds"],
        success_criteria=[
            "db/schema.cds exists",
            "Both entities are properly defined",
            "Relationship is correctly established",
            "CDS file compiles without errors",
        ],
        hints=[
            "Use 'Association' for loose relationships",
            "Use 'Composition' for tight parent-child relationships",
            "Relationships use syntax: fieldName : Association to Entity",
        ],
        focus_areas=["entity relationships", "associations", "compositions"],
    ),
    CapTaskTemplate(
        category="data_modeling",
        complexity="complex",
        description_template=(
            "Create a comprehensive CDS data model for a {domain} application.\n\n"
            "Requirements:\n"
            "- Define {entity1} entity with {num_fields1} fields\n"
            "- Define {entity2} entity with {num_fields2} fields\n"
            "- Define {entity3} entity for many-to-many relationship\n"
            "- Use managed aspects (cuid, managed) for automatic fields\n"
            "- Add @assert.range annotations for numeric fields\n"
            "- Save the model in db/schema.cds"
        ),
        required_files=["db/schema.cds"],
        success_criteria=[
            "db/schema.cds exists",
            "All entities are properly defined",
            "Many-to-many relationship is correctly modeled",
            "Managed aspects are used",
            "Annotations are applied",
            "CDS file compiles without errors",
        ],
        hints=[
            "Managed aspects: `entity Name : cuid, managed { ... }`",
            "Many-to-many needs a junction entity",
            "@assert.range: [min, max] validates numeric fields",
        ],
        focus_areas=["complex relationships", "managed aspects", "annotations"],
    ),
]


# 服务定义模板
SERVICE_DEFINITION_TEMPLATES = [
    CapTaskTemplate(
        category="service_definition",
        complexity="simple",
        description_template=(
            "Create a CDS service that exposes entities for a {domain} application.\n\n"
            "Requirements:\n"
            "- Create a service named '{service_name}'\n"
            "- Expose the {entity1} entity with full CRUD access\n"
            "- Save the service definition in srv/catalog-service.cds"
        ),
        required_files=["srv/catalog-service.cds", "db/schema.cds"],
        success_criteria=[
            "srv/catalog-service.cds exists",
            "Service is properly defined",
            "Entity is exposed in the service",
            "Service compiles without errors",
        ],
        hints=[
            "Service syntax: service ServiceName { ... }",
            "Expose entities: entity EntityName as projection on db.EntityName;",
        ],
        focus_areas=["service definition", "entity exposure"],
    ),
]


# 数据库操作模板
DATABASE_OPERATIONS_TEMPLATES = [
    CapTaskTemplate(
        category="database_operations",
        complexity="medium",
        description_template=(
            "Initialize database and insert seed data for a {domain} application.\n\n"
            "Requirements:\n"
            "- Deploy the CDS model to SQLite database\n"
            "- Insert {num_records} {entity1} records via CSV or direct SQL\n"
            "- Verify data can be queried\n"
            "- Database file should be named 'sqlite.db'"
        ),
        required_files=["sqlite.db", "db/schema.cds"],
        success_criteria=[
            "Database file sqlite.db exists",
            "Tables are created from schema",
            "Required number of records are inserted",
            "Data can be queried successfully",
        ],
        hints=[
            "Use 'cds deploy --to sqlite' to create database",
            "CSV files in db/data/ are auto-loaded",
            "Or use sqlite3 CLI for direct inserts",
        ],
        focus_areas=["database deployment", "data loading", "SQL operations"],
    ),
]


# 业务领域池
BUSINESS_DOMAINS = [
    "bookshop", "inventory", "order management", "employee directory",
    "product catalog", "customer management", "invoice system", "travel booking",
    "project management", "warehouse", "library", "rental service",
]


# 各领域的示例实体与字段
ENTITY_EXAMPLES = {
    "bookshop": [
        ("Books", ["ID: UUID", "title: String(100)", "stock: Integer", "price: Decimal(10,2)"]),
        ("Authors", ["ID: UUID", "name: String(100)", "birthYear: Integer"]),
        ("Genres", ["ID: UUID", "name: String(50)", "description: String(500)"]),
    ],
    "inventory": [
        ("Products", ["ID: UUID", "name: String(100)", "quantity: Integer", "price: Decimal(10,2)"]),
        ("Warehouses", ["ID: UUID", "name: String(100)", "location: String(200)"]),
        ("Suppliers", ["ID: UUID", "name: String(100)", "contact: String(100)"]),
    ],
    "order management": [
        ("Orders", ["ID: UUID", "orderNumber: String(20)", "orderDate: Date", "total: Decimal(10,2)"]),
        ("OrderItems", ["ID: UUID", "quantity: Integer", "price: Decimal(10,2)"]),
        ("Customers", ["ID: UUID", "name: String(100)", "email: String(100)"]),
    ],
}


def build_cap_task(
    category: str,
    complexity: str = "medium",
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """按类别与复杂度采样一个 CAP 任务。

    参数:
        category: 任务类别（如 ``data_modeling``、``service_definition``）。
        complexity: 复杂度（``simple`` / ``medium`` / ``complex``）。
        domain: 业务领域；为 None 时随机选择。

    返回:
        包含题面、验收文件、通过标准等字段的字典。
    """
    if category == "data_modeling":
        templates = [t for t in DATA_MODELING_TEMPLATES if t.complexity == complexity]
    elif category == "service_definition":
        templates = [t for t in SERVICE_DEFINITION_TEMPLATES if t.complexity == complexity]
    elif category == "database_operations":
        templates = [t for t in DATABASE_OPERATIONS_TEMPLATES if t.complexity == complexity]
    else:
        raise ValueError(f"Unknown category: {category}")

    if not templates:
        raise ValueError(f"No templates found for {category}/{complexity}")

    template = random.choice(templates)

    if domain is None:
        domain = random.choice(BUSINESS_DOMAINS)

    entities = ENTITY_EXAMPLES.get(domain, ENTITY_EXAMPLES["bookshop"])

    entity1_name, entity1_fields = entities[0]
    entity2_name, entity2_fields = entities[1] if len(entities) > 1 else ("Related", ["ID: UUID"])
    entity3_name, entity3_fields = entities[2] if len(entities) > 2 else ("Extra", ["ID: UUID"])

    fields1_str = ", ".join(entity1_fields)
    fields2_str = ", ".join(entity2_fields)

    description = template.description_template.format(
        domain=domain,
        entity1=entity1_name,
        entity2=entity2_name,
        entity3=entity3_name,
        fields1=fields1_str,
        fields2=fields2_str,
        num_fields1=len(entity1_fields),
        num_fields2=len(entity2_fields),
        relationship_type=random.choice(["one-to-many", "many-to-many"]),
        service_name="CatalogService",
        num_records=random.randint(3, 10),
    )

    return {
        "task_description": description,
        "category": category,
        "complexity": complexity,
        "domain": domain,
        "required_files": template.required_files,
        "success_criteria": template.success_criteria,
        "hints": template.hints,
        "focus_areas": template.focus_areas,
        "entities": [entity1_name, entity2_name],
    }


def list_cap_templates() -> List[CapTaskTemplate]:
    """返回全部可用模板。"""
    return (
        DATA_MODELING_TEMPLATES +
        SERVICE_DEFINITION_TEMPLATES +
        DATABASE_OPERATIONS_TEMPLATES
    )
