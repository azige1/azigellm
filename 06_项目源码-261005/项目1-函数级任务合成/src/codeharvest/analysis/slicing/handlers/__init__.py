"""切片语句处理器。"""

from codeharvest.analysis.slicing.handlers.base import StatementHandler
from codeharvest.analysis.slicing.handlers.imports import ImportStatementHandler
from codeharvest.analysis.slicing.handlers.definitions import (
    DefinitionStatementHandler,
)
