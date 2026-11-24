"""基于依赖切片的最小上下文构建器。"""

import io
import ast
import contextlib

from codeharvest.core import FunctionRecord, MethodRecord, ClassRecord
from codeharvest.synthesis.context.builder import ContextBuilder
from codeharvest.synthesis.context.formatting import ContextOutputFormat
from codeharvest.analysis.slicing import CodeDependencySlicer, SliceOutputMode

class SlicedContextBuilder(ContextBuilder):

    def __init__(
        self,
        func_meth: FunctionRecord | MethodRecord,
        max_context_size: int | None = None,
        format: ContextOutputFormat = ContextOutputFormat.MARKDOWN_FILES,
    ):
        super().__init__(func_meth, max_context_size, format)
        self.context_type = "sliced"
        self.construct_context()

    def construct_context(self):
        with contextlib.redirect_stdout(io.StringIO()):
            if isinstance(self.func_meth, MethodRecord):
                slicer = CodeDependencySlicer.from_class_models(self.func_meth.parent_class)
            elif isinstance(self.func_meth, FunctionRecord):
                slicer = CodeDependencySlicer.from_function_models(self.func_meth)
            elif isinstance(self.func_meth, ClassRecord):
                slicer = CodeDependencySlicer.from_class_models(self.func_meth)
            else:
                raise ValueError("Unknown input type")

            slicer.run()

            slice_format = SliceOutputMode.MARKDOWN_FILES
            self.context = slicer.dependency_graph.unparse(unparse_type=slice_format)
            self.file2code = slicer.dependency_graph.unparse_by_file()

        
        if self.max_context_size and self.context_size > self.max_context_size:
            self.truncate_context()
