"""代码生成任务模型：由测试目标与规格说明组装出的题目。"""

from codeharvest.core.targets import FunctionTestTarget, MethodTestTarget

class CodegenTask:

    spec: str

class FunctionCodegenTask(CodegenTask, FunctionTestTarget):
    @classmethod
    def from_fut_and_spec(cls, fut: FunctionTestTarget, spec: str):
        fut_dump = fut.model_dump()
        fut_dump.update({"spec": spec})
        return cls(**fut_dump)

class MethodCodegenTask(CodegenTask, MethodTestTarget):
    @classmethod
    def from_mut_and_spec(cls, mut: MethodTestTarget, spec: str):
        mut_dump = mut.model_dump()
        mut_dump.update({"spec": spec})
        return cls(**mut_dump)

def make_codegen_task(obj: FunctionTestTarget | MethodTestTarget, spec: str):
    if isinstance(obj, FunctionTestTarget):
        return FunctionCodegenTask.from_fut_and_spec(obj, spec)
    elif isinstance(obj, MethodTestTarget):
        return MethodCodegenTask.from_mut_and_spec(obj, spec)
    else:
        raise ValueError("Unknown input type")
