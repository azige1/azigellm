"""测试与辅助代码。"""

import os
import ast
import sys
import unittest

from codeharvest.parallel import run_jobs_parallel_iter
from codeharvest.utils.io import read_callable_records
from codeharvest.analysis.slicing import CodeDependencySlicer
from codeharvest.core import RepoRecord, FunctionRecord, MethodRecord
from codeharvest.logging_utils import logger
from codeharvest.workspace import REPOS_DIR, EXTRACTION_DIR

ALL_FUNCTIONS = (
    read_callable_records(EXTRACTION_DIR / "codeharvest_v1_extracted.json")
    if (EXTRACTION_DIR / "codeharvest_v1_extracted.json").exists()
    else []
)

def get_slice(func_idx: int):
    func = ALL_FUNCTIONS[func_idx]
    if not os.path.exists(REPOS_DIR / func.repo.local_repo_path.split("/")[-1]):
        logger.warning(
            f"Skipping {func.id} - no local repo path {func.repo.local_repo_path}"
        )
        return
    if isinstance(func, FunctionRecord):
        slicer = CodeDependencySlicer.from_function_models(func)
        slicer.run()
    elif isinstance(func, MethodRecord):
        slicer = CodeDependencySlicer.from_class_models(func.parent_class)
        slicer.run()
    else:
        raise ValueError(f"Unknown function type: {type(func)}")

    
    return slicer.dependency_graph.unparse()

@unittest.skipIf(not ALL_FUNCTIONS, "path not found")
class TestSlicerMain(unittest.TestCase):

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    

    
    
    
    
    

    def test_slicer_specific(self):
        functions = [
            f for f in ALL_FUNCTIONS if f.id == "klongpy.monads.eval_monad_range"
        ][0]
        assert isinstance(functions, FunctionRecord)

        slicer = CodeDependencySlicer.from_function_models(functions)
        slicer.run()
        

    def test_slicer_specific2(self):
        functions = [
            f for f in ALL_FUNCTIONS if f.id == "sacpy.XrTools.spec_moth_yrmean"
        ][0]
        assert isinstance(functions, FunctionRecord)

        slicer = CodeDependencySlicer.from_function_models(functions)
        slicer.run()
        

    def test_slicer_multiple(self):
        functions = [
            f
            for f in ALL_FUNCTIONS
            if isinstance(f, FunctionRecord)
            if f.file.file_module.module_id.identifier == "chainfury.base"
        ]

        slicer = CodeDependencySlicer.from_function_models(functions)
        slicer.run()
        

    def test_slicer_multiple_from_file(self):
        file = REPOS_DIR / "klongpy/klongpy/monads.py"
        with open(file, "r") as f:
            tree = ast.parse(f.read())
        functions = [f for f in tree.body if isinstance(f, ast.FunctionDef)]
        file_obj_dict = {
            "file_module": {
                "module_id": {"identifier": "klongpy.monads"},
                "module_type": "file",
                "repo": {
                    "repo_org": "",
                    "repo_name": "klongpy",
                    "repo_id": "klongpy",
                    "local_repo_path": "repos/klongpy",
                },
            }
        }
        function_dicts = []
        for function in functions:
            function_dicts.append(
                {
                    "function_id": {"identifier": function.name},
                    "function_name": function.name,
                    "function_complexity": "blah",
                    "function_code": ast.unparse(function),
                    "file": file_obj_dict,
                    "sliced_context": None,
                    "full_context": None,
                }
            )
        functions = [FunctionRecord(**f) for f in function_dicts]

        slicer = CodeDependencySlicer.from_function_models(functions)
        slicer.run()
        

    def test_slicer_multiple2(self):
        file = str(REPOS_DIR / "autoagents/autoagents/eval/hotpotqa/eval_async.py")

        repo = RepoRecord(
            repo_org="",
            repo_name="autoagents",
            repo_id="autoagents",
            local_repo_path=f"repos/autoagents",
        )

        with open(file, "r") as f:
            tree = ast.parse(f.read())
        function_names = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]
        function_objs = []
        for function_name in function_names:
            function_objs.append(
                FunctionRecord.from_name_file_repo(function_name, file, repo)
            )

        slicer = CodeDependencySlicer.from_function_models(function_objs)
        slicer.run()
        

    def test_slicer_multiple3(self):
        file = str(REPOS_DIR / "stable-ts/stable_whisper/audio.py")

        repo = RepoRecord(
            repo_org="",
            repo_name="stable-ts",
            repo_id="stable-ts",
            local_repo_path=f"repos/stable-ts",
        )

        with open(file, "r") as f:
            tree = ast.parse(f.read())

        function_names = [f.name for f in tree.body if isinstance(f, ast.FunctionDef)]
        function_objs = []

        for function_name in function_names:
            function_objs.append(
                FunctionRecord.from_name_file_repo(function_name, file, repo)
            )

        slicer = CodeDependencySlicer.from_function_models(function_objs)
        slicer.run()

        

    def test_slicer_repo_1(self):
        repo_name = "social-network-link-prediction"
        repo_path = REPOS_DIR / repo_name

        repo = RepoRecord(
            repo_org="",
            repo_name=repo_name,
            repo_id=repo_name,
            local_repo_path=f"repos/{repo_name}",
        )

        all_functions = []

        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, "r") as f:
                        tree = ast.parse(f.read())
                    function_names = [
                        f.name for f in tree.body if isinstance(f, ast.FunctionDef)
                    ]
                    function_objs = [
                        FunctionRecord.from_name_file_repo(f, file_path, repo)
                        for f in function_names
                    ]
                    all_functions.extend(function_objs)

        slicer = CodeDependencySlicer.from_function_models(all_functions)
        slicer.run()
        
