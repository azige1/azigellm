"""基于 PyCG 为整个仓库构建调用图。"""

import os
import sys
import shutil

try:
    import PyCG as pycg

    sys.modules["pycg"] = pycg
except ImportError as e:
    print(f"Failed to import PyCG: {e}")

from pycg import formats  
from pycg.pycg import CallGraphGenerator as CallGraphGeneratorPyCG  

from codeharvest.analysis.imports.rewriter import ImportRewriter

class CallGraphBuilder:
    @staticmethod
    def build_repo_call_graph(repo_path: str, max_iter: int = -1) -> dict:
        try:
            repo_path = ImportRewriter.rewrite_repo_imports(repo_path)

            entry_points = []
            for root, dirs, files in os.walk(repo_path):
                for file in files:
                    if file.endswith(".py"):
                        entry_points.append(os.path.abspath(os.path.join(root, file)))

            cg_generator = CallGraphGeneratorPyCG(
                entry_points, repo_path, max_iter, operation="call-graph"
            )
            cg_generator.analyze()
            cgraph = formats.Simple(cg_generator).generate()
        finally:
            shutil.rmtree(repo_path)
        return cgraph
