"""测试与辅助代码。"""

import unittest

from codeharvest.core import RepoRecord
from codeharvest.extraction.targets.repo_data import extract_repo_targets

class TestExtractRepoData(unittest.TestCase):
    def test_1(self):
        repo_dict = {
            "repo_org": "",
            "repo_name": "klongpy",
            "repo_id": "klongpy",
            "local_repo_path": "repos/klongpy",
            "_cached_callgraph": None,
        }
        repo = RepoRecord(**repo_dict)

        extract_repo_targets(repo)
