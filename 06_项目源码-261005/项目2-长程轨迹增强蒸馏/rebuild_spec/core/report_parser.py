"""pytest-json-report 报告的解析：兼容新旧两种结构，供评测与 rollout 判分共用。"""

import json
from pathlib import Path
from typing import Optional, Union


def load_report(report_path: Union[str, Path]) -> Optional[dict]:
    """读取 report.json；不存在或结构不可识别时返回 None。"""
    path = Path(report_path)
    if not path.exists():
        return None
    with open(path, "r") as f:
        report = json.load(f)
    if isinstance(report, dict):
        return report
    if isinstance(report, list):
        # 旧版 pytest-json-report 直接输出用例列表
        return {"tests": report}
    return None


def report_call_outcomes(report: dict) -> dict[str, dict]:
    """把报告整理成 {nodeid: {"outcome", "duration", "longrepr"}}。

    兼容新版（tests[i].call）与旧版（tests[i].when == "call"）两种结构。
    """
    outcomes = {}
    tests = report.get("tests", [])
    if "created" in report:
        # 新版结构
        for item in tests:
            if "call" in item:
                call = item["call"]
                outcomes[item["nodeid"]] = {
                    "outcome": call.get("outcome", "failed"),
                    "duration": call.get("duration", 0),
                    "longrepr": call.get("longrepr", item.get("longrepr", "")),
                }
    else:
        for item in tests:
            if item.get("when") == "call":
                outcomes[item["nodeid"]] = {
                    "outcome": item.get("outcome", "failed"),
                    "duration": item.get("duration", 0),
                    "longrepr": item.get("longrepr", ""),
                }
    return outcomes


def parse_report(
    report_path: Union[str, Path], test_ids: list[str]
) -> dict:
    """汇总通过率：{pass_rate, num_passed, num_tests, outcomes}。

    报告中缺失的用例按失败计。
    """
    report = load_report(report_path)
    if report is None:
        return {
            "pass_rate": 0.0,
            "num_passed": 0,
            "num_tests": len(test_ids),
            "outcomes": {},
        }
    outcomes = report_call_outcomes(report)
    num_passed = 0
    for test_id in test_ids:
        call = outcomes.get(test_id)
        if call is not None and call["outcome"] in ("passed", "xfail"):
            num_passed += 1
    num_tests = len(test_ids)
    pass_rate = num_passed / num_tests if num_tests else 0.0
    return {
        "pass_rate": pass_rate,
        "num_passed": num_passed,
        "num_tests": num_tests,
        "outcomes": outcomes,
    }


def collection_errors(report: dict) -> list[str]:
    """提取 collection 阶段的错误信息列表。"""
    errors = []
    for err in report.get("collectors", []) or []:
        if err.get("outcome") == "failed":
            errors.append(str(err.get("longrepr", err)))
    return errors


__all__ = []
