"""规格提炼的采集记录解析测试：沙箱日志格式与异常记录。"""

import unittest

from codeharvest.synthesis.specs.parsing import (
    collect_captured_types,
    format_captured_value,
)


class CapturedRecordParsingTest(unittest.TestCase):

    def test_sandbox_record_format(self):
        record = {
            "inputs": {"text": "'a,b'"},
            "input_types": {"text": "builtins.str"},
            "output": "['a', 'b']",
            "output_type": "builtins.list",
        }
        rendered = format_captured_value(record)
        self.assertIn("text='a,b'", rendered)
        self.assertIn("['a', 'b']", rendered)

    def test_exception_record(self):
        record = {
            "inputs": {"sep": "''"},
            "input_types": {"sep": "builtins.str"},
            "exception_type": "builtins.ValueError",
            "exception_message": "separator must not be empty",
        }
        rendered = format_captured_value(record)
        self.assertIn("builtins.ValueError: separator must not be empty", rendered)
        input_types, output_types = collect_captured_types([record])
        self.assertIn("sep : builtins.str", input_types)
        self.assertEqual(output_types, "")


if __name__ == "__main__":
    unittest.main()
