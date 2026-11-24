"""测试与辅助代码。"""

import logging
import os
import unittest
from codeharvest.logging_utils import init_logger

@unittest.skip("Skip to keep outputs clean.")
class TestLogger(unittest.TestCase):
    log_file = "test_codeharvest.log"

    def setUp(self):
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

        
        logger = logging.getLogger("codeharvest")
        for handler in logger.handlers[:]:  
            handler.close()  
            logger.removeHandler(handler)  

    def test_file_logging(self):
        
        logger = init_logger(level=logging.DEBUG, log_file=self.log_file)

        
        test_message = "This is a test log message"
        logger.debug(test_message)

        
        file_exists = os.path.exists(self.log_file)
        self.assertTrue(file_exists, f"Log file '{self.log_file}' should exist.")

        
        with open(self.log_file, "r") as f:
            log_contents = f.read()
            self.assertIn(test_message, log_contents)

    def tearDown(self):
        
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

if __name__ == "__main__":
    unittest.main()
