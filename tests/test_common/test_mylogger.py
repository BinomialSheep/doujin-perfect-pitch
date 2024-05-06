import sys
from pathlib import Path
import os

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(root_dir))


from common import mylogger


def test_mylogger():
    logger = mylogger.setup_logger("test_name", "testlog.log")
    logger.info("test_mylogger_method_name")
    assert True
