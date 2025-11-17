from src.utils.logger import setup_logger, replace_output
from src.utils.configParser import ConfigParserWrapper
import os
def test_logger():
    log_parent_dir = ConfigParserWrapper().get_logging()[0]
    log_subdir = "test"
    log_dir = os.path.join(log_parent_dir, log_subdir)

    logger = setup_logger(log_dir= log_dir)
    print("Logging started")
    print(f"Log location: {log_dir}")

if __name__ == '__main__':
    test_logger()