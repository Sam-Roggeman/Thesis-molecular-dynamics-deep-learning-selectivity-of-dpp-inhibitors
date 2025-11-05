import logging
import sys
from datetime import datetime


def _default_log_name():
    """
    Use the current date and time as default log name
    :return:
    """
    return datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")


def setup_logger(log_file="", level=logging.INFO):
    """Setup logger that writes to both file and console"""

    if not log_file:
        log_file = _default_log_name()
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers
    logger.handlers = []

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():  # Ignore empty lines
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

def replace_stdout(logger):
    sys.stdout = LoggerWriter(logger, logging.INFO)
def replace_stderr(logger):
    sys.stderr = LoggerWriter(logger, logging.ERROR)
def replace_output(logger):
    replace_stdout(logger)
    replace_stderr(logger)

