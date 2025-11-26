import logging
import os
import sys
from datetime import datetime
import traceback

def _default_log_name():
    """Use the current date and time as default log name"""
    return datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")


def setup_logger(log_dir, log_file="", level=logging.INFO, logging_enabled=True,
                 console_enabled=True):
    """Setup logger that writes to both file and console

    :param log_dir: Directory to save log files
    :param log_file: Name of log file (auto-generated if empty)
    :param level: Logging level (default: INFO)
    :param logging_enabled: Write to file
    :param console_enabled: Write to console
    :return: Configured logger instance
    """
    if not log_file:
        log_file = _default_log_name()

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers = []  # Remove existing handlers

    # Use different formatters for console and file
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    if logging_enabled:
        log_path = os.path.join(log_dir, log_file)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class LoggerWriter:
    """Redirect stdout/stderr to both console AND logger (file only)"""

    def __init__(self, logger, level, original_stream):
        self.logger = logger
        self.level = level
        self.original_stream = original_stream  # Keep original stdout/stderr
        self.buffer = ""

    def write(self, message):
        """Write to both console and log file (filtering noise from file)"""
        # Always write to console for interactive output
        self.original_stream.write(message)

        if not message:
            return

        # Handle progress bar characters - don't log them to file
        if any(c in message for c in ['|', '#', '▌', '▓', '░']):
            return

        self.buffer += message

        # Log complete lines to file only
        if '\n' in self.buffer:
            lines = self.buffer.split('\n')
            for line in lines[:-1]:
                line = line.strip()
                if line and not self._is_noise(line):
                    self.logger.log(self.level, line)
            self.buffer = lines[-1]

    def flush(self):
        """Flush to both console and file"""
        self.original_stream.flush()
        if self.buffer.strip() and not self._is_noise(self.buffer):
            self.logger.log(self.level, self.buffer.strip())
            self.buffer = ""

    @staticmethod
    def _is_noise(message):
        """Filter out common noise like empty progress indicators"""
        noise_patterns = ['?', 'it/s', 'examples/s', 'files/s', '0%']
        return any(pattern in message for pattern in noise_patterns)

def replace_stdout(logger):
    """Redirect stdout to both console AND logger file"""
    sys.stdout = LoggerWriter(logger, logging.INFO, sys.stdout)


def replace_stderr(logger):
    """Redirect stderr to both console AND logger file"""
    sys.stderr = LoggerWriter(logger, logging.ERROR, sys.stderr)


def replace_output(logger):
    """Redirect both stdout and stderr to file (console output unchanged)"""
    replace_stdout(logger)
    replace_stderr(logger)
