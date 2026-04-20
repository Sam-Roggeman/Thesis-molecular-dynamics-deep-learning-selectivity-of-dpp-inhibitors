# logger_config.py
import logging as logging_module
import os

# Global variable to hold the logger instance
_logger = None
_LOGGER_NAME = "my_app"

# define log_modes from logging module for easy access
DEBUG = logging_module.DEBUG
INFO = logging_module.INFO
WARNING = logging_module.WARNING
ERROR = logging_module.ERROR
CRITICAL = logging_module.CRITICAL
def init_logger(log_mode, model_dir, log_file):
    """Initialize the global logger (call once at startup)"""
    global _logger
    _logger = logging_module.getLogger(_LOGGER_NAME)
    _logger.setLevel(log_mode)

    os.makedirs(model_dir, exist_ok=True)
    formatter = logging_module.Formatter("%(levelname)s: %(message)s")

    log_file_path = os.path.join(model_dir, log_file)
    file_handler = logging_module.FileHandler(log_file_path, mode='w')
    file_handler.setFormatter(formatter)

    console_handler = logging_module.StreamHandler()
    console_handler.setFormatter(formatter)

    # Replace existing handlers to avoid stale paths and duplicate logs.
    _logger.handlers.clear()
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    _logger.propagate = False
    
    return _logger

def get_logger():
    """Get the configured logger from anywhere"""
    global _logger
    if _logger is None:
        _logger = logging_module.getLogger(_LOGGER_NAME)
    return _logger