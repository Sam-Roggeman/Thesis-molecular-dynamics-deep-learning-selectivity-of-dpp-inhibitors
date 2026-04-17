# logger_config.py
import logging as logging_module
import os

# Global variable to hold the logger instance
_logger = None

# define log_modes from logging module for easy access
DEBUG = logging_module.DEBUG
INFO = logging_module.INFO
WARNING = logging_module.WARNING
ERROR = logging_module.ERROR
CRITICAL = logging_module.CRITICAL
def init_logger(log_mode, model_dir, log_file):
    """Initialize the global logger (call once at startup)"""
    global _logger
    _logger = logging_module.getLogger('my_app')
    _logger.setLevel(log_mode)
    
    if not _logger.handlers:
        formatter = logging_module.Formatter("%(levelname)s: %(message)s")
        
        log_file_path = os.path.join(model_dir, log_file)
        file_handler = logging_module.FileHandler(log_file_path, mode='w')
        file_handler.setFormatter(formatter)
        
        console_handler = logging_module.StreamHandler()
        console_handler.setFormatter(formatter)
        
        _logger.addHandler(file_handler)
        _logger.addHandler(console_handler)
    
    return _logger

def get_logger():
    """Get the configured logger from anywhere"""
    if _logger is None:
        # Return a default logger if not initialized
        logging_module.basicConfig(level=logging_module.DEBUG)
        return logging_module.getLogger()
    return _logger