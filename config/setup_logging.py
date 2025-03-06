"""
Logging configuration for the Agent Platform.
"""

import os
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Optional

def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file in addition to console
        
    Returns:
        Root logger instance
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with a higher log level to reduce console noise
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add a filter to prevent certain messages from being logged as errors
    class StreamlitInfoFilter(logging.Filter):
        def filter(self, record):
            # Filter out INFO messages that might be in error streams
            if record.levelno == logging.ERROR:
                msg = record.getMessage()
                # Check for various formats of INFO messages
                if any(pattern in msg for pattern in ["INFO:", ":INFO:", "- INFO -", 
                                                      "INFO -", "- INFO"]):
                    return False
            return True
    
    console_handler.addFilter(StreamlitInfoFilter())
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        # Create logs directory if it doesn't exist
        logs_dir = Path("data/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp
        log_file = logs_dir / f"platform_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create file handler for all logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Also create an error log that only contains ERROR and CRITICAL
        error_log = logs_dir / "errors.log"
        error_handler = logging.FileHandler(error_log)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        # Add the same filter to the error handler
        error_handler.addFilter(StreamlitInfoFilter())
        root_logger.addHandler(error_handler)
        
        logger = logging.getLogger("setup_logging")
        logger.info(f"Logging initialized. Log file: {log_file}")
    
    # Set Streamlit logger to a higher level to reduce noise
    streamlit_logger = logging.getLogger('streamlit')
    streamlit_logger.setLevel(logging.WARNING)
    
    return root_logger

def get_env_log_level() -> str:
    """
    Get log level from environment variables.
    
    Returns:
        Log level string
    """
    return os.environ.get("LOG_LEVEL", "INFO").upper()

if __name__ == "__main__":
    # Test the logging configuration
    logger = setup_logging(log_level=get_env_log_level())
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
