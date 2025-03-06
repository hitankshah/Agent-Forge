"""
Utilities for consistent error handling and logging across the application.
"""

import logging
import traceback
import sys
from typing import Optional, Any, Dict, Union
from functools import wraps

logger = logging.getLogger("error_handling")

class ErrorCodes:
    """Standard error codes for the application"""
    JSON_PARSING_ERROR = "JSON_PARSING_ERROR"
    LLM_API_ERROR = "LLM_API_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    INVALID_PARAMETERS = "INVALID_PARAMETERS"
    DEPLOYMENT_ERROR = "DEPLOYMENT_ERROR"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"

def log_error(error_code: str, message: str, exception: Optional[Exception] = None, 
              context: Optional[Dict[str, Any]] = None):
    """
    Log an error with standardized format.
    
    Args:
        error_code: Standard error code from ErrorCodes class
        message: Human-readable error message
        exception: Optional exception object
        context: Optional additional context information
    """
    error_data = {
        "error_code": error_code,
        "message": message
    }
    
    if exception:
        error_data["exception_type"] = type(exception).__name__
        error_data["exception_message"] = str(exception)
    
    if context:
        error_data["context"] = context
    
    # Log full error details
    logger.error(error_data)
    
    # If there's an exception, also log the traceback at debug level
    if exception:
        logger.debug(f"Exception traceback for {error_code}:", exc_info=exception)
    
    return error_data

def handle_exceptions(error_code: str = ErrorCodes.UNEXPECTED_ERROR):
    """
    Decorator for handling exceptions with standardized logging.
    
    Args:
        error_code: Default error code to use if an exception occurs
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error(
                    error_code=error_code,
                    message=f"Error in {func.__name__}: {str(e)}",
                    exception=e,
                    context={"args": str(args), "kwargs": str(kwargs)}
                )
                # Re-raise the exception
                raise
        return wrapper
    return decorator

def format_user_error_message(error_data: Dict[str, Any]) -> str:
    """
    Format an error message for display to the user.
    
    Args:
        error_data: Error data dictionary from log_error
    
    Returns:
        User-friendly error message
    """
    # Basic error message
    message = error_data.get("message", "An unexpected error occurred")
    
    # Add suggestion based on error code
    error_code = error_data.get("error_code", ErrorCodes.UNEXPECTED_ERROR)
    
    suggestions = {
        ErrorCodes.JSON_PARSING_ERROR: "This might be due to an invalid response format. Try a different prompt or model.",
        ErrorCodes.LLM_API_ERROR: "There was an issue connecting to the AI service. Check your API key and internet connection.",
        ErrorCodes.DATABASE_ERROR: "There was a problem with the database. Please try again later.",
        ErrorCodes.AGENT_NOT_FOUND: "The requested agent could not be found. It may have been deleted or not yet created.",
        ErrorCodes.DEPLOYMENT_ERROR: "Failed to deploy the agent. Please check your configuration and try again."
    }
    
    suggestion = suggestions.get(error_code, "Please try again or contact support if the issue persists.")
    
    return f"{message}\n\nSuggestion: {suggestion}"
