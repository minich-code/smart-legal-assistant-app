import sys
import logging
from typing import Any, Dict, Optional
from dataclasses import dataclass
from fastapi import HTTPException

# Fallback logger configuration (will be replaced by backend/components/logging.py)
logger = logging.getLogger("LegalRAG")
logging.basicConfig(level=logging.ERROR)


@dataclass(frozen=True)
class ErrorDetails:
    """
    A dataclass to hold error details for the Legal RAG system.

    Attributes:
        exc_type (type): The type of the exception.
        exc_value (BaseException): The exception instance.
        exc_traceback (Any): The traceback object.
    """
    exc_type: Optional[type]
    exc_value: Optional[BaseException]
    exc_traceback: Any


class LegalRAGException(Exception):
    """
    Custom exception class for the Enterprise Legal RAG system.

    Attributes:
        message (str): Formatted error message with details.
        error (Exception): The original exception instance.
        error_details (ErrorDetails): Structured error details (type, value, traceback).
        context (Optional[Dict]): Additional context (e.g., query, API service, document IDs).
        error_type (str): Type of error (e.g., EmbeddingError, RetrievalError).
        status_code (int): HTTP status code for FastAPI responses.
    """

    def __init__(
            self,
            error: Exception,
            error_type: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
            status_code: int = 500,
            log_immediately: bool = False,
    ) -> None:
        """
        Initialize the LegalRAGException.

        Args:
            error (Exception): The original exception.
            error_type (Optional[str]): Specific error type (e.g., EmbeddingError).
            context (Optional[Dict]): Additional context (e.g., query, service).
            status_code (int): HTTP status code for API responses (default: 500).
            log_immediately (bool): Whether to log the error immediately (default: True).
            :rtype: None
        """
        self.error = error
        self.context = context or {}
        self.error_type = error_type if error_type else type(error).__name__
        self.status_code = status_code

        # Capture traceback details safely
        error_details = sys.exc_info()
        self.error_details = ErrorDetails(*error_details)

        # Format the error message
        self.message = self._format_error_message()

        # Initialize base Exception with the formatted message
        super().__init__(self.message)

        # Log the error if requested
        if log_immediately:
            self.log_error()

    def _format_error_message(self) -> str:
        """
        Format detailed error information for logging and API responses.

        Returns:
            str: Formatted error message with file, line number, and context.
        """
        try:
            if self.error_details.exc_traceback:
                file_name = self.error_details.exc_traceback.tb_frame.f_code.co_filename
                line_number = self.error_details.exc_traceback.tb_lineno

                formatted_error_message = (
                    f"Error Type: {self.error_type}\n"
                    f"File: {file_name}\n"
                    f"Line Number: {line_number}\n"
                    f"Error Message: {str(self.error)}\n"
                    f"HTTP Status Code: {self.status_code}"
                )
            else:
                formatted_error_message = (
                    f"Error Type: {self.error_type}\n"
                    f"Error Message: {str(self.error)}\n"
                    f"HTTP Status Code: {self.status_code}"
                )

            if self.context:
                context_str = '\n'.join(f"  {k}: {v}" for k, v in self.context.items())
                formatted_error_message += f"\nContext:\n{context_str}"

            return formatted_error_message
        except Exception:
            # Fallback if anything goes wrong with formatting
            return f"Error Type: {self.error_type} | Error: {str(self.error)} | Status: {self.status_code}"

    def log_error(self, level: int = logging.ERROR) -> None:
        """
        Log the formatted error message at the specified level.

        Args:
            level (int): The logging level to use (default: logging.ERROR).
        """
        logger.log(level, self.message)

    def to_http_exception(self) -> HTTPException:
        """
        Convert the exception to a FastAPI HTTPException for API responses.

        Returns:
            HTTPException: FastAPI-compatible exception with status code and details.
        """
        return HTTPException(
            status_code=self.status_code,
            detail=self.message,
        )

    def __str__(self) -> str:
        """Return the formatted error message."""
        return self.message