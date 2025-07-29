

import logging
import logging.config
import os
from datetime import datetime
from pythonjsonlogger import jsonlogger
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional
from pathlib import Path

# Constants
FORMATTER_JSON = "json"
FORMATTER_DETAILED = "detailed"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_COUNT = 20
DEFAULT_LOGGER_NAME = "LegalRAG"

class LoggerConfigurator:
    """
    Configures logging for the Enterprise Legal RAG system with file and console outputs.
    Supports JSON logging for files and detailed formatting for console output.
    """

    def __init__(
        self,
        log_dir: str = os.getenv("LOG_DIR", str(Path(__file__).parent.parent.parent / "logs")),
        log_file_name_pattern: str = "%Y-%m-%d_%H-%M-%S",
        max_bytes: int = DEFAULT_MAX_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        formatters: Optional[Dict] = None,
        handlers: Optional[Dict] = None,
        loggers: Optional[Dict] = None,
        log_level: str = "INFO",
    ):
        """
        Initialize the logger configurator.

        Args:
            log_dir: Directory for log files (default: project_root/logs).
            log_file_name_pattern: DateTime pattern for log file names.
            max_bytes: Maximum size of each log file.
            backup_count: Number of backup files to keep.
            formatters: Custom formatters configuration.
            handlers: Custom handlers configuration.
            loggers: Custom loggers configuration.
            log_level: Default logging level (default: INFO).
        """
        self.log_dir = log_dir
        self.log_file_name_pattern = log_file_name_pattern
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.log_level = log_level.upper()  # Ensure uppercase for consistency

        # Initialize configurations
        self.formatters = formatters if formatters else self._default_formatters()
        self.handlers = handlers if handlers else self._default_handlers()
        self.loggers = loggers if loggers else self._default_loggers()

        self._configure_logging()

    @staticmethod
    def _default_formatters() -> Dict[str, Dict[str, Any]]:
        """Define default formatters for different logging outputs."""
        return {
            "standard": {
                "format": "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"
            },
            FORMATTER_DETAILED: {
                "format": "[%(asctime)s] [%(levelname)s] %(name)s - %(module)s:%(lineno)d - %(message)s"
            },
            FORMATTER_JSON: {
                "()": jsonlogger.JsonFormatter,
                "format": "%(asctime)s %(name)s %(levelname)s %(module)s %(lineno)d %(message)s"
            },
        }

    def _default_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Define default handlers for file and console logging."""
        log_file_path = self._get_log_file_path()
        return {
            "file": {
                "level": self.log_level,
                "class": "logging.handlers.RotatingFileHandler",
                "filename": log_file_path,
                "maxBytes": self.max_bytes,
                "backupCount": self.backup_count,
                "encoding": "utf8",
                "formatter": FORMATTER_JSON,
            },
            "console": {
                "level": self.log_level,
                "class": "logging.StreamHandler",
                "formatter": FORMATTER_DETAILED,
                "stream": "ext://sys.stdout",
            },
        }

    def _default_loggers(self) -> Dict[str, Dict[str, Any]]:
        """Define default logger configuration."""
        return {
            DEFAULT_LOGGER_NAME: {
                "handlers": ["file", "console"],
                "level": self.log_level,
                "propagate": False,
            }
        }

    def _get_log_file_path(self) -> str:
        """Generate and ensure the log file path exists."""
        try:
            log_file_name = f"{datetime.now().strftime(self.log_file_name_pattern)}.log"
            log_path = Path(self.log_dir) / log_file_name
            log_path.parent.mkdir(parents=True, exist_ok=True)
            return str(log_path)
        except OSError as e:
            # Fallback to a safe location if log directory creation fails
            fallback_path = Path.home() / "logs" / "legalrag.log"
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Warning: Could not create log directory {self.log_dir}. Using fallback: {fallback_path}")
            return str(fallback_path)

    def _configure_logging(self) -> None:
        """Apply the logging configuration."""
        try:
            logging_config = {
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": self.formatters,
                "handlers": self.handlers,
                "loggers": self.loggers,
            }
            logging.config.dictConfig(logging_config)
        except Exception as e:
            # Fallback to basic configuration if advanced config_params fails
            print(f"Failed to configure advanced logging: {e}")
            logging.basicConfig(
                level=getattr(logging, self.log_level, logging.INFO),
                format="[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
                handlers=[logging.StreamHandler()]
            )
            print("Using basic logging configuration as fallback")

    @staticmethod
    def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Name of the logger to retrieve.

        Returns:
            logging.Logger: Configured logger instance.
        """
        return logging.getLogger(name)


# Initialize the logger configurator with error handling
try:
    logger_configurator = LoggerConfigurator()
    logger = logger_configurator.get_logger()
except Exception as e:
    print(f"Failed to initialize logger configurator: {e}")
    # Fallback logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(DEFAULT_LOGGER_NAME)
    logger.error(f"Using fallback logger due to configuration error: {e}")