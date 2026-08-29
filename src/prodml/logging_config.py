import contextvars
import datetime as dt
import json
import logging
from typing import override

_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id",
    default="-",
)


def set_correlation_id(correlation_id: str) -> contextvars.Token[str]:
    return _correlation_id.set(correlation_id)


def reset_correlation_id(token: contextvars.Token[str]) -> None:
    _correlation_id.reset(token)


def get_correlation_id() -> str:
    return _correlation_id.get()


class CorrelationIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class DefaultFormatter(logging.Formatter):
    """A custom logging formatter that outputs logs in JSON format."""

    def __init__(
        self,
        fmt="%(levelname)s:\t%(asctime)s - %(module)s - %(message)s",
        datefmt="%H:%M:%S%z",
    ):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format_time(self, record, datefmt=None):
        created = dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc)
        return created.strftime(datefmt) if datefmt else created.isoformat()

    @override
    def format(self, record: logging.LogRecord) -> str:
        record.levelname = f"[{record.levelname}]"
        return super().format(record)


class JSONFormatter(DefaultFormatter):
    """A custom logging formatter that outputs logs in JSON format."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.format_time(record, self.datefmt),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        if record.stack_info:
            log_record["stack_info"] = self.formatStack(record.stack_info)

        # Add any extra fields from the log record
        # (fields added via logger.info("msg", extra={"key": "value"}))
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "taskName",
            ]:
                log_record[key] = value

        return json.dumps(log_record, default=str)


FORMATTERS: dict[str, type[logging.Formatter]] = {
    "json": JSONFormatter,
    "default": DefaultFormatter,
}

LOG_LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "critical": logging.CRITICAL,
}


def setup_logger(log_level: str = "info", log_format: str = "default") -> None:
    """
    Set up the root logger with the specified log level and format.

    Args:
        log_level (str): The log level to use (e.g., "debug", "info").
        log_format (str): The log format to use (e.g., "default", "json").
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure Formatter
    formatter_cls: logging.Formatter = FORMATTERS.get(log_format, DefaultFormatter)
    formatter = formatter_cls()

    # Create new handler
    # TODO: Consider extending this to support file logging.
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Add CorrelationIdFilter to the handler
    handler.addFilter(CorrelationIdFilter())

    root_logger.setLevel(LOG_LEVELS.get(log_level.lower(), logging.INFO))
    root_logger.addHandler(handler)
