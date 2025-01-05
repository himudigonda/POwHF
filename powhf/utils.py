import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(filename)s.%(funcName)s :: %(message)s",
)


def debug_log(message):
    """Logs a debug message with standard formatting."""
    logging.debug(message)


def info_log(message):
    """Logs an info message with standard formatting."""
    logging.info(message)
