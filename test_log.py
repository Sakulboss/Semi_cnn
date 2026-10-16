# in every module
import logging
logger = logging.getLogger("CNN_Training.test_log")


def test_logging():
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")