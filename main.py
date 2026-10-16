# driver
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("CNN_Training")

if __name__ == "__main__":
    logger.info("Starting the CNN Training application")
    # Here you can call your main functions or classes to start the application
    import test_log
    test_log.test_logging()