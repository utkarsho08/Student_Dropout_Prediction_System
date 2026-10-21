import logging
from config.settings import LOGS_DIR

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        file_handler = logging.FileHandler(LOGS_DIR / "app.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger