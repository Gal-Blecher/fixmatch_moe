import logging
from config import setup

def get_logger(name):
    path = './models/' + setup['experiment_name']
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(f"{path}/log_info.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    if handler not in logger.handlers:
        # Add the handler to the logger
        logger.addHandler(handler)

    logger.addHandler(handler)

    return logger