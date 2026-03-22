'''module logger
'''
import logging
import os
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(ROOT_DIR, "logs")

LOG_FORMAT = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s"
handlers = [logging.StreamHandler()]

try:
    os.makedirs(logs_path, exist_ok=True)
    LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)
    handlers.insert(0, logging.FileHandler(LOG_FILE_PATH))
except OSError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=handlers,
)

