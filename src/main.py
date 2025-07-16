# main.py
from exception import CustomException
import sys
from logger import logger

logger.info("🚀 Application Started")

try:
    x = 1 / 0  # Triggers ZeroDivisionError
except Exception as e:
    raise CustomException(e, sys)
