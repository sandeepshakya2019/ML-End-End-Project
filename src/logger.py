import logging
import os
from datetime import datetime

# Generate log filename with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define logs directory and create if not exists
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging globally
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Create a shared logger object for the entire app
logger = logging.getLogger("ml_pipeline")
