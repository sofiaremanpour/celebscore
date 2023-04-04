import logging
import sys

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create a logger object for the entire application
logger = logging.getLogger("celebscore")
