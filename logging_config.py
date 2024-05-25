import os
import logging
from datetime import datetime

LOG_FOLDER = 'log'
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

log_filename = os.path.join(LOG_FOLDER, datetime.now().strftime('%Y-%m-%d') + '.log')

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
