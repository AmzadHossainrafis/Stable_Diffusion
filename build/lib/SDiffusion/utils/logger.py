import os
import sys
import logging
import datetime as dt
from pathlib import Path

today = dt.datetime.today().strftime("%Y-%m-%d")
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"


dir = r"/home/amzad/Desktop/stable_diffusion/logs/"
print(dir)

log_filepath = os.path.join(dir, f"running_logs_{today}.log")
os.makedirs(dir, exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[logging.FileHandler(log_filepath), logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("diffusion_model")
