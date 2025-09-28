"""
Root of the tree
"""
from pathlib import Path
import logging.config
from logging import getLogger
from datetime import datetime as dt
from .console_styles import print
# Useful Paths

PATHS = {}

PATHS["utils"] = Path(__file__).parent
PATHS["src"] = PATHS["utils"].parent.parent
PATHS["root"] = PATHS["src"].parent
PATHS["configs"] = PATHS["root"] / "configs"
PATHS["logs"] = PATHS["root"] / "logs"

for k, v in PATHS.items():
    v.mkdir(parents=True, exist_ok=True)

log_filename = str(PATHS['logs'] /
                   f'Run_{ dt.strftime(dt.now(), "%Y%m%d_%H%M%S")}.log')
print(log_filename, s=1)
config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
        'detailed': {
            'format':
            '%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout',
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': log_filename,
            'mode': 'a',
        },
    },
    'loggers': {
        'cat-learning': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console']
    }
}

LOGGER = getLogger("cat-learning")
LOGGER.info("logger is running normally.")
__ALL__ = ["PATHS", "LOGGER"]
