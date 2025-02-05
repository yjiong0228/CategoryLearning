import yaml
import os
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.yml') 
with open(CONFIG_FILE_PATH, 'r') as f:
    config = yaml.safe_load(f)