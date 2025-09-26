from pathlib import Path
import pyyaml


CONFIG_PATH = Path(__file__).parent.parent.parent.parent /"configs"

def load_config(filename:str|Path):
    return pyyaml.load(filename)


for path in CONFIG_PATH.rglob(".yaml"):
    globals()[path.stem.upper()] = load_config(path)
