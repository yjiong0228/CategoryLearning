from pathlib import Path
import yaml
from .base import PATHS


def load_config(filename: str | Path):
    return yaml.load(open(filename, encoding="utf8"), Loader=yaml.FullLoader)


MODEL_STRUCT = {}

for path in PATHS["configs"].glob("*.yaml"):
    globals()[path.stem.upper()] = load_config(path)

for filename in (PATHS["configs"] / "model_struct").glob("*.yaml"):
    MODEL_STRUCT[filename.stem] = load_config(filename)
