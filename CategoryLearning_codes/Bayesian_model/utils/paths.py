"""Project paths for the migrated Model 0826 package."""
from __future__ import annotations

from pathlib import Path


UTILS_DIR = Path(__file__).resolve().parent
BAYESIAN_STATE_DIR = UTILS_DIR.parent
CODES_DIR = BAYESIAN_STATE_DIR.parent
ROOT_DIR = CODES_DIR.parent
SRC_DIR = ROOT_DIR / "src"

CONFIGS_DIR = ROOT_DIR / "configs"
LOGS_DIR = ROOT_DIR / "logs"
RESULTS_DIR = ROOT_DIR / "results"

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
TASK2_PROCESSED_PATH = PROCESSED_DATA_DIR / "Task2_processed.csv"
TASK1B_ERRORSUMMARY_PATH = PROCESSED_DATA_DIR / "Task1b_errorsummary_24.csv"
TASK1B_ERRORSUMMARY_72_PATH = PROCESSED_DATA_DIR / "Task1b_errorsummary_72.csv"

SIMULATION_RESULTS_DIR = RESULTS_DIR / "state-based-simulation"
