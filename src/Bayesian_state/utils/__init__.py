"""Small compatibility surface for historically package-level utilities.

New code should import from the owning utility module directly.
"""

from .base import LOGGER, PATHS, configure_logging
from .basic_stat import cdist, entropy, euc_dist, softmax
from .classical_tools import two_factor_decay
from .console_styles import (
    COMPOSITE_STYLES,
    PRINT_RESUME,
    PRINT_STYLES,
    SINGLE_STYLES,
    compose_print,
    gen_rand_str,
    print,
)
from .load_config import MODEL_STRUCT, load_config
from .paths import CONFIGS_DIR, LOGS_DIR, ROOT_DIR, SRC_DIR, UTILS_DIR
from .seeding import (
    derive_hyper_candidate_seed,
    derive_module_seed,
    derive_simulation_point_seed,
    derive_trajectory_seed,
    inject_module_seed_from_trajectory,
    stable_seed,
)

__all__ = [
    "COMPOSITE_STYLES",
    "CONFIGS_DIR",
    "LOGGER",
    "LOGS_DIR",
    "MODEL_STRUCT",
    "PATHS",
    "PRINT_RESUME",
    "PRINT_STYLES",
    "ROOT_DIR",
    "SINGLE_STYLES",
    "SRC_DIR",
    "UTILS_DIR",
    "cdist",
    "compose_print",
    "configure_logging",
    "derive_hyper_candidate_seed",
    "derive_module_seed",
    "derive_simulation_point_seed",
    "derive_trajectory_seed",
    "entropy",
    "euc_dist",
    "gen_rand_str",
    "inject_module_seed_from_trajectory",
    "load_config",
    "print",
    "softmax",
    "stable_seed",
    "two_factor_decay",
]
