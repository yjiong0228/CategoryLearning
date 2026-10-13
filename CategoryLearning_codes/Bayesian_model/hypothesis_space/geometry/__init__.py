"""Category geometry over shared continuous or discrete spaces."""

from .active_set_acceleration import (
    active_set_numba_available,
    warmup_active_set_numba,
)
from .boundary import BoundaryGeometry, BoundaryProjectionError
from .discrete_rule import DiscreteRuleGeometry
from .prototype import PrototypeGeometry

__all__ = [
    "BoundaryGeometry",
    "BoundaryProjectionError",
    "DiscreteRuleGeometry",
    "PrototypeGeometry",
    "active_set_numba_available",
    "warmup_active_set_numba",
]
