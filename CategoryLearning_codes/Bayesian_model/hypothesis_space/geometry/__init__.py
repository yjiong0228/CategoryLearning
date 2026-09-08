"""Category geometry over shared continuous or discrete spaces."""

from .boundary import BoundaryGeometry, BoundaryProjectionError
from .dykstra_acceleration import (
    dykstra_numba_available,
    warmup_dykstra_numba,
)
from .discrete_rule import DiscreteRuleGeometry
from .prototype import PrototypeGeometry

__all__ = [
    "BoundaryGeometry",
    "BoundaryProjectionError",
    "dykstra_numba_available",
    "DiscreteRuleGeometry",
    "PrototypeGeometry",
    "warmup_dykstra_numba",
]
