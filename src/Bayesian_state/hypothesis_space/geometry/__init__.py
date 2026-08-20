"""Category geometry over shared continuous or discrete spaces."""

from .boundary import BoundaryGeometry, BoundaryProjectionError
from .discrete_rule import DiscreteRuleGeometry
from .prototype import PrototypeGeometry

__all__ = [
    "BoundaryGeometry",
    "BoundaryProjectionError",
    "DiscreteRuleGeometry",
    "PrototypeGeometry",
]
