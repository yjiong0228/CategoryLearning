"""Category geometry over shared continuous or discrete spaces."""

from .boundary import BoundaryGeometry
from .discrete_rule import DiscreteRuleGeometry
from .prototype import PrototypeGeometry

__all__ = ["BoundaryGeometry", "DiscreteRuleGeometry", "PrototypeGeometry"]
