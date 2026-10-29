"""Exact perception of ordered black/white squares for the Exp5 baseline."""
import numpy as np

from .base_module import BaseModule, ModulePhase, ModuleRole


class ExactBinaryPerceptionModule(BaseModule):
    """Preserve -1/+1 inputs; this explicit assumption has no noise parameter."""

    phase = ModulePhase.PRE_CHOICE
    role = ModuleRole.PERCEPTION

    def __init__(self, engine, *, features: int = 5, module_seed: int | None = None):
        super().__init__(engine)
        if features != 5:
            raise ValueError("Exp5 exact perception requires five ordered squares")
        self.features = features
        # The shared particle interface assigns every module an RNG seed.
        # This stateless deterministic module does not consume a random stream.

    def process(self, **kwargs):
        values = np.asarray(self.engine.observation[0], dtype=float)
        if values.shape != (self.features,) or not np.isin(values, [-1., 1.]).all():
            raise ValueError("Exp5 perception requires five finite -1/+1 values")
        self.engine.observation = (values.copy(), *self.engine.observation[1:])
        return values.copy()
