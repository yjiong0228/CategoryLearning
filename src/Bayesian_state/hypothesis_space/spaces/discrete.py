"""Define and build the discrete parity-rule hypothesis space."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterator

from .common import FrozenParameters


FAMILY_PARITY_RULE = "parity_rule"


@dataclass(frozen=True)
class DiscreteHypothesisSpec:
    """One fixed-label monomial/parity rule over binary features."""

    dims: tuple[int, ...]
    polarity: int
    index: int = field(default=-1, compare=False)
    family: str = field(default=FAMILY_PARITY_RULE, init=False, compare=False)

    def __post_init__(self) -> None:
        dims = tuple(int(dimension) for dimension in self.dims)
        polarity = int(self.polarity)
        if polarity not in (-1, 1):
            raise ValueError(f"polarity must be -1 or 1, got {self.polarity!r}.")
        if len(set(dims)) != len(dims) or tuple(sorted(dims)) != dims:
            raise ValueError("dims must be unique and sorted.")
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "polarity", polarity)
        object.__setattr__(self, "index", int(self.index))

    @property
    def label(self) -> str:
        if not self.dims:
            base = "Intercept"
        else:
            base = "*".join(f"X{dimension + 1}" for dimension in self.dims)
        return base if self.polarity > 0 else f"-{base}"


@dataclass(frozen=True)
class DiscreteHypothesisSpace:
    """Ordered space consumed by exact discrete-rule geometry."""

    n_dims: int
    n_cats: int
    hypotheses: tuple[DiscreteHypothesisSpec, ...]
    include_intercept: bool
    version: str = "discrete_parity_fixed_labels_v1"

    def __post_init__(self) -> None:
        n_dims = int(self.n_dims)
        n_cats = int(self.n_cats)
        hypotheses = tuple(self.hypotheses)
        if n_dims < 1:
            raise ValueError(f"n_dims must be positive, got {n_dims}.")
        if n_cats != 2:
            raise ValueError("DiscreteHypothesisSpace supports binary categories only.")
        if tuple(item.index for item in hypotheses) != tuple(range(len(hypotheses))):
            raise ValueError("Hypothesis indices must be contiguous and match list order.")
        for hypothesis in hypotheses:
            if any(dimension < 0 or dimension >= n_dims for dimension in hypothesis.dims):
                raise ValueError(
                    f"Hypothesis {hypothesis.index} contains a dimension outside "
                    f"[0, {n_dims})."
                )
        object.__setattr__(self, "n_dims", n_dims)
        object.__setattr__(self, "n_cats", n_cats)
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "include_intercept", bool(self.include_intercept))

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self) -> Iterator[DiscreteHypothesisSpec]:
        return iter(self.hypotheses)

    def __getitem__(self, index: int) -> DiscreteHypothesisSpec:
        return self.hypotheses[index]

    @property
    def parameters(self) -> FrozenParameters:
        return FrozenParameters.from_mapping(
            {"include_intercept": self.include_intercept}
        )

    @property
    def signature(self) -> tuple:
        return (
            self.version,
            self.n_dims,
            self.n_cats,
            self.include_intercept,
        )


def build_discrete_hypothesis_space(
    n_dims: int = 5,
    n_cats: int = 2,
    *,
    include_intercept: bool = False,
) -> DiscreteHypothesisSpace:
    """Build the historical parity-rule inventory in its original order."""
    n_dims = int(n_dims)
    n_cats = int(n_cats)
    if n_dims < 1:
        raise ValueError(f"n_dims must be positive, got {n_dims}.")
    if n_cats != 2:
        raise ValueError("Discrete parity rules currently support binary categories only.")

    hypotheses: list[DiscreteHypothesisSpec] = []
    start_size = 0 if bool(include_intercept) else 1
    for size in range(start_size, n_dims + 1):
        for dims in combinations(range(n_dims), size):
            for polarity in (1, -1):
                hypotheses.append(
                    DiscreteHypothesisSpec(
                        dims=tuple(dims),
                        polarity=polarity,
                        index=len(hypotheses),
                    )
                )
    return DiscreteHypothesisSpace(
        n_dims=n_dims,
        n_cats=n_cats,
        hypotheses=tuple(hypotheses),
        include_intercept=bool(include_intercept),
    )


__all__ = [
    "DiscreteHypothesisSpace",
    "DiscreteHypothesisSpec",
    "FAMILY_PARITY_RULE",
    "build_discrete_hypothesis_space",
]
