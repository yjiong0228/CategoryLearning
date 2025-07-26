"""
Bayesian Engine
"""
from typing import Dict, Tuple, List, Any, Literal
import numpy as np
from ..utils import softmax

EPS = 1e-15


class BaseSet:
    """Immutable"""

    def __init__(self, elements: Dict | Tuple | List):
        """init"""
        self.elements = tuple(elements)
        self._size = len(elements)
        self.inv = dict((elt, i) for i, elt in enumerate(elements))

    @property
    def length(self):
        """
        Property length
        """
        return self._size

    def __getitem__(self, key):
        assert key in self.inv, f"Invalid key: {key} in {self.inv}"
        return self.inv.get(key, -1)

    def __repr__(self):
        return f"Base Set of (index, value)'s:\n{self.elements}"

    def __iter__(self):
        return iter(self.elements)


class BaseDistribution:
    """
    Base Distribution
    """

    def __init__(self, space: BaseSet):
        """Initialize"""
        self.value = np.ones(space.length, dtype=float) / space.length

    def update(self, value: np.ndarray):
        """
        Update value
        """
        self.value = value.copy()

    def get_value(self):
        """
        Get value
        """
        return self.value


class BasePrior(BaseDistribution):
    """
    Base Prior
    """

    @property
    def prior(self):
        """
        Get Prior
        """
        return self.value


class BaseLikelihood(BaseDistribution):
    """
    Base Class of Likelihood
    """

    def __init__(self, space: BaseSet, **kwargs):
        """
        Parameters
        ----------
        h_set
        d_set
        """
        super().__init__(space)
        self.h_set = space  # hypotheses set!
        # self.d_set = d_set
        self.kwargs = kwargs
        self.cache = {None: self.value}

    def get_likelihood(self, observation, **kwargs):
        """
        Parameters
        ----------
        observation: any
        """
        # print(observation)
        # if observation in self.cache:
        #     return self.cache[observation]

        raise NotImplementedError("get_likelihood not implemented")
        # return None  # 1. Calculate 2. Save 3. Return.

    def set_all(self, value: Dict[Any, np.ndarray]):
        """
        Set the whole matrix
        """
        self.cache.update(value)

    def set_row(self, row, value):
        """
        Set Row
        """
        self.cache[row] = value


class BaseEngine:
    """
    Base Bayesian Engine
    """

    def __init__(self, hypotheses_set: BaseSet, observation_set: BaseSet,
                 prior: BasePrior, likelihood: BaseLikelihood):
        """

        """
        self.hypotheses_set = hypotheses_set
        self.observation_set = observation_set
        self.prior = prior
        self.likelihood = likelihood
        self.h_state = self.prior

    def generate_drift(self,
                       base: np.ndarray,
                       mode: Literal["add", "add&norm", "scale&norm"] = "add",
                       **kwargs) -> np.ndarray:
        """
        Generate drift on Bayesian posterior. As another mechanism of
        limited (imperfect) memory.
        """
        step_size = kwargs.get("step_size", 0.02)
        generator = kwargs.get("generator", np.random.normal)

        delta = generator(
            kwargs.get(
                "generator_kwargs", {
                    "loc": 1. if mode == "scale&norm" else 0.,
                    "scale": step_size,
                    "size": self.hypotheses_set.length()
                }))

        match mode:
            case "add":
                ret = base + delta
            case "add&norm":
                ret = base + delta
                ret /= np.sum(ret)
            case "scale&norm":
                ret = base * delta
                ret /= np.sum(ret)

        return ret

    def infer_single(self, observation, **kwargs) -> float:
        """
        Parameters
        ----------
        observation:
        """

        likelihood_row = self.likelihood.get_likelihood(observation, **kwargs)
        self.h_state.update(self.h_state.value * likelihood_row)
        self.h_state.update(self.h_state.value / self.h_state.value.sum())

        if "drift" in kwargs:
            value = self.generate_drift(self.h_state.value, **kwargs)
            self.h_state.update(value)

        return self.h_state.value

    def infer(self, observations: list | tuple, **kwargs) -> float:
        """
        Parameters
        ----------
        observations: List | Tuple of observations
        """
        for obs in observations:
            self.infer_single(obs, **kwargs)

        return self.h_state.value

    def infer_log(self,
                  observations,
                  update: bool = False,
                  **kwargs) -> np.ndarray:
        """
        Log version of inference
        """
        likelihood = self.likelihood.get_likelihood(observations, **kwargs)
        log_likelihood = np.log(np.maximum(likelihood, EPS))
        log_prior = np.log(np.maximum(self.h_state.value, EPS))

        log_posterior = log_prior + (np.sum(log_likelihood, axis=0) if len(
            log_likelihood.shape) == 2 else log_likelihood)
        posterior = softmax(log_posterior, beta=1.)
        if update:
            self.h_state.update(posterior)
        return posterior
