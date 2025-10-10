"""
Bayesian Engine
"""
from typing import Dict, Tuple, List, Any, Literal
import numpy as np
from ..utils import softmax, print, LOGGER

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


# the new bayesian engine
class BaseEngine:
    """
    A flexible Bayesian engine that can be dynamically configured with various
    computational modules.
    """
    upper_numerical_bound = 1e15
    lower_numerical_bound = 1e-15

    def __init__(self,
                 module_configs: dict | None = None,
                 agenda: List[str] | None = None,
                 **kwargs):
        """
        Initializes the engine.

        Parameters
        ----------
        module_configs : dict, optional
            A dictionary defining the modules to be built and registered.
            If provided, modules are built upon initialization.
        """

        self.hypotheses_set = None
        self.prior = None
        self.posterior = None
        self.likelihood = None
        self.h_state = None
        self.observation = None  # 记录当前观测
        
        # 模块列表，按顺序更新
        self.agenda = agenda if agenda is not None and {} else ["__self__"]
        self.modules = {"__self__": self}
        
        self.log_posterior = None
        self.log_likelihood = None
        self.log_posterior = None

        # 在初始化时直接构建模块
        self.build_modules(module_configs)

    @staticmethod
    def translate_from_log(log: np.ndarray) -> np.ndarray:
        log -= np.max(log)
        exp = np.exp(log)
        return exp / np.sum(exp)

    @staticmethod
    def translate_to_log(exp: np.ndarray) -> np.ndarray:
        return np.log(
            np.max([np.min([exp, BaseEngine.upper_numerical_bound]),
                   BaseEngine.lower_numerical_bound]))

    def build_modules(self, module_configs: Dict[str, Dict]):
        """
        Dynamically builds and registers modules from a configuration dictionary.

        Each module is instantiated and becomes an attribute of the engine.
        The engine instance itself is passed to each module's constructor,
        allowing modules to access the engine's state.

        Parameters
        ----------
        module_configs : Dict[str, Dict]
            Configuration for the modules. **Attention: Order is very important**
            Example:
            {
                'perception': {
                    'class': PerceptionModule,
                    'params': {'noise': 0.1}
                },
                'memory': {
                    'class': MemoryModule,
                    'params': {'decay_rate': 0.95}
                }
            }
        """

        for name, config in module_configs.items():

            module_class = config['class']
            module_params = config.get('params', {})

            print(
                f"  - Instantiating module '{name}' of type {module_class.__name__}..."
            )
            module_instance = module_class(engine=self, **module_params)

            # 将实例化的模块注册为 engine 的一个成员 (属性)
            setattr(self, name, module_instance)
            LOGGER.info(f"  - Module '{name}' registered as 'self.{name}'.")
            # 添加到模块列表
            self.modules[name](module_instance)
        print("All modules built successfully.")

    def update_all_modules(self, **kwargs):
        """
        Calls the 'update' method on all registered modules in sequence.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments to pass to each module's update method.
        """

        for module in self.modules:
            # 约定：每个模块都应该有一个 update 方法
            if hasattr(module, 'update'):
                module.update(**kwargs)
            else:
                print(
                    f"Warning: Module {module.__class__.__name__} has no 'update' method and was skipped."
                )

    def infer_single(self, observation, mod_kwargs: Dict | None = None):
        """
        Infer single new data.
        """
        self.observation = observation
        if self.agenda is None:
            raise Exception("inference agenda is not defined.")
        valid_modules = [x not in self.modules for x in self.agenda]
        if any(valid_modules):
            raise Exception(
                "unknown agenda items:",
                [self.agenda[i] for i, x in enumerate(valid_modules) if x])

        for mod_name in self.agenda:
            self.modules[mod_name].process(**mod_kwargs.get(mod_name, {}))

    def process(self, **kwargs):
        """
        A standard Bayesian Inference
        """
        self.log_prior = self.translate_to_log(self.prior)
        self.log_likelihood = self.translate_to_log(self.likelihood)
        self.log_posterior = self.log_prior + self.log_likelihood
        self.posterior = self.translate_from_log(self.log_posterior)
        return self.posterior

    # ========================================================================================== #

    def infer_single_old(self, observation, **kwargs) -> float:

        self.observation = observation  # 更新当前观测
        # 按顺序调用所有模块的 update 方法
        self.update_all_modules(**kwargs)

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

    # 对state先遗忘衰减，再进行贝叶斯更新。
    def infer_log_state(self,
                        observation,
                        apply_trans=None,
                        trans_kwargs=None,
                        update: bool = True,
                        **kwargs) -> np.ndarray:
        """
        贝叶斯log更新，支持对h_state先进行处理，再进行贝叶斯更新

        Parameters
        ----------
        observation: 当前观测
        apply_trans: 可选，对h_state的转换函数，形如 f(h_state, **trans_kwargs)
        trans_kwargs: dict, 传递给apply_trans的参数
        update: 是否更新h_state
        kwargs: 传递给likelihood的参数
        """

        # 可选：对prior进行转换
        if apply_trans is not None:
            trans_kwargs = trans_kwargs or {}
            prior = apply_trans(prior, **trans_kwargs)
            prior = prior / prior.sum()
        # log Bayesian update
        likelihood = self.likelihood.get_likelihood(observation, **kwargs)
        log_likelihood = np.log(np.maximum(likelihood, EPS))
        log_prior = np.log(np.maximum(self.h_state.value, EPS))
        log_posterior = log_prior + (np.sum(log_likelihood, axis=0) if len(
            log_likelihood.shape) == 2 else log_likelihood)
        posterior = softmax(log_posterior, beta=1.)
        if update:
            self.h_state.update(posterior)
        return posterior
