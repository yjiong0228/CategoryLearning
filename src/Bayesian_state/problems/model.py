"""
Base Model
"""
from typing import List
from copy import deepcopy
from pathlib import Path
import numpy as np
from .base_problem import BaseSet, BaseEngine
from .partitions import Partition
from ..utils import MODEL_STRUCT
from ..utils.paths import PROCESSED_DATA_DIR



class StateModel:
    """
    State Model, initialize an engine
    Refreshes the engine step by step:
        StateModel ----[data]----> engine
        engine ----[posterior]----> StateModel
    """

    def __init__(self, engine_config, **kwargs):
        """
        """

        if engine_config is None:
            raise ValueError("engine_config must be a dict; got None. Check MODEL_STRUCT and model_choice settings.")
        # Initialize attributes
        self.engine_config = deepcopy(engine_config)
        self.all_centers = None
        self.data = None
        self.hypotheses_set = BaseSet([])
        self.observation_set = BaseSet([])

        self.condition = kwargs.get("condition", 1)
        self.subject_id = kwargs.pop("subject_id", None)
        processed_data_dir = kwargs.pop("processed_data_dir", None)
        self.dataset_paths = kwargs.pop("dataset_paths", None)
        if processed_data_dir is None:
            self.processed_data_dir = PROCESSED_DATA_DIR.resolve()
        else:
            self.processed_data_dir = Path(processed_data_dir).resolve()

        n_dims = 4
        self.n_cats = 2 if self.condition == 1 else 4

        # Initialize partition
        self.partition_model = kwargs.get(
            "partition", Partition(n_dims, self.n_cats))
        # Initialize hypotheses set (length = partition_model.length)
        self.hypotheses_set = kwargs.get(
            "space", BaseSet(list(range(self.partition_model.length))))

        # Merge module overrides provided via kwargs
        for key, value in kwargs.items():
            if key in self.engine_config.get("modules", {}):
                self.engine_config["modules"][key].update(value)

        # initialize engine
        self.engine = BaseEngine(
            self.engine_config["agenda"],
            hypotheses_set=self.hypotheses_set,
            partition=self.partition_model,
        )
        # expose shared context for modules (e.g., PerceptionModule auto loading)
        self.engine.subject_id = self.subject_id
        self.engine.processed_data_dir = self.processed_data_dir
        self.engine.dataset_paths = self.dataset_paths
        # build modules for the engine
        self.engine.build_modules(self.engine_config["modules"])


    def save(self, posterior_log, step_log):
        """
        保存结果
        """
        self.posterior_log = posterior_log
        self.step_log = step_log
        

    def fit_step_by_step(self, data: List | np.ndarray, **kwargs):
        """
        """
        
        # load module kwargs
        mod_kwargs = kwargs.get("module_kwargs", {})
        # fit step by step
        data = data or self.data
        step_log = []
        posterior_log = []
        prior_log = []
        for datum in data:
            posterior, log = self.engine.infer_single(datum, mod_kwargs)
            # DEBUG
            #print("Current observation:", self.engine.observation, s=2)
            step_log += [log]
            posterior_log += [posterior]
            prior_log += [log.get('prior')]

        self.save(posterior_log, step_log)
        return posterior_log, prior_log



