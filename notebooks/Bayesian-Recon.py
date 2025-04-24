import os
import sys
import importlib
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt


# 设定项目根目录
project_root = Path(os.getcwd())
# sys.path.append(str(project_root))

# 导入模型
from src.Bayesian_recon import *

import src.Bayesian_recon.problems.model as model
from src.Bayesian_recon.problems.model import StandardModel as Model

import src.Bayesian_recon.problems.config as config
from src.Bayesian_recon.problems.config import config_fgt

from src.Bayesian_recon.problems import *

from src.Bayesian_recon.utils.optimizer import Optimizer

module_config = {
    "cluster": (PartitionCluster, {
        "amount_range": [(0, 5), (0, 5), (0, 5)],
        "transition_spec": ["posterior_random", "ksimilar_centers", "random"]
    }),
    "memory": (BaseMemory, {
        "personal_memory_range": {"gamma": (0.1, 1.0), "w0": (0.01, 0.1)},
        "param_resolution": 10
    })
}
optimizer = Optimizer(module_config, n_jobs=100)

processed_path = Path(project_root) / 'data' / 'processed'
optimizer.prepare_data(processed_path / 'Task2_processed.csv')

res = optimizer.optimize_params_with_subs_parallel(
    config_fgt,
    # [1,4]
    list(range(1, 25)) 
)

# 保存拟合结果
result_path = Path(project_root) / 'results' / 'Bayesian_recon'
os.makedirs(result_path, exist_ok=True)

joblib.dump(res, result_path / 'M_fgt_cl_ran_5_5_5_10_10.joblib')