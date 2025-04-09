import os
import sys

from pathlib import Path
import pandas as pd
import numpy as np

from tqdm import tqdm

from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt

# 设定项目根目录
project_root = Path(os.getcwd()).parent
sys.path.append(str(project_root))

# 导入数据
processed_path = Path(project_root) / 'data' / 'processed'
learning_data = pd.read_csv(processed_path / 'Task2_processed.csv')
# 导入模型
from src.Bayesian_new import *

import src.Bayesian_new.problems.model_v1 as model

from src.Bayesian_new.problems.model_v1 import SingleRationalModel

import src.Bayesian_new.problems.config as config

from src.Bayesian_new.problems.config import config_base
from src.Bayesian_new.problems import hypo_transitions as ht


def main():
    p = ht.PartitionCluster(4,
                            4,
                            transition_spec=[(3, "top_posterior"),
                                             (5, "ksimilar_centers"),
                                             (3, "random")])
    print(p.cluster_transition_strategy)
    ref = np.random.dirichlet([0.1] * 10)
    posterior = {i: ref[i] for i in range(10)}
    x = np.argmax(ref)

    center = p.centers[x][1][2]

    p.cluster_transition(posterior=posterior,
                         stimulus=center,
                         proto_hypo_amount=3)


if __name__ == '__main__':
    main()
