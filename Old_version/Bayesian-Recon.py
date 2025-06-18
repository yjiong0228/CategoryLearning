import os
from pathlib import Path
import joblib

import numexpr
numexpr.set_num_threads(64)

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# 设定项目根目录
project_root = Path(os.getcwd())
# sys.path.append(str(project_root))

# 导入模型
from src.Bayesian_recon import *
from src.Bayesian_recon.problems.config import config_fgt
from src.Bayesian_recon.problems import *
from src.Bayesian_recon.utils.optimizer import Optimizer

# 模型配置
def post_acc_amount_f(x):
    if x <= 0.2:
        return 0
    elif 0.2 < x < 0.3:
        return 1
    elif 0.3 <= x < 0.4:
        return 2
    elif 0.4 <= x < 0.5:
        return 3
    elif 0.5 <= x < 0.6:
        return 4
    elif 0.6 <= x < 0.7:
        return 5
    elif 0.7 <= x < 0.8:
        return 6
    elif 0.8 <= x <= 1:
        return 7

def random_acc_amount_f(x):
    return 7 - post_acc_amount_f(x)


post_setting = ("random_7", "random_posterior")
ksimilar_setting = (1, "ksimilar_centers")
random_setting = ("opp_random_7", "random")

module_config = {
    "cluster": (PartitionCluster, {
        "transition_spec": [post_setting,
                            ksimilar_setting,
                            random_setting]}),
    "memory": (BaseMemory, {
        "personal_memory_range": {
            "gamma": (0.05, 1.0),
            "w0": (0.075, 0.15)
        },
        "param_resolution": 20
    })
}
optimizer = Optimizer(module_config, n_jobs=120)

processed_path = Path(project_root) / 'data' / 'processed'
optimizer.prepare_data(processed_path / 'Task2_processed.csv')


res = optimizer.optimize_params_with_subs_parallel(
    config_fgt, list(range(1, 25)), 16, 5, 1000)
# list(range(1, 25)))


# 保存拟合结果
result_path = Path(project_root) / 'results' / 'Bayesian_recon'
os.makedirs(result_path, exist_ok=True)

joblib.dump(res, result_path / 'M_fgt_cl_rand7_randp_k1_rand7.joblib')
