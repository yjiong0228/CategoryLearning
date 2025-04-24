import os
from pathlib import Path
import joblib
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

# 保存拟合结果
result_path = Path(project_root) / 'results' / 'Bayesian_recon'
os.makedirs(result_path, exist_ok=True)

for i in range(1, 11):
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt,
        [i]
        # list(range(1, 25)) 
    )

    joblib.dump(res, result_path / f'M_fgt_cl_ran_5_5_5_10_10_sub{i}.joblib')
    logger.info(f"Finished processing for sub {i}.")