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

# M0_base model
module_configs = {}

# # M1_P model
# module_configs = {"perception": (BasePerception, {})}

# # M2_M model
# module_configs = {"memory": (BaseMemory, {
#                     "personal_memory_range": {
#                         "gamma": (0.05, 1.0),
#                         "w0": (0.075, 0.15)
#                     },
#                     "param_resolution": 20
#                 })}

# # M3_H model


# # M4_PM model
# module_configs = {
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})}

# M5_PH model


# M6_MH model


# M7_PMH model
# from .fit_config import module_configs, window_size_configs, grid_repeat_configs, mc_sample_configs


optimizer = Optimizer(module_configs, n_jobs=120)

processed_path = Path(project_root) / 'data' / 'processed'
optimizer.prepare_data(processed_path / 'Task2_processed.csv')


# M0_base model
res = optimizer.optimize_params_with_subs_parallel(
    config_fgt, list(range(1, 25)), 16, 1, 1)

# # M1_P model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 1, 1000)

# # M2_M model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 1, 1)

# # M3_H model


# # M4_PM model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 5, 1000)


# Full model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), window_size_configs, 5, 1000)



# 保存拟合结果
result_path = Path(project_root) / 'results' / 'Model_results'
os.makedirs(result_path, exist_ok=True)

# optimizer.save_results(res, 'M_fgt_cl_per', result_path)
joblib.dump(res, result_path / 'M0_base.joblib') 
