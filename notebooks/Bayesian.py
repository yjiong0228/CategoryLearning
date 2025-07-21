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
from src.Bayesian import *
from src.Bayesian.problems.config import config_fgt
from src.Bayesian.problems import *
from src.Bayesian.utils.optimizer import Optimizer

# def post_acc_amount_f(x):
#     if x <= 0.2:
#         return 0
#     elif 0.2 < x < 0.3:
#         return 1
#     elif 0.3 <= x < 0.4:
#         return 2
#     elif 0.4 <= x < 0.5:
#         return 3
#     elif 0.5 <= x < 0.6:
#         return 4
#     elif 0.6 <= x < 0.7:
#         return 5
#     elif 0.7 <= x < 0.8:
#         return 6
#     elif 0.8 <= x <= 1:
#         return 7

# def random_acc_amount_f(x):
#     return 7 - post_acc_amount_f(x)

# M0_base model
module_configs = {}

# # M1_P model
# module_configs = {"perception": (BasePerception, {})}

# M2_M model
# module_configs = {"memory": (BaseMemory, {
#                     "personal_memory_range": {
#                         "gamma": (0.05, 1.0),
#                         "w0": (0.075, 0.15)
#                     },
#                     "param_resolution": 20
#                 })}

# # M3_H model
# module_configs = {"cluster": (PartitionCluster, {
#             "transition_spec": [("random_7", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
#             "init_strategy": [(10, "random")]})}

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
# module_configs = {"cluster": (PartitionCluster, {
#                             "transition_spec": [("random_7", "random_posterior"),
#                                                 (1, "ksimilar_centers"),
#                                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
#                             "init_strategy": [(10, "random")]}),
#             "perception": (BasePerception, {})}

# M6_MH model
# from .fit_config_M6 import module_configs, window_size_configs

# M7_PMH model
# from ..src.Bayesian.problems.fit_config import module_configs, window_size_configs


optimizer = Optimizer(module_configs, n_jobs=120)

processed_path = Path(project_root) / 'data' / 'processed'
optimizer.prepare_data(processed_path / 'Task2_processed.csv')


# M0_base model
res = optimizer.optimize_params_with_subs_parallel(
    config_fgt, list(range(1, 25)), 16, 1, 1)

# # M1_P model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 0, 100)

# # M2_M model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 1, 1)

# # M3_H model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 0, 1000)

# # M4_PM model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 3, 500)

# # M5_PH model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), 16, 0, 1000)

# # M6_MH model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, [1,4,7,10,13,16,19,22], window_size_configs, 5, 1000)

# # M7_PMH model
# res = optimizer.optimize_params_with_subs_parallel(
#     config_fgt, list(range(1, 25)), window_size_configs, 5, 1000)


# 保存拟合结果
result_path = Path(project_root) / 'results' / 'Model_results'
os.makedirs(result_path, exist_ok=True)

# optimizer.save_results(res, 'M_fgt_cl_per', result_path)
joblib.dump(res, result_path / 'M0_Base.joblib') 
