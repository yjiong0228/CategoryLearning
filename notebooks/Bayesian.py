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

result_path = Path(project_root) / 'results' / 'Model_results'
os.makedirs(result_path, exist_ok=True)


# 导入模型
from src.Bayesian import *
from src.Bayesian.problems.config import config_fgt
from src.Bayesian.problems import *
from src.Bayesian.utils.optimizer import Optimizer
from src.Bayesian.utils.model_evaluation import ModelEval


# model_name = 'M0_Base'
# model_name = 'M1_P'
# model_name = 'M2_M'
# model_name = 'M3_H'
model_name = 'M4_PM'
# model_name = 'M5_PH'
# model_name = 'M6_MH'
# model_name = 'M7_PMH'


from src.Bayesian.problems.fit_config import window_size_configs

if model_name == 'M0_Base':
    module_configs = {}

elif model_name == 'M1_P':
    module_configs = {"perception": (BasePerception, {})}

elif model_name == 'M2_M':
    module_configs = {"memory": (BaseMemory, {
                        "personal_memory_range": {
                            "gamma": (0.05, 1.0),
                            "w0": (0.075, 0.15)
                        },
                        "param_resolution": 20
                    })}
    
elif model_name == 'M3_H':
    from src.Bayesian.problems.fit_config import module_configs_M3
    module_configs = module_configs_M3

elif model_name == 'M4_PM':
    from src.Bayesian.problems.fit_config import module_configs_M4
    module_configs = module_configs_M4

elif model_name == 'M5_PH':
    module_configs = {
            "memory": (BaseMemory, {
                "personal_memory_range": {
                    "gamma": (0.05, 1.0),
                    "w0": (0.075, 0.15)
                },
                "param_resolution": 20
            }),
            "perception": (BasePerception, {})}

elif model_name == 'M6_MH':
    from src.Bayesian.problems.fit_config import module_configs_M6
    module_configs = module_configs_M6

elif model_name == 'M7_PMH':
    from src.Bayesian.problems.fit_config import module_configs_M7
    module_configs = module_configs_M7



optimizer = Optimizer(module_configs, n_jobs=120)
subsect_ids = list(range(1, 25)) 

processed_path = Path(project_root) / 'data' / 'processed'
optimizer.prepare_data(processed_path / 'Task2_processed.csv')



if model_name == 'M0_Base':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 1, 1)

elif model_name == 'M1_P':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 0, 500)
    
elif model_name == 'M2_M':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 1, 1)
    
elif model_name == 'M3_H':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 0, 1000)
    
elif model_name == 'M4_PM':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 3, 500)
    
elif model_name == 'M5_PH':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 0, 1000)
    
elif model_name == 'M6_MH': 
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 5, 1000)

elif model_name == 'M7_PMH':
    res = optimizer.optimize_params_with_subs_parallel(
        config_fgt, subsect_ids, window_size_configs, 5, 1000)
    
# 保存结果
# optimizer.save_results(res, model_name, result_path)
joblib.dump(res, result_path / f'{model_name}.joblib')



# 加载模型结果
# res = joblib.load(result_path / f'{model_name}.joblib')

# plot posterior probabilities
model_eval = ModelEval()
model_eval.plot_posterior_probabilities(
    res, save_path=result_path/f'{model_name}_post.png')

# get predictions 
optimizer.set_results(res)
prediction = optimizer.predict_with_subs_parallel(
    config_fgt, subsect_ids)
joblib.dump(prediction, result_path / f'{model_name}_predict.joblib')

# plot accuracy comparison
model_eval.plot_accuracy_comparison(prediction, save_path=result_path/f'{model_name}_acc.png')

# plot model k vs oral k
from src.Bayesian.utils.oral_process import Oral_to_coordinate
oral_to_coordinate = Oral_to_coordinate()

learning_data = pd.read_csv(processed_path / 'Task2_processed.csv')
oral_hypo_hits = oral_to_coordinate.get_oral_hypo_hits(learning_data)

model_eval.plot_k_oral_comparison(
    res, oral_hypo_hits,
    range(1,25), save_path=result_path/f'{model_name}_oral.png')


# plot gamma and w0 grids
if model_name in ['M2_M', 'M4_PM', 'M6_MH', 'M7_PMH']:
    model_eval.plot_error_grids(res, fname=['gamma','w0'], save_path=result_path/f'{model_name}_grid.png')

if model_name in ['M3_H', 'M5_PH', 'M6_MH', 'M7_PMH']:
    model_eval.plot_cluster_amount(res, window_size=16, save_path=result_path/f'{model_name}_amount.png')