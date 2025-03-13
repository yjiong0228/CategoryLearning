"""
Test Forget Model
"""
import sys
import os
from pathlib import Path
import importlib

sys.path.append("..")
import numpy as np

import pandas as pd
import src.Bayesian_new.problems.forget as forget

importlib.reload(forget)
from src.Bayesian_new.problems.forget import ForgetModel

import src.Bayesian_new.problems.config as config

importlib.reload(config)
from src.Bayesian_new.problems.config import config_fgt
from src.Bayesian.utils.partition import Partition

import src.Bayesian.M_fgt_grid as model_forget

importlib.reload(model_forget)
from src.Bayesian.M_fgt_grid import M_Fgt, ModelParams

import src.Bayesian.config as config

importlib.reload(config)
from src.Bayesian.config import config_fgt

model_forget = M_Fgt(config_fgt)

partition = Partition()
all_centers = {
    '2_cats': partition.get_centers(4, 2),
    '4_cats': partition.get_centers(4, 4)
}
model_forget.set_centers(all_centers)

# 设定项目根目录
project_root = Path(os.getcwd()).parent
sys.path.append(str(project_root))
# 导入数据
processed_path = Path(project_root) / 'data' / 'processed'
learning_data = pd.read_csv(processed_path / 'Task2_processed.csv')


def main(args):
    """

    """
    new_model = ForgetModel(config_fgt)
    old_model = model_forget

    for i, (iSub, data_old) in enumerate(learning_data.groupby('iSub')):
        if i > 0:
            continue
        print(data_old)
        data_new = (data_old[["feature1", "feature2", "feature3",
                              "feature4"]].values, data_old["choice"].values,
                    data_old["feedback"].values, data_old["category"].values)
        result_old = old_model.fit_trial_by_trial(data_old, gamma=0.8, w0=0.2)
        result_new = new_model.fit_trial_by_trial(data_new, gamma=0.8, w0=0.2)

    return result_old, result_new


def cmp_log_likelihood():
    new_model = ForgetModel(config_fgt)
    old_model = model_forget

    for i, (iSub, data_old) in enumerate(learning_data.groupby('iSub')):
        if i > 0:
            continue
        print(data_old)
        data_new = (data_old[["feature1", "feature2", "feature3",
                              "feature4"]].values, data_old["choice"].values,
                    data_old["feedback"].values, data_old["category"].values)

        result_old = old_model.posterior(ModelParams(
            1, 1.0, 0.8, 0.2), data_old, 1) + np.log(
                old_model.prior(ModelParams(1, 1.0, 0.8, 0.2), 1))
        result_new = new_model.get_weighted_log_likelihood(
            0, data_new, 1.0, 0.8, 0.2)

    return result_old, result_new


if __name__ == '__main__':
    result = cmp_log_likelihood()
    print(result)
