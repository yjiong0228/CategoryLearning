from src.Bayesian_recon import *
from src.Bayesian_recon.problems.config import config_fgt
from src.Bayesian_recon.problems import *
from src.Bayesian_recon.utils.optimizer import Optimizer

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



module_configs = {
    1: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_2", "top_posterior"),
                                ("opp_random_2", "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },
    4: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_2", "top_posterior"),
                                ("opp_random_2", "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        }),
        "perception": (BasePerception, {}),
    },
        13: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_2", "top_posterior"),
                                ("opp_random_2", "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        }),
        "perception": (BasePerception, {}),
    },
}

window_size_configs = {
    1: 8,
    4: 8,
    13:8
}

grid_repeat_configs = {
    1: 5,
    4: 3,
}

mc_sample_configs = {
    1: 10,
    4: 5,
}