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


module_configs = {
    1: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    2: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    3: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
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
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    5: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    6: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    7: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    8: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    9: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    10: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    11: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    12: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    13: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    14: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    15: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    16: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    17: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    18: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    19: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    20: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    21: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    22: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    23: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },

    24: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        })
    },
}

window_size_configs = {
    1: 8,
    2: 16,
    3: 16,
    4: 8,
    5: 16,
    6: 16,
    7: 8,
    8: 16,
    9: 16,
    10: 8,
    11: 16,
    12: 16,
    13: 8,
    14: 16,
    15: 16,
    16: 8,
    17: 16,
    18: 16,
    19: 8,
    20: 16,
    21: 16,
    22: 8,
    23: 16,
    24: 16}

# grid_repeat_configs = {
#     1: 5,
#     4: 3,
# }

# mc_sample_configs = {
#     1: 10,
#     4: 5,
# }
