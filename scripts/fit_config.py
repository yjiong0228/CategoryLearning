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

memory_params_M2 = {
    1: {'gamma': np.float64(0.5499999999999999), 'w0': 0.15},
    2: {'gamma': np.float64(0.7999999999999999), 'w0': 0.0075},
    3: {'gamma': np.float64(0.05), 'w0': 0.049999999999999996},
    4: {'gamma': np.float64(0.9), 'w0': 0.15},
    5: {'gamma': np.float64(0.65), 'w0': 0.012499999999999999},
    6: {'gamma': np.float64(0.7), 'w0': 0.010714285714285714},
    7: {'gamma': np.float64(0.05), 'w0': 0.049999999999999996},
    8: {'gamma': np.float64(0.39999999999999997), 'w0': 0.024999999999999998},
    9: {'gamma': np.float64(0.6), 'w0': 0.15},
    10: {'gamma': np.float64(0.05), 'w0': 0.049999999999999996},
    11: {'gamma': np.float64(0.7), 'w0': 0.01},
    12: {'gamma': np.float64(0.75), 'w0': 0.0075},
    13: {'gamma': np.float64(0.1), 'w0': 0.03},
    14: {'gamma': np.float64(0.6), 'w0': 0.0075},
    15: {'gamma': np.float64(0.75), 'w0': 0.03},
    16: {'gamma': np.float64(0.1), 'w0': 0.01875},
    17: {'gamma': np.float64(0.7), 'w0': 0.013636363636363636},
    18: {'gamma': np.float64(0.5499999999999999), 'w0': 0.075},
    19: {'gamma': np.float64(0.05), 'w0': 0.049999999999999996},
    20: {'gamma': np.float64(0.05), 'w0': 0.0075},
    21: {'gamma': np.float64(0.49999999999999994), 'w0': 0.02142857142857143},
    22: {'gamma': np.float64(0.05), 'w0': 0.02142857142857143},
    23: {'gamma': np.float64(0.75), 'w0': 0.013636363636363636},
    24: {'gamma': np.float64(0.7), 'w0': 0.15},
}

sub_cond1 = [i for i in range(1, 25) if i % 3 == 1]
sub_cond2 = [i for i in range(1, 25) if i % 3 == 2]
sub_cond3 = [i for i in range(1, 25) if i % 3 == 0]


module_configs = {}

# Add configurations for sub_cond1
module_configs.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[i]['gamma'], memory_params_M2[i]['gamma']),
                "w0": (memory_params_M2[i]['w0'], memory_params_M2[i]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    } for i in sub_cond1
})

# Add configurations for sub_cond2
module_configs.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("max_2", "random_posterior"),
                                ("random_9", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.8 else 0, 1), "top_posterior"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[i]['gamma'], memory_params_M2[i]['gamma']),
                "w0": (memory_params_M2[i]['w0'], memory_params_M2[i]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    } for i in sub_cond2
})

# Add configurations for sub_cond3
module_configs.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("max_2", "random_posterior"),
                                ("random_6", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.8 else 0, 1), "top_posterior"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[i]['gamma'], memory_params_M2[i]['gamma']),
                "w0": (memory_params_M2[i]['w0'], memory_params_M2[i]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    } for i in sub_cond3
})

module_configs[2] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [("max_2", "random_posterior"),
                                ("random_9", "random_posterior"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.85 else 0, 1), "top_posterior"),
                                (2, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 5), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[23]['gamma'], memory_params_M2[23]['gamma']),
                "w0": (memory_params_M2[23]['w0'], memory_params_M2[23]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

module_configs[3] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [(PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.82 else 0, 1), "random_posterior"),
                                (5, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[3]['gamma'], memory_params_M2[3]['gamma']),
                "w0": (memory_params_M2[3]['w0'], memory_params_M2[3]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

module_configs[5] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_5", "random_posterior"),
                                (3, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.85 else 0, 1), "top_posterior"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 2), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[5]['gamma'], memory_params_M2[5]['gamma']),
                "w0": (memory_params_M2[5]['w0'], memory_params_M2[5]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

module_configs[7] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.85 else 0, 1), "top_posterior")],
            "init_strategy": [(3, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[3]['gamma'], memory_params_M2[3]['gamma']),
                "w0": (memory_params_M2[3]['w0'], memory_params_M2[3]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

module_configs[8] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_5", "random_posterior"),
                                (3, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.85 else 0, 1), "top_posterior"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 2), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[8]['gamma'], memory_params_M2[8]['gamma']),
                "w0": (memory_params_M2[8]['w0'], memory_params_M2[8]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }


module_configs[12] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_3", "random_posterior"),
                                (3, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.85 else 0, 1), "top_posterior"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 5), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[12]['gamma'], memory_params_M2[12]['gamma']),
                "w0": (memory_params_M2[12]['w0'], memory_params_M2[12]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

def random_and_acc_amount_f(x, f):
    if x>0.7:
        return f(2)
    else:
        return f(4)
module_configs[17] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [(PartitionCluster._amount_accuracy_gen(lambda x: random_and_acc_amount_f(x, PartitionCluster._amount_max_gen), 4), "random_posterior"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: random_and_acc_amount_f(x, PartitionCluster._amount_random_gen), 4), "random_posterior"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x<0.6 else 0, 4), "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 5), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[17]['gamma'], memory_params_M2[17]['gamma']),
                "w0": (memory_params_M2[17]['w0'], memory_params_M2[17]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

module_configs[23] = {
        "cluster": (PartitionCluster, {
            "transition_spec": [("max_2", "random_posterior"),
                                (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0.76 else 0, 1), "top_posterior"),
                                (5, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
            "init_strategy": [(25, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (memory_params_M2[23]['gamma'], memory_params_M2[23]['gamma']),
                "w0": (memory_params_M2[23]['w0'], memory_params_M2[23]['w0'])
            },
            "param_resolution": 1
        }),
        "perception": (BasePerception, {}),
    }

# module_configs = {
#     1: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (memory_params_M2, memory_params_M2),
#                 "w0": (memory_params_M2, memory_params_M2)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     2: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.8, 0.8),
#                 "w0": (0.0075, 0.0075)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     3: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 0.05),
#                 "w0": (0.05, 0.05)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     4: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.9, 0.9),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     5: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.65, 0.65),
#                 "w0": (0.0125, 0.0125)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     6: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.01, 0.01)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     7: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {}),
#     },

#     8: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.0136, 0.0136)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     9: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     10: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {}),
#     },

#     11: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.0136, 0.0136)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     12: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     13: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {}),
#     },

#     14: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.0136, 0.0136)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     15: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     16: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(10, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {}),
#     },

#     17: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.0136, 0.0136)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     18: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     19: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {}),
#     },

#     20: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.0136, 0.0136)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     21: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     22: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("random_4", "top_posterior"),
#                                 ("opp_random_4", "random")],
#             "init_strategy": [(3, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {}),
#     },

#     23: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_9", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.0136, 0.0136)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },

#     24: {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [("max_2", "random_posterior"),
#                                 ("random_6", "random_posterior"),
#                                 (1, "ksimilar_centers"),
#                                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 3), "random")],
#             "init_strategy": [(25, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.7, 0.7),
#                 "w0": (0.15, 0.15)
#             },
#             "param_resolution": 1
#         }),
#         "perception": (BasePerception, {}),
#     },
# }

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

grid_repeat_configs = {
    1: 5,
    4: 3,
}

mc_sample_configs = {
    1: 10,
    4: 5,
}
