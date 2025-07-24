from src.Bayesian import *
from src.Bayesian.problems.config import config_fgt
from src.Bayesian.problems import *
from src.Bayesian.utils.optimizer import Optimizer



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


sub_cond1 = [i for i in range(1, 25) if i % 3 == 1]
sub_cond2 = [i for i in range(1, 25) if i % 3 == 2]
sub_cond3 = [i for i in range(1, 25) if i % 3 == 0]


module_configs_M3 = {}
module_configs_M3.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
    } for i in sub_cond1 
})

module_configs_M3.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
    } for i in (sub_cond2 + sub_cond3)
})



module_configs_M5 = {}

module_configs_M5.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "perception": (BasePerception, {}),
    } for i in sub_cond1
})

module_configs_M5.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random")],
            "init_strategy": [(10, "random")]}),
        "perception": (BasePerception, {}),
    } for i in (sub_cond2 + sub_cond3)
})


module_configs_M6 = {}

module_configs_M6.update({
    i: {
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
        }),
        "perception": (BasePerception, {}),
    } for i in sub_cond1
})

module_configs_M6.update({
    i: {
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
        }),
        "perception": (BasePerception, {}),
    } for i in (sub_cond2 + sub_cond3)
})


module_configs_M7 = {}

module_configs_M7.update({
    i: {
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
        }),
        "perception": (BasePerception, {}),
    } for i in sub_cond1
})

module_configs_M7.update({
    i: {
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
        }),
        "perception": (BasePerception, {}),
    } for i in (sub_cond2 + sub_cond3)
})


# module_configs[4] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 5 if x<=0.8 else 0, 4, True), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>=0.9 else 0, 4, True), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x<=0 else PartitionCluster._amount_random_gen(4), 4, False), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 8 if x<0 else 0, 8, False), "random"),
#                                 ],
#             "init_strategy": [(10, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[22] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x>=0 else 0, 4, False), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>=0.9 else 0, 2, True), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 5 if x<0.8 else 0, 5, True), "random"),],
#             "init_strategy": [(4, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[2] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>0.8 else 0, 2, True), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x<0 else PartitionCluster._amount_random_gen(7), 7, False), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 1 if x>0 else 0, 1, False), "ksimilar_centers"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x<=0 else 0, 4, False), "random"),
#                 (8, "random")],
#             "init_strategy": [(10, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[20] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>0.8 else 0, 2, True), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x<0 else PartitionCluster._amount_random_gen(7), 7, False), "random_posterior"),
#                 (1, "ksimilar_centers"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x<=0 else 0, 4, False), "random"),
#                 (8, "random")],
#             "init_strategy": [(10, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[23] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>0.8 else 0, 2, True), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x<=0 else PartitionCluster._amount_random_gen(9), 9, False), "random_posterior"),
#                 (1, "ksimilar_centers"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x<=0 else 0, 4, False), "random"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 8 if x<0.8 else 0, 2, True), "random")],
#             "init_strategy": [(10, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[3] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>=0.86 else 0, 2, True), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x<=0 else PartitionCluster._amount_random_gen(9), 9, False), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x>0.4 and x<0.86 else 0, 4, True), "ksimilar_centers"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x<=0 else 0, 4, False), "random"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 8 if x<=0.4 else 0, 2, True), "random")],
#             "init_strategy": [(4, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[9] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x>=0.65 else 0, 4, True), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>0 else 0, 2, False), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x<=0.65 and x>0.5 else PartitionCluster._amount_random_gen(9), 9, True), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 8 if x>0.5 and x<=0.65 else 0, 8, True), "ksimilar_centers"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 4 if x<=0 else 0, 4, False), "random"),],
#             "init_strategy": [(4, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }

# module_configs[15] = {
#         "cluster": (PartitionCluster, {
#             "transition_spec": [
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 2 if x>=0.8 else 0, 2, True), "top_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 0 if x>=0.8 else PartitionCluster._amount_random_gen(5), 5, True), "random_posterior"),
#                 (PartitionCluster._amount_accuracy_gen(lambda x: 5 if x<=0.8 else 0, 5, True), "ksimilar_centers"),
#                 (PartitionCluster._amount_accuracy_gen(random_acc_amount_f, 7), "random"),],
#             "init_strategy": [(4, "random")]}),
#         "memory": (BaseMemory, {
#             "personal_memory_range": {
#                 "gamma": (0.05, 1.0),
#                 "w0": (0.075, 0.15)
#             },
#             "param_resolution": 20
#         }),
#         "perception": (BasePerception, {})
# }
