from src.Bayesian import *
from src.Bayesian.problems.config import config_fgt
from src.Bayesian.problems import *
from src.Bayesian.utils.optimizer import Optimizer




subIDs = []
for condition in [1, 2, 3]:
    for last_two_digits in range(1, 50):
        subID = condition * 100 + last_two_digits
        subIDs.append(subID)


#### window_size config
window_size_configs = {}
for subID in subIDs:
    first_digit = subID // 100
    window_size_configs[subID] = 8 if first_digit == 1 else 16



#### module config for M3
module_configs_M3 = {}
module_configs_M3.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
    } for i in subIDs if i // 100 == 1
})
module_configs_M3.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                ("opp_acc_7", "random")],
            "init_strategy": [(10, "random")]}),
    } for i in subIDs if i // 100 in [2, 3]
})


#### module config for M5
module_configs_M5 = {}
module_configs_M5.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_4", "top_posterior"),
                                ("opp_random_4", "random")],
            "init_strategy": [(3, "random")]}),
        "perception": (BasePerception, {}),
    } for i in subIDs if i // 100 == 1
})
module_configs_M5.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                ("opp_acc_7", "random")],
            "init_strategy": [(10, "random")]}),
        "perception": (BasePerception, {}),
    } for i in subIDs if i // 100 in [2, 3]
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
    } for i in subIDs if i // 100 == 1
})
module_configs_M6.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                ("opp_acc_7", "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        }),
    } for i in subIDs if i // 100 in [2, 3]
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
    } for i in subIDs if i // 100 == 1
})
module_configs_M7.update({
    i: {
        "cluster": (PartitionCluster, {
            "transition_spec": [("random_7", "random_posterior"),
                                (1, "ksimilar_centers"),
                                ("opp_acc_7", "random")],
            "init_strategy": [(10, "random")]}),
        "memory": (BaseMemory, {
            "personal_memory_range": {
                "gamma": (0.05, 1.0),
                "w0": (0.075, 0.15)
            },
            "param_resolution": 20
        }),
        "perception": (BasePerception, {}),
    } for i in subIDs if i // 100 in [2, 3]
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
