config_base = {
    'param_bounds': {
        'beta': (0.001, 30)
    },
    'param_inits': {
        'beta': 1
    }
}

config_fgt = {
    'param_bounds': {
        'beta': (0.001, 30),
        'gamma': (0.001, 1)
    },
    'param_inits': {
        'beta': 1,
        'gamma': 0.7
    }
}

config_dec = {
    'param_bounds': {
        'beta': (0.001, 30),
        'phi': (0.001, 1)
    },
    'param_inits': {
        'beta': 1,
        'phi': 0.5
    }
}

