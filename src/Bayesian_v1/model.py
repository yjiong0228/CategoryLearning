import numpy as np
from dataclasses import make_dataclass
from collections import OrderedDict
from scipy.optimize import minimize

from .modules import Module, Base, Decision
from .utils.data import CategoryDataset, TrialData
from .utils.partition import Partition

from typing import Dict, Tuple, List, Callable

# Define all modules, each module is a tuple of (module, priority)
all_modules = {
    'base': (Base(), 0),
    'decision': (Decision(), 1),
}
all_modules = dict(sorted(all_modules.items(), key=lambda x: x[1][1]))
    
from .config import config



class BayesianModel:
    def __init__(self, iSub: int, modules: List[str] = ['base']):
        self.config = config
        self.reload(iSub, modules)

    def reload(self, iSub: int, modules: List[str] = ['base']):
        self.all_data = CategoryDataset.load_data_from_file().get_subject(iSub)
        self.condition = self.all_data.condition
        self.all_centers = self.load_centers(self.condition)
        self.modules = self.load_modules(modules)
        self.params, self.init_values = self.set_params(config)

    def load_modules(self, modules: List[str]) -> Dict[str, Module]:
        """ Load all modules."""
        return OrderedDict({k: all_modules[k][0] for k in modules})
    
    def set_params(self, config: Dict) -> Tuple[Dict, Dict]:
        """ Set the parameters for all modules."""
        params = {}
        init_values = {}           
        for name, module in self.modules.items():
            params.update(module.params)
            init_values.update({k: v['default'] for k, v in config[name].items()})
        return params, init_values


    def load_centers(self, condition: int) -> List[Tuple[str, Dict[int, List[Tuple]]]]:
        """ Load all category centers for different k and conditions."""
        partition = Partition()
        all_centers = {
            '2_cats': partition.get_centers(4, 2),
            '4_cats': partition.get_centers(4, 4)
        }
        return all_centers['2_cats'] if condition == 1 else all_centers['4_cats']

        
    @property
    def max_k(self) -> int:
        """Return max k based on the condition."""
        return len(self.all_centers)
    
    def fit(self):
        res_dict = {k: [] for k in self.params.keys()}
        for i in range(len(self.all_data)):
            for name, module in self.modules.items():
                if name == 'base':
                    params, prior_fn, likelihood_fn = module.fit(
                        self.params,
                        self.init_values,
                        self.all_data[:i+1], 
                        self.all_centers, 
                        self.max_k,
                        self.config[name]['beta']['bounds']
                    )
                    for k, v in params.items():
                        res_dict[k].append(v)
                        self.init_values[k] = v
                elif name == 'decision':
                    params, prior_fn, likelihood_fn = module.fit(
                        self.params,
                        self.init_values,
                        self.all_data[:i+1], 
                        (prior_fn, likelihood_fn),
                        self.config[name]['phi']['bounds']
                    )
                    for k, v in params.items():
                        res_dict[k].append(v)
                else:
                    raise ValueError("No module found")
        return res_dict

        
                
            

                    

            
        
        
    

    

        

        
