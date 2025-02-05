import numpy as np
from dataclasses import make_dataclass
from collections import OrderedDict
from scipy.optimize import minimize

from .modules import Module, Base
from .utils.data import CategoryDataset, TrialData
from .utils.partition import Partition

from typing import Dict, Tuple, List

# Define all modules, each module is a tuple of (module, priority)
all_modules = {
    'base': (Base(), 0)
}
all_modules = dict(sorted(all_modules.items(), key=lambda x: x[1][1]))
    
from .config import config



class BayesianModel:
    def __init__(self, iSub: int, modules: List[str] = ['base']):
        self.config = config
        self.all_data = CategoryDataset.load_data_from_file().get_subject(iSub)
        self.condition = self.all_data.condition
        self.all_centers = self.load_centers(self.condition)
        self.modules = self.load_modules(modules)
        self.params = self.set_params(config)

    def reload(self, iSub: int, modules: List[str] = ['base']):
        self.all_data = CategoryDataset.load_data_from_file().get_subject(iSub)
        self.condition = self.all_data.condition
        self.all_centers = self.load_centers(self.condition)
        self.modules = self.load_modules(modules)
        self.params = self.set_params(config)

    def load_modules(self, modules: List[str]) -> Dict[str, Module]:
        """ Load all modules."""
        return OrderedDict({k: all_modules[k][0] for k in modules})
    
    def set_params(self, config: Dict) -> make_dataclass:
        """ Set the parameters for all modules."""
        params = {}
        init_values = {}            
        for name, module in self.modules.items():
            params.update(module.params)
            init_values.update({k: v['default'] for k, v in config[name].items()})
        cls = make_dataclass('ModelParams', params)
        return cls(**init_values)     


    def load_centers(self, condition: int) -> Dict[str, List[Tuple[str, Dict[int, List[Tuple]]]]]:
        """ Load all category centers for different k and conditions."""
        partition = Partition()
        all_centers = {
            '2_cats': partition.get_centers(4, 2),
            '4_cats': partition.get_centers(4, 4)
        }
        return all_centers['2_cats'] if condition == 1 else all_centers['4_cats']
    
    def get_centers(self, k: int) -> np.ndarray:
        """ Get the specific category centers for a given k."""
        if 0 <= k < len(self.all_centers):
            return np.array(list(self.all_centers[k][1].values()), dtype=np.float32)
        else:
            raise ValueError(f"Invalid k for condition {self.condition}")
        
    @property
    def max_k(self) -> int:
        """Return max k based on the condition."""
        return len(self.all_centers)
    
    def fit(self):
        res_dict = {}
        if 'base' in self.modules:
            base_cls = make_dataclass('BaseParams', self.modules['base'].params)
            all_betas : List[List[float]] = [] #shape (max_k, n_trials)
            all_losses : List[list[float]] = [] #shape (max_k, n_trials)
            beta_bounds = config['base']['beta']['bounds']                
            for k in range(self.max_k):
                betas = []
                losses = []     
                for i in range(len(self.all_data)):
                    result = minimize(
                        lambda beta: self.modules['base'].loss_fn(base_cls(k=k, beta=beta), self.all_data[:i+1], self.get_centers(k), max_k=self.max_k),
                        x0=[betas[-1]] if betas else [config['base']['beta']['default']],
                        bounds=[beta_bounds]
                    )
                    betas.append(result.x[0])
                    losses.append(result.fun)
                all_betas.append(betas)
                all_losses.append(losses)
            best_k = np.argmin(all_losses, axis=0) #shape (n_trials,)
            best_betas = [all_betas[k][i] for i, k in enumerate(best_k)] #shape (n_trials,)
            self.params.k , self.params.beta = best_k[-1], best_betas[-1]
            print(f"Best k: {best_k[-1]}, Best beta: {best_betas[-1]}")
            res_dict['base'] ={
                'k': best_k,
                'beta': best_betas,
            }                    
        else:
            raise ValueError("No module found")
        return res_dict
        
            
                
            

                    

            
        
        
    

    

        

        
