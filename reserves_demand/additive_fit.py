import pandas as pd
import numpy as np
from tqdm import tqdm 

from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

class AdditiveFit:
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 dep_var: str,
                 main_indep_var :str,
                 dep_var_name: str = None,                 
                 main_indep_var_name: str = None,
                 constant: bool = True,
                 curves = ['logistic', 'redLogistic',
                           'fixLogistic', 'doubleExp',
                           'exponential', 'fixExponential',
                           'arctan', 'linear'],
                 Q = 0.9,
                 nFolds=5,
                 parallel=False,
                 search_method='backward'
                 ):
        assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
        assert dep_var in data.columns, f"{dep_var} not found in the dataset"
        assert main_indep_var in data.columns, f"{main_indep_var} not found in the dataset"
        assert 0 < Q < 1, "Q should be between 0 and 1 for valid confidence intervals"
        assert nFolds > 1, "nFolds must be greater than 1 for cross-validation"
        assert search_method in ['backward', 'forward', 'all_combinations'], "search_method must be 'backward', 'forward', or 'all_combinations'"

        # Preprocessing data
        data = data.sort_values(main_indep_var).dropna().reset_index(drop=True)        
        
        # Prepare the y variable
        self.y = data[dep_var]
        self.dep_var_name = dep_var_name if dep_var_name is not None else dep_var
        
        # Prepare the x variables
        x = data.drop(columns=[dep_var])
        
        self.main_indep_var = main_indep_var
        self.main_indep_var_name = main_indep_var_name if main_indep_var_name is not None else main_indep_var
        
        # Normalize x to make the optimization easier
        self.x_min = x.min()
        self.x_max = x.max()
        self.x_raw = x
        self.x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # add constant column if requested
        if constant:
            self.constant = True
            self.x.insert(0, 'constant', 1)
        
        self.Q = Q
        self.nFolds = nFolds
        self.kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=42)
        self.parallel = parallel
        self.search_method = search_method
        
        curves = {
            "logistic": self._logistic,
            "redLogistic": self._redLogistic,
            "fixLogistic": self._fixLogistic,
            "doubleExp": self._doubleExp,
            "exponential": self._exponential,
            "fixExponential": self._fixExponential,
            "arctan": self._arctan,
            "linear": self._linear
        }
        param_names = {
            # Names of parameters for each curve type
            "logistic": ["alpha", "beta", "kappa"],
            "redLogistic": ["alpha", "beta"],
            "fixLogistic": ["alpha"],
            "doubleExp": ["alpha", "beta", "rho"],
            "exponential": ["alpha", "beta"],
            "fixExponential": ["beta"],
            "arctan": ["alpha", "beta"],
            "linear": []
        }

        self.curves = {_c: curves[_c] for _c in curves}
        self.param_names = {_c: param_names[_c] for _c in curves}

        self.varselect_result = pd.DataFrame()
        return None