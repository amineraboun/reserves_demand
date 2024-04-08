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
        self.x = x / (self.x_max - self.x_min)
        self.excess_reserves = self.x[self.main_indep_var]
        self.exogs = self.x.drop(columns=[self.main_indep_var])

        
        self.Q = Q
        self.nFolds = nFolds
        self.kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=42)
        self.parallel = parallel
        self.search_method = search_method
        
        curves = {
            "logistic": _logistic,
            "redLogistic": _redLogistic,
            "fixLogistic": _fixLogistic,
            "doubleExp": _doubleExp,
            "exponential": _exponential,
            "fixExponential": _fixExponential,
            "arctan": _arctan,
            "linear": _linear
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
            "linear": ["alpha", "beta"]
        }

        self.curves = {_c: curves[_c] for _c in curves}
        self.param_names = {_c: param_names[_c] for _c in curves}

        self.varselect_result = pd.DataFrame()
        return None
    
    def curveOpt(self, curvename, x, y):
        
        p0 = -1* np.ones(len(self.param_names[curvename]))
        curvefunc = self.curves[curvename]
        
        popt, pcov = curve_fit(curvefunc, x, y, p0=p0)
        return popt, pcov

# Helper functions
def _logistic(x, alpha, beta, kappa):
    return alpha + kappa / (1 - beta * np.exp(x))

def _redLogistic(x, alpha, beta):
    return alpha + 1 / (1 - beta * np.exp(x))

def _fixLogistic(x, alpha):
    return alpha + 1 / (1 - np.exp(x))

def _doubleExp(x, alpha, beta, rho):
    return alpha + beta * np.exp(rho * np.exp(x))

def _exponential(x, alpha, beta):
    return alpha + beta * np.exp(x)

def _fixExponential(x, beta):
    return beta * np.exp(x)

def _arctan(x, alpha, beta):
    return alpha + beta * np.arctan(x)

def _linear(x):
    return alpha + beta *x