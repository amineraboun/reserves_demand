import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error, median_absolute_error
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import itertools

class CurveFitter:
    """
    Class to fit a curve to a dataset using a variety of curve types.
    The class uses a cross-validated approach to select the best curve type and the best set of variables.
    The class supports the following curve types:
        - Logistic
        - Reduced Logistic
        - Fixed Logistic
        - Double Exponential
        - Exponential
        - Fixed Exponential
        - Arctan
        - Linear

    Parameters
    ----------
        data : pd.DataFrame
            The dataset to fit the curve to.
        dep_var : str
            The name of the dependent variable.
        dummy_col : str, optional
            The name of the dummy variable column. Default is None.
        constant : bool, optional
            Whether to add a constant column to the dataset. Default is True.
        Q : float, optional
            Paramter to set the confidence interval. CI = [(1-Q)/2, 1-(1-Q)/2]. Default is 0.9 for [5%, 95%].
        nFolds : int, optional
            The number of folds to use in the cross-validation. Default is 5.
        parallel : bool, optional
            Whether to use parallel processing for the cross-validation. Default is False.
        search_method : str, optional
            The search method to use for variable selection. Must be either 'backward' or 'forward'. Default is 'backward'.
    
    Attributes
    ----------
        y : pd.Series
            The dependent variable.
        x : pd.DataFrame
            The independent variables.
        dummy : pd.Series
            The dummy variable.
        x_min : pd.Series
            The minimum values of the independent variables.
        x_max : pd.Series
            The maximum values of the independent variables.
        curves : dict
            A dictionary of curve functions.    
        param_names : dict
            A dictionary of parameter names for each curve type.
        init_params : dict
            A dictionary of initial parameters for each curve type.
    """   
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 dep_var: str,
                 main_indep_var :str,
                 dep_var_name: str = None,                 
                 main_indep_var_name: str = None,
                 dummy_col:str=None,
                 constant: bool = True,
                 curves = ['logistic', 'redLogistic',
                           'fixLogistic', 'doubleExp',
                           'exponential', 'fixExponential',
                           'arctan', 'linear'],
                 Q = 0.9,
                 nFolds=5,
                 parallel=False,
                 search_method = 'backward'
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
        if dummy_col is None:
            self.dummy = None
        else:
            assert dummy_col in x.columns, f"{dummy_col} not found in the dataset"
            self.dummy = x.pop(dummy_col)
        
        self.main_indep_var = main_indep_var
        self.main_indep_var_name = main_indep_var_name if main_indep_var_name is not None else main_indep_var
        
        # Normalize x to make the optimization easier
        self.x_min = x.min()
        self.x_max = x.max()
        self.x_raw = x
        self.x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # add constant column if requested
        if constant:
            self.x.insert(0, 'constant', 1)
        
        self.Q = Q
        self.nFolds = nFolds
        kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=42)
        self.cv_indices = kf.split(self.x)
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
            "linear": None
        }
        init_params = {
            # Initial parameters for each curve type
            "logistic": np.array([0, 1, 0]),
            "redLogistic": np.array([0, 1]),
            "fixLogistic": np.array([0.1]), 
            "doubleExp": np.array([0, 0.1, -0.1]),
            "exponential": np.array([0, 0.1]), 
            "fixExponential": np.array([0.1]),
            "arctan": np.array([.1, -.1]), 
            "linear": np.array([])
        }
        self.curves = {_c: curves[_c] for _c in curves}
        self.param_names = {_c: param_names[_c] for _c in curves}
        self.init_params = {_c: init_params[_c] for _c in curves}
        return None
    
    def curve(self, x, w, curvename):
        """Applies the specified curve function to the input data x."""
        assert curvename in self.curves, f"Invalid curve type. Must be one of: {', '.join(self.curves.keys())}"
        p = x.shape[1]
        b = w[-p:]
        g = self._gX(x, b)
        return self.curves[curvename](w, g)
    
    def loss(self, w, curvename, x, y, q):
        """Computes the quantile loss for curve fitting."""
        ypred = self.curve(x, w, curvename)
        e = y - ypred
        return np.maximum(q * e, (q - 1) * e).mean()

    def curveOpt(self, curvename, x, y, q):
        """Optimizes the curve parameters for the given data and curve type."""
        assert curvename in self.curves, f"Invalid curve type: {curvename}"
        
        eta = 1 if self.dummy is not None else 0
        # Initial parameters setup
        M = x.abs().max()
        xinit = -1 / M
        winit = np.concatenate((self.init_params[curvename], np.repeat(-0.1, eta), xinit.values))  
        
        # Bounds setup 
        # To make the convergence easier, bound the exponential parameters to a reasonable range
        _Kepmin = np.log(1e-4) / M
        _Kepmax = np.log(1e3) / M
        lb = [None for _ in range(len(winit) - x.shape[1])] + list(_Kepmin.values)
        ub = [None for _ in range(len(winit) - x.shape[1])] + list(_Kepmax.values)
        bounds = list(zip(lb, ub))

        # Perform the optimization
        result = minimize(self.loss, winit, args=(curvename, x, y, q), bounds=bounds, method='L-BFGS-B')
        return result.x, result

    def _run_fold(self, train_index, test_index, columns, curvename):
        """Executes a single fold of cross-validation for the given columns and curve."""
        
        X_train, X_test = self.x.iloc[train_index][columns], self.x.iloc[test_index][columns]
        y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

        # Fit the model using the specified curve type and predict on both training and testing sets
        # The ross validation is chosen on the mean absolute error loss q=0.5
        popt, _ = self.curveOpt(curvename, X_train, y_train, 0.5)
        y_train_pred = self.curve(X_train, popt, self.dummy, curvename)
        y_test_pred = self.curve(X_test, popt, self.dummy, curvename)

        # Evaluate and return performance metrics for both sets
        train_metrics = self._evaluate_metrics(y_train, y_train_pred)
        test_metrics = self._evaluate_metrics(y_test, y_test_pred)
        return train_metrics, test_metrics

    def _run_cv(self, columns_subset, curvename):
        """Runs cross-validation for the specified columns subset and curve type."""

        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self._run_fold, train_idx, test_idx, columns_subset, curvename)
                           for train_idx, test_idx in self.cv_splits]
                metrics = [future.result() for future in as_completed(futures)]
        else:
            metrics = [self._run_fold(train_idx, test_idx, columns_subset, curvename)
                       for train_idx, test_idx in self.cv_splits]

        # Compute the mean of the metrics across all folds for training and validation sets
        metrics_df = pd.concat([pd.concat(m, axis=1) for m in metrics], axis=1)
        import pdb; pdb.set_trace()
        train_metrics_mean = metrics_df.xs(0, axis=1, level=1).mean(axis=1)
        validation_metrics_mean = metrics_df.xs(1, axis=1, level=1).mean(axis=1)
        return train_metrics_mean, validation_metrics_mean

    def _find_best_combination(self, combinations, curvename, on):
        """Finds the best combination of columns based on the specified performance metric for a specific curve type."""
        
        best_combination, best_validation_metrics, best_train_metrics = None, None, None
        if self.parallel:
            with ProcessPoolExecutor() as executor:
                # Map each combination to a future object for asynchronous execution
                future_to_comb = {executor.submit(self._run_cv, comb, curvename): comb for comb in combinations}
                for future in as_completed(future_to_comb):
                    comb = future_to_comb[future]
                    try:
                        train_metrics, validation_metrics = future.result()
                        # Determine the best combination by comparing validation performance
                        if best_validation_metrics is None or validation_metrics[on] < best_validation_metrics[on]:
                            best_combination, best_validation_metrics, best_train_metrics = comb, validation_metrics, train_metrics
                    except Exception as exc:
                        print(f'Combination {comb} generated an exception: {exc}')
        else:
            for comb in tqdm(combinations):
                # Sequentially evaluate each combination for non-parallel execution
                train_metrics, validation_metrics = self._run_cv(comb, curvename)
                if best_validation_metrics is None or validation_metrics[on] < best_validation_metrics[on]:
                    best_combination, best_validation_metrics, best_train_metrics = comb, validation_metrics, train_metrics

        return best_combination, best_train_metrics, best_validation_metrics

    def cross_validate(self, curvename, on='RMSE'):
        """Performs cross-validation to find the best combination of variables for the specified curve type."""
        perf_stats = ['RMSE', 'MAE', 'MAPE', 'R2', 'MSLE', 'MedianAE']
        assert on in perf_stats, f'Performance comparison is made on one of the following metrics {perf_stats}'
        columns = list(self.x.columns)
    
        # Initialize variables to store the best metrics and combination found
        best_combination, best_train_metrics, best_validation_metrics = None, None, None

        # Generate combinations based on the specified search method
        if self.search_method in ['backward', 'forward']:
            best_combination = columns[:] if self.search_method == 'backward' else []
            while True:
                # Generate candidate combinations for the next iteration
                if self.search_method == 'backward':
                    candidate_columns = [best_combination[:i] + best_combination[(i + 1):] for i in range(len(best_combination))] 
                else:
                    candidate_columns = [best_combination + [c] for c in columns if c not in best_combination]
                
                # If no candidate combinations are available, break from the loop
                if not candidate_columns:
                    break
                
                # Find the best combination from the candidates
                best_combination, best_train_metrics, best_validation_metrics = self._find_best_combination(candidate_columns, curvename, on)
                
                # In forward search, remove selected columns from the list of remaining columns
                if self.search_method == 'forward' and best_combination:
                    columns = [col for col in columns if col not in best_combination]
                    if not columns:
                        break

        elif self.search_method == 'all_combinations':
            # Evaluate all possible combinations of columns
            all_combinations = [list(c) for i in range(1, len(columns) + 1) for c in itertools.combinations(columns, i)]
            best_combination, best_train_metrics, best_validation_metrics = self._find_best_combination(all_combinations, curvename, on)

        # Return the best combination of features and the corresponding training and validation metrics
        return best_combination, best_train_metrics, best_validation_metrics
    

    def variable_select(self, on='RMSE', verbose=True):
        """Performs variable selection for each curve type and returns the best curve based on the specified metric."""
        results = []
        for curvename in self.curves.keys():
            if verbose:
                print('=' * 50)
                print(f"Running cross-validation for {curvename} curve")
                print('=' * 50)
            # Perform cross-validation and gather metrics
            combination, train_metrics, val_metrics = self.cross_validate(curvename, on)
            result = {
                'Curve': curvename,
                'Best Combination': combination,
                **{f'Train {metric}': value for metric, value in train_metrics.items()},
                **{f'Validation {metric}': value for metric, value in val_metrics.items()}
            }
            results.append(result)
            if verbose:
                print(f"Best combination for {curvename}: {combination}")
                print(f"Training Metrics: {train_metrics}")
                print(f"Validation Metrics: {val_metrics}\n")

        varselect_result = pd.DataFrame(results)
        self.varselect_result = varselect_result  # Store for later use

        # Optionally identify the best curve based on a specific validation metric, e.g., RMSE
        best_curve = varselect_result.sort_values(by=f'Validation {on}').iloc[0]['Curve']
        if verbose:
            print(f"Best curve based on Validation {on}: {best_curve}")
        return best_curve, varselect_result

    def _get_params(self, curvename, popt, xcols):
        """Extracts the parameters from the optimized curve parameters."""
        if self.dummy is not None:
            paramnames  = self.param_names[curvename] + ['eta'] + xcols
        else:
            paramnames  = self.param_names[curvename] + xcols
        return {paramnames[_i]: popt[_i] for _i in range(len(paramnames))}
            
    # Fit the curve with the best combination of variables
    def fit_curve(self, curvename, xcols):
        """Fits the curve to the data using a selected list of exogenous variables."""
        # Fit the model with the best combination of variables
        X = self.x[xcols]
        # Fit the mean curve        
        popt, _ = self.curveOpt(curvename, X, self.y, q=0.5)
        # Fit the upper and lower curves for the confidence interval
        qlb = (1 - self.Q) / 2
        qub = 1 - qlb
        popt_up, _ = self.curveOpt(curvename, X, self.y, q=qub)
        popt_down, _ = self.curveOpt(curvename, X, self.y, q=qlb)

        return popt, popt_up, popt_down

    def predict(self, curvename, x, xcols, popt, popt_up, popt_down):
        """Predicts the dependent variable using the fitted curve and the confidence interval."""
        X = x[xcols]
        ypred = self.curve(X, popt, curvename)
        yqlb = self.curve(X, popt_down, curvename)
        yqub = self.curve(X, popt_up, curvename)
        return ypred, yqlb, yqub, X
    
    def perf_metrics(self, y, ypred, yqlb, yqub):
        """
        Computes performance metrics for the fitted curve.
        The performance metrics include 
            RMSE: Root Mean Squared Error
            MAE: Mean Absolute Error
            MAPE: Mean Absolute Percentage Error
            R2: R-squared
            MSLE: Mean Squared Log Error
            MedianAE: Median Absolute Error
            MIS: Mean Interval Score
        """
        metrics = self._evaluate_metrics(y, ypred)
        metrics['MIS'] = self._mis(y, yqlb, yqub)
        return metrics
    
    def plot_curve_with_confidence_interval(self, ypred, yqlb, yqub, ax = None, title =''):
        """Plots the fitted curve along with the confidence interval."""
        # Plot the data
        if ax is None:
            _, ax = plt.subplots(1, 1)
        
        ax.scatter(self.x_raw[self.main_indep_var], self.y, label='Data')
        ax.plot(self.x_raw[self.main_indep_var], ypred, 'r-', label='Fitted function')

        # Plot the confidence interval
        ax.fill_between(self.x_raw[self.main_indep_var], yqlb, yqub, color='gray', alpha=0.5, label=f'{int(self.Q*100)}% CI')
        ax.legend()
        ax.set_xlabel(self.main_indep_var_name)
        ax.set_ylabel(self.dep_var_name)
        ax.set_title(title)
        return ax
    
    def fit_best_curves(self):
        if not hasattr(self, 'varselect_result'):
                self.variable_select(verbose=False)
                
        fitted_curves = self.varselect_result['Curve'].unique()
        _best_params = {}
        for curvename in fitted_curves:
            best_combination = self.varselect_result.loc[self.varselect_result['Curve'] == curvename, 'Best Combination'].values[0]
            popt, popt_up, popt_down = self.fit_curve(curvename, best_combination)
            _best_params[curvename] = {'vars':best_combination, 'avg': popt, 'upper':popt_up, 'lower':popt_down}
        self.best_curves_params = _best_params
        return _best_params
    
    def predict_best_curve(self, curvename, X=None):
        if not hasattr(self, 'best_curves_params'):
            self.fit_best_curves(verbose=False)
        if X is None:
            X = self.x.copy()
        _params = self.best_curves_params[curvename]       

        ypred, yqlb, yqub, X = self.predict(curvename, X, _params['vars'], _params['avg'], _params['up'], _params['down'])
        return ypred, yqlb, yqub, X
    
    def compare_curves(self, plot=True):
        if not hasattr(self, 'best_curves_params'):
                self.fit_best_curves(verbose=False)
                
        fitted_curves = self.fit_best_curves.keys()
        predictions = {}; _params = {}; _perf_metrics = {}
        _perf_metrics = {}; _params = {}
        for curvename in fitted_curves:
            ypred, yqlb, yqub, _ = self.predict_best_curve(curvename)
            predictions[curvename] = {'ypred': ypred, 'yqlb': yqlb, 'yqub': yqub}
            _params = self.best_curves_params[curvename]       
            _perf_metrics[curvename] = self.perf_metrics(self.y, ypred, yqlb, yqub)
            _params[curvename] = pd.concat([self._get_params(self, _params['avg'], _params['vars']), 
                                            self._get_params(self, _params['up'], _params['vars']),
                                            self._get_params(self, _params['down'], _params['vars'])
                                            ], axis=1, keys = ['avg', 'upper bound', 'lower bound'])
        perf_metrics_df = pd.concat(_perf_metrics, axis=1)
        param_df = pd.concat(_params)

        if plot:
            curves_l = list(predictions.keys())
            nplots = len(curves_l) 
            ncols = 2
            nrows = int(np.ceil(nplots/ncols))
            plt.subplots(nrows,ncols, figsize = (ncols*5, nrows*5))
            for i, curvename in enumerate(curves_l):
                ax = plt.subplot(nrows,ncols, i+1)
                preds = predictions[curvename]
                ypred, yqlb, yqub, _ = preds['ypred'], preds['yqlb'], preds['yqub']
                self.plot_curve_with_confidence_interval(ypred, yqlb, yqub, ax, title=curvename) 
            plt.tight_layout()
            plt.show()
            
        return perf_metrics_df, param_df
    
    
    # Helper functions
    def _evaluate_metrics(self, y_true, y_pred):
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'R2': r2_score(y_true, y_pred),
            'MSLE': mean_squared_log_error(y_true, y_pred),
            'MedianAE': median_absolute_error(y_true, y_pred)
        }
        return pd.Series(metrics)
    
    def _mis(self, y, yqlb, yqub):
        return ((yqub - yqlb) + (2 / self.Q) * (yqlb - y) * (y < yqlb) + (2 / self.Q) * (y - yqub) * (y > yqub)).mean()

    def _gX(self, x, b):
        """Computes the dot product of x and b for the curve calculations."""
        return np.dot(x, b)
        
    # Various curves follow, call them through curve()
    def _logistic(self, w, g):
        alpha, beta, kappa = w[:3]
        yhat = alpha + kappa / (1 - beta * np.exp(g))

        if self.dummy is not None:
            eta = w[3]
            yhat += eta * self.dummy
        return yhat

    def _redLogistic(self, w, g):
        alpha, beta = w[:2]
        yhat = alpha + 1 / (1 - beta * np.exp(g))

        if self.dummy is not None:
            eta = w[2]
            yhat += eta * self.dummy
        return yhat

    def _fixLogistic(self, w, g):
        alpha = w[0]
        yhat = alpha + 1 / (1 - np.exp(g))

        if self.dummy is not None:
            eta = w[1]
            yhat += eta * self.dummy
        return yhat

    def _doubleExp(self, w, g):
        alpha, beta, rho = w[:3]
        yhat = alpha + beta * np.exp(rho * np.exp(g))

        if self.dummy is not None:
            eta = w[3]
            yhat += eta * self.dummy
        return yhat

    def _exponential(self, w, g):
        alpha, beta = w[:2]
        yhat = alpha + beta * np.exp(g)

        if self.dummy is not None:
            eta = w[2]
            yhat += eta * self.dummy
        return yhat

    def _fixExponential(self, w, g):
        beta = w[0]
        yhat = beta * np.exp(g)

        if self.dummy is not None:
            eta = w[1]
            yhat += eta * self.dummy
        return yhat

    def _arctan(self, w, g):
        alpha, beta = w[:2]
        yhat = alpha + beta * np.arctan(g)

        if self.dummy is not None:
            eta = w[2]
            yhat += eta * self.dummy
        return yhat

    def _linear(self, w, g):
        yhat = g

        if self.dummy is not None:
            eta = w[0]
            yhat += eta * self.dummy
        return yhat
