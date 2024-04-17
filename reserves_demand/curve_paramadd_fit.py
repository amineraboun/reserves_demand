import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy.optimize import minimize
from sklearn.model_selection import KFold
from reserves_demand.utils import evaluate_metrics, perf_metrics
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1)
plt.rc('axes', titlesize='large')    
plt.rc('axes', labelsize='large')   
plt.rc('xtick', labelsize='large')   
plt.rc('ytick', labelsize='large')   
plt.rc('legend', fontsize='large')   
plt.rc('figure', titlesize='x-large') 


class CurveParamAdditiveFit:
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
            The dataset containing the dependent and independent variables.
        dep_var : str
            The name of the dependent variable.
        main_indep_var : str
            The name of the main independent variable.
        dep_var_name : str, optional
            The name of the dependent variable for plotting. Default is None.
        main_indep_var_name : str, optional
            The name of the main independent variable for plotting. Default is None.
        constant : bool, optional
            Whether to include a constant column in the independent variables. Default is True.
        curves : list, optional
            The list of curve types to consider. Default is None.
        Q : float, optional
            The confidence level for the confidence interval. Default is 0.9.
        nFolds : int, optional
            The number of folds for cross-validation. Default is 5.
        parallel : bool, optional
            Whether to run cross-validation in parallel. Default is False.
        search_method : str, optional
            The method to use for variable selection. Default is 'backward'.
    
    Attributes
    ----------
        y : pd.Series
            The dependent variable.
        x : pd.DataFrame
            The independent variables.
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
                 curves:list = None,
                 Q: float = 0.9,
                 nFolds: int=5,
                 parallel: bool =False,
                 search_method: str = 'backward'
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
        self.dep_var = dep_var
        self.dep_var_name = dep_var_name if dep_var_name is not None else dep_var
        
        # Prepare the x variables
        x = data.drop(columns=[dep_var])
        
        self.main_indep_var = main_indep_var
        self.main_indep_var_name = main_indep_var_name if main_indep_var_name is not None else main_indep_var
        self.other_indep_vars = list(x.columns.difference([main_indep_var]))

        # Normalize x to make the optimization easier
        self.x_min = x.min()
        self.x_max = x.max()
        self.x_raw = x
        self.x = x  / (self.x_max - self.x_min)
        
        self.Q = Q
        self.nFolds = nFolds
        self.kf = KFold(n_splits=self.nFolds, shuffle=True, random_state=42)
        self.parallel = parallel
        self.search_method = search_method
        
        curves = {
            "logistic": self.logistic,
            "redLogistic": self.redLogistic,
            "fixLogistic": self.fixLogistic,
            "doubleExp": self.doubleExp,
            "exponential": self.exponential,
            "fixExponential": self.fixExponential,
            "arctan": self.arctan,
            "linear": self.linear
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
            "linear": ["alpha"]
        }
        init_params = {
            # Initial parameters for each curve type
            "logistic": np.array([0, 0.1, 0]),
            "redLogistic": np.array([0, 0.1]),
            "fixLogistic": np.array([0.1]), 
            "doubleExp": np.array([0, 0.1, -0.1]),
            "exponential": np.array([0, 0.1]), 
            "fixExponential": np.array([0.1]),
            "arctan": np.array([0.1, -0.1]), 
            "linear": np.array([0.1])
        }
        if curves is None:
            curves = list(curves.keys())
        else:
            assert all(_c in curves for _c in curves), f"Invalid curve type. Must be one of: {', '.join(curves.keys())}"
        
        self.curves = {_c: curves[_c] for _c in curves}
        self.param_names = {_c: param_names[_c] for _c in curves}
        self.init_params = {_c: init_params[_c] for _c in curves}

        self.varselect_result = pd.DataFrame()
        self.best_curves_params = None
        return None
    
    def plot_x_y(self, title =''):
        """Plots the dependent variable y against all the independent variables x."""
        
        nplots = self.x_raw.shape[1]
        ncols = 1 if nplots == 1 else 2
        nrows = int(np.ceil(nplots/ncols))
        _ = plt.subplots(nrows,ncols, figsize = (ncols*5, nrows*5))
        for i, col in enumerate(self.x_raw.columns):
            ax = plt.subplot(nrows,ncols, i+1)
            ax.scatter(self.x_raw[col], self.y)
            ax.set_xlabel(col)
            ax.set_ylabel(self.dep_var_name)
        if title == '':
            title = f"{self.dep_var_name} against independent variables"
        plt.suptitle(title)        
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.show()
        return None
        
    def curve(self, x, w, curvename):
        """Applies the specified curve function to the input data x."""
        assert curvename in self.curves, f"Invalid curve type. Must be one of: {', '.join(self.curves.keys())}"
        
        assert self.main_indep_var in x.columns, f"{self.main_indep_var} not found in the dataset"
        main_x = x[self.main_indep_var]
        p1 = len(self.param_names[curvename])
        # function contribution
        funccontrib = np.nan_to_num(self.curves[curvename](w[:p1], w[p1]*main_x))
        
        # other exog variables contribution
        other_x = x.drop(columns=[self.main_indep_var])
        if other_x.empty:
            return funccontrib
        else:
            p2 = other_x.shape[1]
            exogcontrib = self.gX(other_x, w[-p2:])
            return funccontrib + exogcontrib

    def curveOpt(self, curvename, x, y, q, winit=None, verbose=False):
        """Optimizes the curve parameters for the given data and curve type."""
        assert curvename in self.curves, f"Invalid curve type: {curvename}"
        
        # Initial parameters setup
        if winit is None:
            M = x.abs().max()
            xinit = -1 / M
            winit = np.concatenate((self.init_params[curvename], xinit.values))  
        else:
            assert len(winit) == len(self.init_params[curvename]) + x.shape[1], "Invalid initial parameters"

        # Use minimize for other cases
        result = minimize(self.loss, winit, args=(curvename, x, y, q))
        popt = result.x
        if verbose and not result.success:
            print(f"Optimization failed for {curvename} curve with message: {result.message}")

        return popt
    
    def loss(self, w, curvename, x, y, q):
        """Computes the quantile loss for curve fitting."""
        ypred = self.curve(x, w, curvename)
        e = y - ypred
        if q == 0.5:
            return (e**2).mean()
        else:
            return np.maximum(q * e, (q - 1) * e).mean()

    def run_fold(self, train_index, test_index, columns, curvename):
        """Executes a single fold of cross-validation for the given columns and curve."""
        
        X_train, X_test = self.x.iloc[train_index][columns], self.x.iloc[test_index][columns]
        y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

        # Fit the model using the specified curve type and predict on both training and testing sets
        # The ross validation is chosen on the mean absolute error loss q=0.5
        popt = self.curveOpt(curvename, X_train, y_train, 0.5)
        y_train_pred = self.curve(X_train, popt, curvename)
        y_test_pred = self.curve(X_test, popt, curvename)

        # Evaluate and return performance metrics for both sets
        train_metrics = evaluate_metrics(y_train, y_train_pred)
        test_metrics = evaluate_metrics(y_test, y_test_pred)
        return train_metrics, test_metrics

    def run_cv(self, columns_subset, curvename):
        """Runs cross-validation for the specified columns subset and curve type."""

        # Initialize lists to store metrics for all folds
        train_metrics_list = []
        test_metrics_list = []

        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.run_fold, train_idx, test_idx, columns_subset, curvename)
                        for train_idx, test_idx in self.kf.split(self.x)]
                # Collect results as they complete
                for future in as_completed(futures):
                    train_metrics, test_metrics = future.result()
                    train_metrics_list.append(train_metrics)
                    test_metrics_list.append(test_metrics)
        else:
            for train_idx, test_idx in self.kf.split(self.x):
                train_metrics, test_metrics = self.run_fold(train_idx, test_idx, columns_subset, curvename)
                train_metrics_list.append(train_metrics)
                test_metrics_list.append(test_metrics)

        # Convert lists of Series to DataFrames and compute mean for each metric
        train_metrics_df = pd.DataFrame(train_metrics_list)
        test_metrics_df = pd.DataFrame(test_metrics_list)
        train_metrics_mean = train_metrics_df.mean()
        validation_metrics_mean = test_metrics_df.mean()

        return train_metrics_mean, validation_metrics_mean

    def find_best_combination(self, combinations, curvename, on):
        """Finds the best combination of columns based on the specified performance metric for a specific curve type."""

        train_metrics = {}
        validation_metrics = {}
        if self.parallel:
            with ProcessPoolExecutor() as executor:
                future_to_comb = {executor.submit(self.run_cv, comb, curvename): comb for comb in combinations}
                progress_bar = tqdm(total=len(combinations), desc="", unit=" it")
                for future in as_completed(future_to_comb):
                    comb = future_to_comb[future]
                    try:
                        train_metrics[tuple(comb)], validation_metrics[tuple(comb)] = future.result()                        
                    except Exception as exc:
                        print(f'Combination {comb} generated an exception: {exc}')
                    finally:
                        # Update progress bar by one step for each completed task
                        progress_bar.update(1)
            progress_bar.close()
        else:            
            for comb in tqdm(combinations):
                train_metrics[tuple(comb)], validation_metrics[tuple(comb)] = self.run_cv(comb, curvename)
                 
        # Select the combination with the best (minimum) specified validation metric
        if validation_metrics and all(on in metrics for metrics in validation_metrics.values()):
            best_combination = min(validation_metrics, key=lambda x: validation_metrics[x][on])
            best_train_metrics, best_validation_metrics = train_metrics[best_combination], validation_metrics[best_combination]
            best_combination = list(best_combination)
        else:
            raise ValueError(f"The metric '{on}' is not found in the validation metrics")

        return best_combination, best_train_metrics, best_validation_metrics


    def cross_validate(self, curvename, on='RMSE'):
        """Performs cross-validation to find the best combination of variables for the specified curve type."""
        perf_stats = ['RMSE', 'MAE', 'MAPE', 'R2', 'MSLE', 'MedianAE']
        assert on in perf_stats, f'Performance comparison is made on one of the following metrics {perf_stats}'
        columns = list(self.other_indep_vars)        

        # Generate combinations based on the specified search method
        if self.search_method in ['backward', 'forward']:
            
            # Initialize the best combination with the result of all the columns in the backward setup 
            # or the main independent variable in the forward setup 
            best_combination = [self.main_indep_var] if self.search_method == 'forward' else [self.main_indep_var] + columns
            
            best_train_metrics, best_validation_metrics = self.run_cv(best_combination, curvename)
            best_validation_metric_value = best_validation_metrics[on]

            while True:
                # Generate candidate combinations for the next iteration
                if self.search_method == 'backward':
                    _eligb_columns = [c for c in best_combination if c != self.main_indep_var]
                    if len(_eligb_columns) ==0:
                        candidate_combinations = [[self.main_indep_var]]
                    else:
                        candidate_combinations = [[self.main_indep_var] + list(comb) for comb in itertools.combinations(_eligb_columns, len(_eligb_columns) - 1)] 
                else:
                    candidate_combinations = [best_combination + [c] for c in columns if c not in best_combination]
                
                # Find the best combination from the candidates
                candidates_best_combination, candidates_best_train_metrics, candidates_best_validation_metrics = self.find_best_combination(candidate_combinations, curvename, on)
                if candidates_best_validation_metrics[on] < best_validation_metric_value:
                    best_combination, best_train_metrics, best_validation_metrics = candidates_best_combination, candidates_best_train_metrics, candidates_best_validation_metrics
                    best_validation_metric_value = best_validation_metrics[on]
                else:
                    break

                # If no candidate combinations are available, break from the loop
                if (self.search_method == 'backward'):
                    if (not _eligb_columns):
                        break
                
                # In forward search, remove selected columns from the list of remaining columns
                if self.search_method == 'forward' and best_combination:
                    columns = [col for col in columns if col not in best_combination]
                    if not columns:
                        break

        elif self.search_method == 'all_combinations':
            # Evaluate all possible combinations of columns
            if len(columns) > 1:
                all_combinations = [[self.main_indep_var]] + [[self.main_indep_var] +list(c) for i in range(1, len(columns) + 1) for c in itertools.combinations(columns, i)]
            elif len(columns) == 1:
                all_combinations = [[self.main_indep_var], [self.main_indep_var] + columns]
            else:
                all_combinations = [[self.main_indep_var]]
            best_combination, best_train_metrics, best_validation_metrics = self.find_best_combination(all_combinations, curvename, on)

        # Return the best combination of features and the corresponding training and validation metrics
        return best_combination, best_train_metrics, best_validation_metrics
    
    def variable_select(self, on='RMSE', verbose=True, plot=True):
        """
        Performs variable selection for each curve type and returns the best curve based on the specified metric.
        The function also returns the performance metrics for each curve type.
        
        """
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
                print("Training Metrics:")
                print(train_metrics)
                print("\nValidation Metrics:")
                print(val_metrics)
                print('\n')
        cv_result = pd.DataFrame(results)  # Store for later use
        self.varselect_result = cv_result

        # Optionally identify the best curve based on a specific validation metric, e.g., RMSE
        best_curve = self.varselect_result.sort_values(by=f'Validation {on}').iloc[0]['Curve']
        if verbose:
            print('=' * 50)
            print(f"Best curve based on Validation {on}: \033[1m{best_curve}\033[0m curve")            
            print('=' * 50)

        if plot:
            _oncomp = cv_result.sort_values(f'Validation {on}').set_index('Curve')
            _, ax = plt.subplots(1,1,figsize=(10, 5))
            _oncomp[[f'Train {on}', f'Validation {on}']].plot(kind = 'barh', ax=ax)
            ax.set_xlabel(on)
            ax.set_title(f'Comparison of {on} for different curves')
            ax.legend(loc='lower right', title='', frameon=False)
            plt.show()
        return best_curve, self.varselect_result

    def indep_vars_occurence(self, normalize=True, plot=True, title=''):
        """
        Gives the number of occurences of each independent variable in the best combinations of variables for each curve.
        The function can be used to identify the most important variables in the dataset.
        Parameters
        ----------
            normalize : bool, optional
                Whether to normalize the occurrences as a percentage. Default is True.
            plot : bool, optional
                Whether to plot the occurrences. Default is True.
            title : str, optional
                The title of the plot. Default is ''.
        Returns
        -------
            variable_percentage : pd.Series
                The percentage of occurrence of each independent variable in the best combinations.
        """
        if self.varselect_result.empty:
            self.variable_select(verbose=False)
        curves_best_combo = self.varselect_result.loc[:, 'Best Combination']
        # curves_best_combo = self.varselect_result.set_index('Curve')['Best Combination']
        all_variables = [var for sublist in curves_best_combo for var in sublist]

        # Count occurrences of each variable
        variable_counts = Counter(all_variables)

        if normalize:
            # Compute the percentage for each variable
            total_curves = len(curves_best_combo)
            variable_percentage = pd.Series({var: (count / total_curves) * 100 for var, count in variable_counts.items()}).sort_values()
        else:
            variable_percentage = pd.Series(variable_counts).sort_values()

        if plot:
            xlabel = 'Percentage of Occurrence' if normalize else 'Number of Occurrence'
            
            _, ax = plt.subplots(1,1,figsize=(10, 5))
            variable_percentage.plot(kind='barh', ax =ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Independent Variables')
            if title == '':
                title = 'Occurrence of Independent Variables in best combinations per curve'
            ax.set_title(title)
            plt.show()
        return variable_percentage
    
    def get_params(self, curvename, popt, xcols):
        """Extracts the parameters from the optimized curve parameters."""
        paramnames  = self.param_names[curvename] + xcols
        return {paramnames[_i]: popt[_i] for _i in range(len(paramnames))}           
    
    def fit_curve(self, curvename, xcols):
        """Fits the curve to the data using a selected list of exogenous variables."""
        # Fit the model with the best combination of variables
        X = self.x[xcols]
        # Fit the mean curve        
        popt = self.curveOpt(curvename, X, self.y, q=0.5)
        # Fit the upper and lower curves for the confidence interval
        qlb = (1 - self.Q) / 2
        qub = 1 - qlb
        popt_up = self.curveOpt(curvename, X, self.y, q=qub, winit = popt)
        popt_down = self.curveOpt(curvename, X, self.y, q=qlb, winit = popt)

        return popt, popt_up, popt_down

    def predict(self, curvename, x, xcols, popt, popt_up, popt_down):
        """Predicts the dependent variable using the fitted curve and the confidence interval."""
        X = x[xcols]
        ypred = self.curve(X, popt, curvename)
        yqlb = self.curve(X, popt_down, curvename)
        yqub = self.curve(X, popt_up, curvename)
        return ypred, yqlb, yqub, X
    
    def plot_curve_with_confidence_interval(self, ypred, yqlb=None, yqub=None, ax = None, title =''):
        """Plots the fitted curve along with the confidence interval."""
        # Plot the data
        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.scatter(self.x_raw[self.main_indep_var], self.y, label='Data')
        ax.plot(self.x_raw[self.main_indep_var], ypred, 'r-', label='Predicted')
        ax.set_xlabel(self.main_indep_var_name)
        ax.set_ylabel(self.dep_var_name)
        ax.set_title(title)

        if (yqlb is not None) and (yqub is not None):
            ax.fill_between(self.x_raw[self.main_indep_var], yqlb,
                            yqub, color='gray', alpha=0.5, label=f'{int(self.Q*100)}% CI')            
        ax.legend()
        return ax

    def fit_best_curves(self):
        if self.varselect_result.empty:
            self.variable_select(verbose=False)
        
        if self.parallel:
            tasks = [(curvename, self.varselect_result.loc[self.varselect_result['Curve'] == curvename, 'Best Combination'].iloc[0]) for curvename in self.varselect_result['Curve'].unique()]
            _best_params = {}
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(self.fit_curve, task[0], task[1]) for task in tasks]
                
                for future, task in zip(futures, tasks):
                    curvename, best_combination = task
                    try:
                        popt, popt_up, popt_down = future.result()
                        _best_params[curvename] = {'vars': best_combination, 'avg': popt, 'upper': popt_up, 'lower': popt_down}
                    except Exception as exc:
                        print(f'{curvename} curve fitting generated an exception: {exc}')
        else: 
            fitted_curves = self.varselect_result['Curve'].unique()
            _best_params = {}
            for curvename in fitted_curves:
                best_combination = self.varselect_result.loc[self.varselect_result['Curve'] == curvename, 'Best Combination'].values[0]
                popt, popt_up, popt_down = self.fit_curve(curvename, best_combination)
                _best_params[curvename] = {'vars':best_combination, 'avg': popt, 'upper':popt_up, 'lower':popt_down}

        self.best_curves_params = _best_params
        return _best_params
    
    def predict_best_curve(self, curvename, X=None):
        if self.best_curves_params is None:
            self.fit_best_curves()
        if X is None:
            X = self.x.copy()
        else:
            # normalize the data
            X = X / (self.x_max - self.x_min)

        _params = self.best_curves_params[curvename]       
        ypred, yqlb, yqub, X = self.predict(curvename, X, _params['vars'], _params['avg'], _params['upper'], _params['lower'])
        
        # transform back X to the original scale
        X = X * (self.x_max[X.columns] - self.x_min[X.columns])
        prediction = pd.concat([X, pd.DataFrame({'ypred': ypred, 'ypred_upper': yqlb, 'ypred_lower': yqub}, index=X.index)], axis=1)
        return prediction
    
    def most_probable_value(self, curvename, known_value, verbose=True):
        """
        Estimates the most probable value of a dependent or independent variable using a fitted curve,
        based on a provided known value. 
        This method performs linear interpolation between the closest data points or uses boundary values
        if the known value exceeds the available data range.

        This function is particularly useful when you have a value of one variable and need to estimate
        the corresponding value of another variable based on the relationship defined by the specified curve.

        Parameters:
        -----------
            curvename : str
                The name of the curve used to obtain the prediction. This curve should already be fitted
                and should correspond to one of the curves supported by the system.
            
            known_value : tuple
                A tuple containing:
                - The name of the variable (str) for which the value is known.
                - The known value (float) of that variable.

                The variable name must be one of the dependent or main independent variables used in the model.

        Returns:
        --------
            float
                The estimated value of the other variable. This could be an interpolated value between data points
                or a boundary value if the known value is outside the range of the available data.
        """
        dep_var = self.dep_var
        main_indep_var = self.main_indep_var
        assert known_value[0] in [dep_var, main_indep_var], "known_value[0] must be either dep_var or main_indep_var"
        
        # Fetch the predictions and adjust column names
        predictions_df = self.predict_best_curve(curvename=curvename, X=None)
        predictions_df.rename(columns={'ypred': dep_var}, inplace=True)
        
        # Sort DataFrame by the variable of interest
        predictions_df.sort_values(by=known_value[0], inplace=True)
        
        # Find rows where the known value is just above and below the input value
        filter_above = predictions_df[known_value[0]] >= known_value[1]
        filter_below = predictions_df[known_value[0]] <= known_value[1]
        
        if not filter_above.any():
            # If known value is above all available data points
            boundary_value = predictions_df.iloc[-1]
            if verbose:
                print(f"Known value {known_value[1]} is above the highest available value. Using boundary value.")
            if known_value[0] == main_indep_var:
                to_return = boundary_value[dep_var]
            else:
                to_return = boundary_value[main_indep_var]
        elif not filter_below.any():
            # If known value is below all available data points
            boundary_value = predictions_df.iloc[0]
            if verbose:
                print(f"Known value {known_value[1]} is below the lowest available value. Using boundary value.")
            if known_value[0] == main_indep_var:
                to_return = boundary_value[dep_var]
            else:
                to_return = boundary_value[main_indep_var]
        
        above = predictions_df[filter_above].iloc[0]
        below = predictions_df[filter_below].iloc[-1]
        
        x_above, y_above = above[main_indep_var], above[dep_var]
        x_below, y_below = below[main_indep_var], below[dep_var]

        # Perform linear interpolation
        if known_value[0] == main_indep_var:
            y = y_below + (y_above - y_below) * (known_value[1] - x_below) / (x_above - x_below)
            to_return = y
        else:
            x = x_below + (x_above - x_below) * (known_value[1] - y_below) / (y_above - y_below)
            to_return = x
        
        if verbose:
            imputname = self.dep_var_name if known_value[0] == dep_var else self.main_indep_var_name
            print(f"Estimated value of {imputname} at {known_value[1]}: {to_return}")
        return to_return
        
    def compare_best_curves(self, plot=True, CI = True):
        if self.best_curves_params is None:
            self.fit_best_curves()
                
        fitted_curves = self.best_curves_params.keys()
        predictions_d = {}
        params_d = {}
        perf_metrics_d = {}
        for curvename in fitted_curves:
            _params = self.best_curves_params[curvename]       
            params_d[curvename] = pd.concat([pd.Series(self.get_params(curvename, _params['avg'], _params['vars'])), 
                                            pd.Series(self.get_params(curvename, _params['upper'], _params['vars'])),
                                            pd.Series(self.get_params(curvename, _params['lower'], _params['vars']))
                                            ], axis=1, keys = ['avg', 'upper bound', 'lower bound'])
            predictions_d[curvename] = self.predict_best_curve(curvename)
            ypred, yqlb, yqub = predictions_d[curvename]['ypred'], predictions_d[curvename]['ypred_lower'], predictions_d[curvename]['ypred_upper']
            perf_metrics_d[curvename] = perf_metrics(self.y, ypred, yqlb, yqub, self.Q)
        
        if plot:
            curves_l = list(predictions_d.keys())
            nplots = len(curves_l) 
            ncols = 2
            nrows = int(np.ceil(nplots/ncols))
            plt.subplots(nrows,ncols, figsize = (ncols*5, nrows*5))
            for i, curvename in enumerate(curves_l):
                ax = plt.subplot(nrows,ncols, i+1)
                preds = predictions_d[curvename]
                if CI:
                    self.plot_curve_with_confidence_interval(preds['ypred'], preds['ypred_lower'], preds['ypred_upper'], ax, title=curvename) 
                else :
                    self.plot_curve_with_confidence_interval(preds['ypred'], ax=ax, title=curvename)
            plt.tight_layout()
            plt.show()

        perf_metrics_df = pd.concat(perf_metrics_d, axis=1)
        param_df = pd.concat(params_d)
        predictions_df = pd.concat(predictions_d, axis=1)    
        return perf_metrics_df, predictions_df, param_df
    
    # Helper functions
    def gX(self, x, b):
        """Computes the dot product of x and b for the curve calculations."""
        return np.dot(x, b)
        
    # Various curves follow, call them through curve()
    def logistic(self, w, g):
        alpha, beta, kappa = w[:3]
        return alpha + kappa / (1 - beta * np.exp(g))

    def redLogistic(self, w, g):
        alpha, beta = w[:2]
        return alpha + 1 / (1 - beta * np.exp(g))        

    def fixLogistic(self, w, g):
        alpha = w[0]
        return alpha + 1 / (1 - np.exp(g))

    def doubleExp(self, w, g):
        alpha, beta, rho = w[:3]
        return alpha + beta * np.exp(rho * np.exp(g))

    def exponential(self, w, g):
        alpha, beta = w[:2]
        return alpha + beta * np.exp(g)

    def fixExponential(self, w, g):
        beta = w[0]
        return beta * np.exp(g)

    def arctan(self, w, g):
        alpha, beta = w[:2]
        return alpha + beta * np.arctan(g)

    def linear(self, w, g):
        alpha = w[0]
        return alpha + g
