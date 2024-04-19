import pandas as pd
import numpy as np

from pygam import LinearGAM, LogisticGAM, GammaGAM, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from reserves_demand.utils import evaluate_metrics

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', font_scale=1)
plt.rc('axes', titlesize='large')    
plt.rc('axes', labelsize='large')   
plt.rc('xtick', labelsize='large')   
plt.rc('ytick', labelsize='large')   
plt.rc('legend', fontsize='large')   
plt.rc('figure', titlesize='x-large') 

class CurveNonParamFit:
    """
    This class is used to fit a non-parametric curve to a dataset using two different models:
    - Generalized Additive Models (GAM) with splines
    - Random Forest
    
    In a cross-validation setting, the class compares the performance of the two models based on a specified metric.
        - The GAM models test different model types for curve fitting (linear, logistic, gamma) and different number of splines, 
    then picks the best model.
        - The Random Forest model tests different hyperparameters (n_estimators, max_depth, min_samples_split, min_samples_leaf), 
    then picks the best model.

    Finally, the class compares the best GAM model with the best Random Forest model based on the specified metric, and chooses the best model.
    Forecasting can be done either using the best model, or the best set-up for the GAM and Random Forest model families. 
    The class provides a method to plot the best model's predictions.

    Parameters:
    -----------
    - data (pd.DataFrame): The dataset to fit the curve to.
    - dep_var (str): The dependent variable to predict.
    - main_indep_var (str): The main independent variable to use for curve fitting.
    - dep_var_name (str): The name of the dependent variable.
    - main_indep_var_name (str): The name of the main independent variable.
    - nFolds (int): The number of folds to use in cross-validation.

    Methods:
    --------
    - tune_gam(param_grid=None, verbose=False): Tune the GAM model using GridSearchCV.
    - tune_rf(param_grid=None, verbose=False): Tune the Random Forest model using GridSearchCV.
    - compare_models(on='RMSE', verbose=True, plot=True): Compare the performance of GAM and Random Forest models based on a specified metric.
    - predict_best(model, X=None): Predict the best model's output.
    - plot_x_y(title=''): Plot the dependent variable y against all the independent variables x.
    - plot_best(title='Best Model Predictions'): Plot the best model's predictions for both the Random Forest and GAM models families.

    Attributes:
    -----------
    - best_gam: The best GAM model found.
    - best_rf: The best Random Forest model found.
    - best_model: The best model found based on the specified metric.
    - best_model_type: The best model type found based on the specified metric.

    Raises:
    -------
    - AssertionError: If the data is not a DataFrame, or if the dependent or independent variables are not found in the dataset.
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 dep_var: str,
                 main_indep_var :str,
                 dep_var_name: str = None,                 
                 main_indep_var_name: str = None,
                 nFolds:int=5,
                 ):
        assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
        assert dep_var in data.columns, f"{dep_var} not found in the dataset"
        assert main_indep_var in data.columns, f"{main_indep_var} not found in the dataset"

        # Preprocessing data
        data = data.sort_values(main_indep_var).dropna().reset_index(drop=True)        
        
        # Prepare the y variable
        self._y = data[dep_var]
        self._dep_var = dep_var
        self._dep_var_name = dep_var_name if dep_var_name is not None else dep_var
        
        # Prepare the x variables
        x = data.drop(columns=[dep_var])
        
        self._main_indep_var = main_indep_var
        self._main_indep_var_name = main_indep_var_name if main_indep_var_name is not None else main_indep_var
        
        # Normalize x to make the optimization easier
        self._x_min = x.min()
        self._x_max = x.max()
        self._x_raw = x
        self._x = x  / (self._x_max - self._x_min)
        self._nFolds = nFolds
        
        # Split data
        self._X_train, self._X_val, self._y_train, self._y_val = train_test_split(
            self._x, self._y, test_size=0.2, random_state=42)

        self.best_gam = None
        self.best_rf = None
        self.best_model = None
        self.best_model_type = None
        return None 
    
    def tune_gam(self, param_grid = None, verbose=False):
        """
        A function to tune the GAM model using GridSearchCV.
        
        Parameters:
        -----------
            param_grid: dict, default=None
                A dictionary containing the hyperparameters to tune. If None, the function will test different model types,
                number of splines, and regularization parameters.
            verbose: bool, default=False
                If True, prints the best parameters found and the best score from GridSearchCV.

        Returns:
        --------
            opt_param: dict
                The optimal parameters found by GridSearchCV.
            train_metrics: dict
                The performance metrics on the training set.
            val_metrics: dict
                The performance metrics on the validation set.         
        """
        # Define parameter grid, including model_type to test different GAMs
        if param_grid is None:
            param_grid = {
                'model_type': ['linear', 'logistic', 'gamma'], 
                'n_splines': [10, 20],
                'lam': np.logspace(-3, 3, 2),
            }

        # Define a scorer, e.g., MSE for regression problems
        mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)

        # Initialize the GAMRegressor
        gam_regressor = GAMRegressor()
        
        # Setup GridSearchCV
        grid_search = GridSearchCV(estimator=gam_regressor,
                                   param_grid=param_grid,
                                   cv=self._nFolds,
                                   scoring=mse_scorer)

        grid_search.fit(self._X_train, self._y_train)
        self.best_gam = grid_search.best_estimator_
        
        opt_param = grid_search.best_params_
        train_metrics = evaluate_metrics(self._y_train, self.best_gam.predict(self._X_train))
        val_metrics = evaluate_metrics(self._y_val, self.best_gam.predict(self._X_val))

        if verbose: 
            print("Best parameters found:", grid_search.best_params_)
            print("Best score from GridSearchCV:", grid_search.best_score_)
        
        return opt_param, train_metrics, val_metrics

    def tune_rf(self, param_grid=None, verbose=False):
        """
        A function to tune the Random Forest model using GridSearchCV.

        Parameters:
        -----------
            param_grid: dict, default=None
                A dictionary containing the hyperparameters to tune. If None, the function will test different n_estimators,
                max_depth, min_samples_split, and min_samples_leaf.
            verbose: bool, default=False
                If True, prints the best parameters found and the best score from GridSearchCV.

        Returns:
        --------
            opt_param: dict
                The optimal parameters found by GridSearchCV.
            train_metrics: dict
                The performance metrics on the training set.
            val_metrics: dict
                The performance metrics on the validation set.
        """        
        # Define the parameter grid for Random Forest
        if param_grid is None:
            param_grid = {
                'n_estimators': [10, 50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                                   param_grid=param_grid, 
                                   cv=self._nFolds,
                                   n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')
        
        # Fit and find best estimator
        grid_search.fit(self._X_train, self._y_train)
        self.best_rf = grid_search.best_estimator_
        
        opt_param = grid_search.best_params_
        train_metrics = evaluate_metrics(self._y_train, self.best_rf.predict(self._X_train))
        val_metrics = evaluate_metrics(self._y_val, self.best_rf.predict(self._X_val))
        if verbose: 
            print("Best parameters found:", grid_search.best_params_)
            print("Best score from GridSearchCV:", grid_search.best_score_)
        return opt_param, train_metrics, val_metrics

    def compare_models(self, on='RMSE', verbose=True, plot=True):
        """
        A function to compare the performance of GAM and Random Forest models based on a specified metric.

        Parameters:
        -----------
            on: str, default='RMSE'
                The metric to compare the models on. Options: 'RMSE', 'MAE', 'R2'
            verbose: bool, default=True
                If True, prints the best model based on the specified metric and the comparison results.
            plot: bool, default=True
                If True, plots a comparison of the specified metric for different models.

        Returns:
        --------
            best_model_type: str
                The best model based on the specified metric.
            comparison_result: pd.DataFrame
                A DataFrame containing the comparison results of the two models based on the specified metric.
        """
        results = []
        model_types = ['GAM Splines', 'Random Forest']

        for model_type in model_types:
            if verbose:
                print('=' * 50)
                print(f"Running cross-validation for {model_type} model")
                print('=' * 50)
            
            # Perform cross-validation and gather metrics. 
            if model_type == 'GAM Splines':
                opt_param, train_metrics, val_metrics = self.tune_gam()
            elif model_type == 'Random Forest':
                opt_param, train_metrics, val_metrics = self.tune_rf()
            
            result = {
                'Model Type': model_type,
                **{f'Train {metric}': value for metric, value in train_metrics.items()},
                **{f'Validation {metric}': value for metric, value in val_metrics.items()}
            }
            results.append(result)
            
            if verbose:
                print(f"Best parameters for {model_type}: {opt_param}")
                print("Training Metrics:")
                print(train_metrics)
                print("\nValidation Metrics:")
                print(val_metrics)
                print('\n')

        # Convert results to DataFrame for easier handling
        comparison_result = pd.DataFrame(results)  

        # Identify the best model based on a specific validation metric
        best_model_type = comparison_result.sort_values(by=f'Validation {on}').iloc[0]['Model Type']
        
        # set the best models information
        self.best_model_type = best_model_type
        if best_model_type == 'GAM Splines':
            self.best_model = self.best_gam
        elif best_model_type == 'Random Forest':
            self.best_model = self.best_rf        

        if verbose:
            print('=' * 50)
            print(f"Best model based on Validation {on}: \033[1m{best_model_type}\033[0m")
            print('=' * 50)

        if plot:
            # Plot comparison of the specified metric for different models            
            metric_comparison = comparison_result.sort_values(f'Validation {on}').set_index('Model Type')[[f'Train {on}', f'Validation {on}']]
            _, ax = plt.subplots(figsize=(10, 5))
            metric_comparison.plot(kind='barh', ax=ax)
            ax.set_xlabel(on)
            ax.set_title(f'Comparison of {on} for Different Models')
            plt.show()

        return best_model_type, comparison_result     
    
    def predict_best(self, model, X=None):
        """
        A function to predict the best model's output.

        Parameters:
        -----------
            model: str
                The model to use for prediction. Options: 'GAM Splines', 'Random Forest', 'Best Model'.
            X: pd.DataFrame, default=None
                The data to predict on. If None, the function will use the training data.

        Returns:
        --------
            prediction: pd.DataFrame
                A DataFrame containing the predictions, upper and lower bounds of the 90% confidence interval.
        """
        assert model in ['GAM Splines', 'Random Forest', 'Best Model'], "model must be 'GAM Splines', 'Random Forest', or 'Best Model'"
        if not hasattr(self, 'best_model'):
            self.compare_models()
        if X is None:
            X = self._x.copy()
        else:
            # normalize the data
            X = X / (self._x_max - self._x_min)
        
        if model == 'Best Model': 
            model = self.best_model_type

        if model == 'GAM Splines':
            model = self.best_gam
            ypred = model.predict(X)
            gam_intervals = model.get_pygam_model().prediction_intervals(X, width=0.9)
            yqlb, yqub = gam_intervals[:, 0], gam_intervals[:, 1]
        elif model == 'Random Forest':
            model = self.best_rf
            ypred = model.predict(X)
            rf_preds_per_tree = np.array([tree.predict(X) for tree in model.estimators_])
            yqlb = np.percentile(rf_preds_per_tree, 5, axis=0)
            yqub = np.percentile(rf_preds_per_tree, 95, axis=0)
        
        prediction = pd.concat([X, pd.DataFrame({'ypred': ypred, 'ypred_upper': yqlb, 'ypred_lower': yqub}, index=X.index)], axis=1)
        return prediction 

    def plot_x_y(self, title =''):
        """
        Plots the dependent variable y against all the independent variables x.

        Parameters:
        -----------
            title: str, default=''
                The title of the plot.

        Returns:
        --------
            None
        """
        
        nplots = self._x_raw.shape[1]
        ncols = 1 if nplots == 1 else 2
        nrows = int(np.ceil(nplots/ncols))
        _ = plt.subplots(nrows,ncols, figsize = (ncols*5, nrows*5))
        for i, col in enumerate(self._x_raw.columns):
            ax = plt.subplot(nrows,ncols, i+1)
            ax.scatter(self._x_raw[col], self._y)
            ax.set_xlabel(col)
            ax.set_ylabel(self._dep_var_name)
        if title == '':
            title = f"{self._dep_var_name} against independent variables"
        plt.suptitle(title)        
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        plt.show()
        return None
    
    def plot_best(self, title:str ='Best Model Predictions'):
        """
        A function to plot the best model's predictions for both the Random Forest and GAM models families.

        Parameters:
        -----------
            title: str, default='Best Model Predictions'
                The title of the plot.

        Returns:
        --------
            None
        """
        if not hasattr(self, 'best_model'):
            self.compare_models()
        
        _, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i, model_type in enumerate(['GAM Splines', 'Random Forest']):
                                      
            prediction = self.predict_best(model_type)
            ypred, yqlb, yqub = prediction['ypred'], prediction['ypred_lower'], prediction['ypred_upper']
            ax = axes[i]
            ax.plot(self._x_raw[self._main_indep_var], self._y, 'o', label='Data')
            ax.plot(self._x_raw[self._main_indep_var], ypred, c='r', label='Predicted')
            ax.fill_between(self._x_raw[self._main_indep_var], yqlb, yqub, color='gray', alpha=0.2, label='90% CI')
            ax.set_xlabel(self._main_indep_var_name)
            ax.set_ylabel(self._dep_var_name)
            ax.set_title(f'{model_type}')
            ax.legend(frameon=False)
        plt.tight_layout()
        plt.suptitle(title)
        plt.show()
        
class GAMRegressor(BaseEstimator, RegressorMixin):
    """
    A wrapper class to define a Generalized Additive Model (GAM) regressor using pyGAM.

    Parameters:
    -----------
    - model_type: str, default='linear'
        The type of model to use for curve fitting. Options: 'linear', 'logistic', 'gamma'.
    - n_splines: int, default=25
        The number of splines to use in the model.
    - lam: float, default=0.6
        The regularization parameter to use in the model.

    Methods:
    --------
    - fit(X, y): Fit the GAM model to the data.
    - predict(X): Predict the output using the fitted model.
    - get_params(): Get the parameters of the model.
    - get_pygam_model(): Get the pyGAM model.
    - set_params(**parameters): Set the parameters of the model.

    Raises:
    -------
    - AssertionError: If the model type is not one of 'linear', 'logistic', or 'gamma'.
    """
    def __init__(self, model_type='linear', n_splines=25, lam=0.6):
        self._model_type = model_type
        self._n_splines = n_splines
        self._lam = lam
        self._model = None

    def fit(self, X, y):
        """
        Fit the GAM model to the data.

        Parameters:
        -----------
            X: np.array
                The independent variables.
            y: np.array
                The dependent variable.

        Returns:
        --------
            self: GAMRegressor
                The fitted GAM model.
        """
        if self._model_type == 'linear':
            self._model = LinearGAM(s(0, n_splines=self._n_splines), lam=self._lam).fit(X, y)
        elif self._model_type == 'logistic':
            self._model = LogisticGAM(s(0, n_splines=self._n_splines), lam=self._lam).fit(X, y)
        elif self._model_type == 'gamma':
            self._model = GammaGAM(s(0, n_splines=self._n_splines), lam=self._lam).fit(X, y)
        # Add additional model types as needed
        return self

    def predict(self, X):
        """
        Predict the output using the fitted model.

        Parameters:
        -----------
            X: np.array
                The independent variables.

        Returns:
        --------
            ypred: np.array
                The predicted output.
        """
        return self._model.predict(X)

    def get_params(self, deep=True):
        """
        Get the parameters of the model.
        """
        return {"model_type": self._model_type, "n_splines": self._n_splines, "lam": self._lam}

    def get_pygam_model(self):
        """
        Get the pyGAM model.
        """
        return self._model
    
    def set_params(self, **parameters):
        """
        Set the parameters of the model.
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
