import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pygam import LinearGAM, LogisticGAM, GammaGAM, s
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

from reserves_demand.utils import evaluate_metrics

class CurveNonParamFit:
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 dep_var: str,
                 main_indep_var :str,
                 dep_var_name: str = None,                 
                 main_indep_var_name: str = None,
                 nFolds=5,
                 ):
        assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
        assert dep_var in data.columns, f"{dep_var} not found in the dataset"
        assert main_indep_var in data.columns, f"{main_indep_var} not found in the dataset"

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
        self.x = x  / (self.x_max - self.x_min)
        self.nFolds = nFolds
        
        # Split data
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.x, self.y, test_size=0.2, random_state=42)

        self.best_gam = None
        self.best_rf = None
        self.best_model = None
        return None 
    
    def tune_gam(self, param_grid = None, verbose=False):
        
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
                                   cv=self.nFolds,
                                   scoring=mse_scorer)

        grid_search.fit(self.X_train, self.y_train)
        self.best_gam = grid_search.best_estimator_
        
        opt_param = grid_search.best_params_
        train_metrics = evaluate_metrics(self.y_train, self.best_gam.predict(self.X_train))
        val_metrics = evaluate_metrics(self.y_val, self.best_gam.predict(self.X_val))

        if verbose: 
            print("Best parameters found:", grid_search.best_params_)
            print("Best score from GridSearchCV:", grid_search.best_score_)
        
        return opt_param, train_metrics, val_metrics

    def tune_rf(self, param_grid=None, verbose=False):
        
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
                                   cv=self.nFolds,
                                   n_jobs=-1, verbose=0,
                                   scoring='neg_mean_squared_error')
        
        # Fit and find best estimator
        grid_search.fit(self.X_train, self.y_train)
        self.best_rf = grid_search.best_estimator_
        
        opt_param = grid_search.best_params_
        train_metrics = evaluate_metrics(self.y_train, self.best_rf.predict(self.X_train))
        val_metrics = evaluate_metrics(self.y_val, self.best_rf.predict(self.X_val))
        if verbose: 
            print("Best parameters found:", grid_search.best_params_)
            print("Best score from GridSearchCV:", grid_search.best_score_)
        return opt_param, train_metrics, val_metrics

    def compare_models(self, on='RMSE', verbose=True, plot=True):
        """
        Compares models (GAM splines and Random Forest) on the same train-validation set,
        picks the best parameters for each, and then compares the two models based on the specified metric.
        Returns the performance metrics for each model type and identifies the best model.
        """
        results = []
        model_types = ['GAM Splines', 'Random Forest']

        for model_type in model_types:
            if verbose:
                print('=' * 50)
                print(f"Running cross-validation for {model_type} model")
                print('=' * 50)
            
            # Perform cross-validation and gather metrics. This is a placeholder,
            # replace with your actual cross-validation logic for GAM and Random Forest.
            # Example:
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
                print("Training Metrics:", train_metrics)
                print("Validation Metrics:", val_metrics)
                print('\n')

        comparison_result = pd.DataFrame(results)  # Convert results to DataFrame for easier handling

        # Identify the best model based on a specific validation metric
        best_model_type = comparison_result.sort_values(by=f'Validation {on}').iloc[0]['Model Type']
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
            metric_comparison = comparison_result.set_index('Model Type')[[f'Train {on}', f'Validation {on}']].sort_values(f'Validation {on}')
            _, ax = plt.subplots(figsize=(10, 5))
            metric_comparison.plot(kind='barh', ax=ax)
            ax.set_xlabel(on)
            ax.set_title(f'Comparison of {on} for Different Models')
            plt.show()

        return best_model_type, comparison_result     

    def plot_best(self):
        if not hasattr(self, 'best_model'):
            self.compare_models()
        
        _, axes = plt.subplots(1, 2, figsize=(15, 5))
        for i, model_type in enumerate(['GAM Splines', 'Random Forest']):
                                      
            prediction = self.predict_best(model_type)
            ypred, yqlb, yqub = prediction['ypred'], prediction['ypred_lower'], prediction['ypred_upper']
            ax = axes[i]
            ax.plot(self.x_raw[self.main_indep_var], self.y, 'o', label='Data')
            ax.plot(self.x_raw[self.main_indep_var], ypred, c='r', label='Predicted')
            ax.fill_between(self.x_raw[self.main_indep_var], yqlb, yqub, color='gray', alpha=0.2, label='90% CI')
            ax.set_xlabel(self.main_indep_var_name)
            ax.set_ylabel(self.dep_var_name)
            ax.set_title(f'{model_type}')
            ax.legend(frameon=False)
        plt.tight_layout()
        plt.suptitle('Best Model Predictions')
        plt.show()

    def predict_best(self, model, X=None):
        assert model in ['GAM Splines', 'Random Forest', 'Best Model'], "model must be 'GAM Splines', 'Random Forest', or 'Best Model'"
        if not hasattr(self, 'best_model'):
            self.compare_models()
        if X is None:
            X = self.x.copy()
        else:
            # normalize the data
            X = X / (self.x_max - self.x_min)
            
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

        else: 
            model = self.best_model
        
        prediction = pd.concat([X, pd.DataFrame({'ypred': ypred, 'ypred_upper': yqlb, 'ypred_lower': yqub}, index=X.index)], axis=1)
        return prediction      
        


class GAMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type='linear', n_splines=25, lam=0.6):
        self.model_type = model_type
        self.n_splines = n_splines
        self.lam = lam
        self.model = None

    def fit(self, X, y):
        if self.model_type == 'linear':
            self.model = LinearGAM(s(0, n_splines=self.n_splines), lam=self.lam).fit(X, y)
        elif self.model_type == 'logistic':
            self.model = LogisticGAM(s(0, n_splines=self.n_splines), lam=self.lam).fit(X, y)
        elif self.model_type == 'gamma':
            self.model = GammaGAM(s(0, n_splines=self.n_splines), lam=self.lam).fit(X, y)
        # Add additional model types as needed
        return self

    def predict(self, X):
        return self.model.predict(X)

    def get_params(self, deep=True):
        return {"model_type": self.model_type, "n_splines": self.n_splines, "lam": self.lam}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def get_pygam_model(self):
        return self.model