############################################
# Import Libraries
############################################


# Data Manipulation Libraries
import pandas as pd 
import numpy as np
from numpy.random import seed # for reproducible results
seed(1)
from datetime import datetime
from dateutil.relativedelta import relativedelta # may need to pip install python-dateutil
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
# import shap # For interpretting models

# System libraries
import os # For file management
from joblib import dump, load # For model loading and running parallel tasks
import sys
# from statsmodels.tsa.ar_model import AutoReg
# sys.path.insert(0, 'C:/Users/Robpr/OneDrive/Documents/Projects/CodeLib')

RUN_LOCALLY = False

if RUN_LOCALLY:
    RESULTS_FOLDER = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Results/'
    MODELS_FOLDER = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Models/'
    CV_RESULTS_FOLDER = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/CVResults/'
    FEATIMP_FOLDER = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/FeatureImportances/'
    PRED_FOLDER = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Predictions/'
    DATA_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Data/'

else:
    RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results1/"
    MODELS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Models1/"
    CV_RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/CVResults1/"
    FEATIMP_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/FeatureImportances/"
    PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions1/"
    DATA_DIR = '/lustre06/project/6070422/rproner/InflationForecasting/Data/'


############################################
# Helper functions
############################################

def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


############################################
# Load, split, and scale data
############################################

h = sys.argv[2]
target = 'infl_tp' + str(h)

# IMPORT data
X = pd.read_csv(DATA_DIR + 'inflfc_features.csv')

y = pd.read_csv(DATA_DIR + 'inflfc_targets.csv')

# autoregressive terms
X_ar = pd.read_csv(DATA_DIR + 'ar_terms.csv')
# X_unrate_ar = pd.read_csv(DATA_DIR + 'ar_terms_unrate.csv', index_col='date')


X['date'] = pd.to_datetime(X['date'], format='%Y-%m-%d')
X_ar['date'] = pd.to_datetime(X_ar['date'], format='%Y-%m-%d')
y['date'] = pd.to_datetime(y['date'], format='%Y-%m-%d')
# X_unrate_ar.index = pd.to_datetime(X_unrate_ar.index)

X = X.set_index('date')
X_ar = X_ar.set_index('date')
y = y.set_index('date')

# Concatenate features and AR terms
# X = pd.concat([X,X_ar,X_unrate_ar], axis=1)
X = pd.concat([X,X_ar], axis=1)

# Select target variable
y = y[target].to_frame()

# Split data

# Define cutoff dates
train_start = '1965-01-01'
train_cutoff_year = sys.argv[1]
train_cutoff = train_cutoff_year + '-12-01'
test_start = datetime.strptime(train_cutoff, '%Y-%m-%d') + relativedelta(months=1)
test_finish = test_start + relativedelta(months=11)

# Split data
X_train = X.loc[train_start:train_cutoff]
X_test = X.loc[test_start:test_finish]
X_ar_train = X_ar.loc[train_start:train_cutoff]
X_ar_test = X_ar.loc[test_start:test_finish]

y_train = y.loc[train_start:train_cutoff]
y_test = y.loc[test_start:test_finish]

# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 

# Scale data
# scaler = StandardScaler()
s = StandardScaler()

X_train = pd.DataFrame(s.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(s.transform(X_test),columns=X_test.columns, index=X_test.index)

# Random walk model y_{t} = y_{t-1} + e_{t}, i.e., the best we can do is use last periods y as our forecast
rw_preds = X_ar_test['infl_tm0'].values.reshape(-1,)

############################################
# Train and evaluate models
############################################

tscv = TimeSeriesSplit(n_splits=5, gap=0)

# make sure these sum up to number of cpus per task
n_jobs_models = 2
n_jobs_gridsearch = -1

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso': Lasso(),
    'Elastic Net': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=1,max_features='sqrt', n_jobs=n_jobs_models),
    'Extremely Randomized Trees': ExtraTreesRegressor(random_state=1,max_features='sqrt', n_jobs=n_jobs_models),
    'Gradient Boosted Trees': GradientBoostingRegressor(random_state=1, max_features='sqrt')
}
model_param_grids = {
    'Linear Regression': {},
    'Ridge Regression': {'alpha': [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'Lasso': {'alpha': [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'Elastic Net': {'alpha': [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0], 'l1_ratio': np.linspace(0.1,0.9,9)},
    'Random Forest': {'n_estimators':[100, 200, 500, 1000], 'max_depth':list(range(1, 40+2, 5))},
    'Extremely Randomized Trees': {'n_estimators':[100, 200, 500, 1000], 'max_depth':list(range(1, 40+2, 5))},
    'Gradient Boosted Trees': {'learning_rate' : [0.1, 0.01, 0.001],'n_estimators' : [20, 50, 100, 200] ,'subsample' : [0.25, 0.5, 1.0],'max_depth' : list(range(1, 10+2, 2))}
}

# Other ML models
y_true = y_test.values.reshape(-1,)
predictions = pd.DataFrame()
for model in models.keys():
    
    print(f'Model:{model}')

    model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.joblib'
    if model == 'Neural Network':
        model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.h5'
    
    if not os.path.isfile(model_path):
        # Train model
        print('No existing model found, training...')
        grid_search = GridSearchCV(
            models[model], 
            model_param_grids[model], 
            refit=True, 
            cv=tscv, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=n_jobs_gridsearch,
            return_train_score=True,
            verbose=0
        )
        
        # Perform grid search and refit with best params
        best_fit = grid_search.fit(X_train, y_train.values.reshape(-1,))

        # Save cv results
        # pd.DataFrame(best_fit.cv_results_).to_csv(CV_RESULTS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '_cvresults.csv')

        # Save model
        best_params = best_fit.best_params_
        print(f'Model fit complete. Best parameters: {best_params}')
        print('Saving model...')
        dump(best_fit, model_path)

    else:
        print('Model already exists, loading...')
        best_fit = load(model_path)
    
    y_pred = best_fit.predict(X_test).reshape(-1,)
    y_pred_df = pd.DataFrame({model: y_pred.reshape(-1,)}, index=y_test.index)
    predictions = pd.concat([predictions, y_pred_df], axis=1)


# AR(p) model
print('AR(p) model')
model = 'AR_p'
model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.joblib'
ar_p = LinearRegression()
aics = {
    'p': [],
    'AIC': []
}
for p in range(12+1):
    # fit AR model with up to p lags
    ar_model = 'AR_'+ str(p)
    X_train_p = X_ar_train.loc[:, 'infl_tm0':('infl_tm' + str(p))]
    
    ar_p.fit(X_train_p, y_train)
    
    ar_p_pred = ar_p.predict(X_train_p)
    n = X_train_p.shape[0]
    mse = mean_squared_error(y_train, ar_p_pred)
    n_params = p + 1
    aic = calculate_aic(n, mse, n_params)
    aics['p'].append(p)
    aics['AIC'].append(aic)

best_ar_model_ind = aics['AIC'].index(max(aics['AIC']))
best_p = aics['p'][best_ar_model_ind]
# print(f'Best AR(p) model:{str(p)}. AIC: {aics['AIC'][best_ar_model_ind]}')
pd.DataFrame(aics).to_csv(CV_RESULTS_FOLDER + 'AR_p' + '_' + train_cutoff_year + '_' + target + '_cvresults.csv')

# initialize model
ar_p = LinearRegression()

# Train Model according to best p
X_train_p = X_ar_train.loc[:, 'infl_tm0':('infl_tm' + str(best_p))]
X_test_p = X_ar_test.loc[:, 'infl_tm0':('infl_tm' + str(best_p))]
ar_p.fit(X_train_p, y_train)

# save model
dump(ar_p, model_path)

# generate predictions
ar_preds = ar_p.predict(X_test_p).reshape(-1,)

# save predictions
predictions['AR_p'] = ar_preds
predictions['RW'] = rw_preds

# Historical mean
dummy = DummyRegressor()
dummy.fit(X_train, y_train)
predictions['HistMean'] = dummy.predict(X_test).ravel()

# Save results
print('Saving predictions...')
predictions.to_csv(PRED_FOLDER + 'all_models_' + target + '_' + train_cutoff_year + '_predictions.csv')
print('Complete.')