import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from datetime import datetime
from dateutil.relativedelta import relativedelta # may need to pip install python-dateutil
import warnings
warnings.filterwarnings("ignore")
import os # For file management
import sys

from joblib import load, dump


def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


# Folder paths

MODELS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Models10/"
PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions10/"
RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results10/"

h = sys.argv[2]
target = 'unrate_tp' + str(h)

# IMPORT data
X = pd.read_csv('inflfc_features.csv')
y = pd.read_csv('delta_unrate.csv')

# AR(p) features
unrate_ar = pd.read_csv('ar_terms_unrate_ad_adj.csv', index_col='date')

X['date'] = pd.to_datetime(X['date'], format='%Y-%m-%d')
unrate_ar.index = pd.to_datetime(unrate_ar.index)
y['date'] = pd.to_datetime(y['date'], format='%Y-%m-%d')

X = X.set_index('date')
y = y.set_index('date')

# Concatenate features and AR terms
X = pd.concat([X,unrate_ar], axis=1)

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

unrate_ar_train = unrate_ar.loc[train_start:train_cutoff]
unrate_ar_test = unrate_ar.loc[test_start:test_finish]

y_train = y.loc[train_start:train_cutoff]
y_test = y.loc[test_start:test_finish]

# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 

# Scale data
# scaler = StandardScaler()
s = StandardScaler()

X_train = pd.DataFrame(s.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(s.transform(X_test),columns=X_test.columns, index=X_test.index)


rw_preds = unrate_ar_test['unrate_tm0'].values.reshape(-1,)

predictions = pd.DataFrame()

n_jobs_models = -1
n_jobs_grid_search = 2
models = {
    'Elastic Net': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=1,max_features='sqrt', n_jobs=n_jobs_models)
}
model_param_grids = {
    'Elastic Net': {'alpha': [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0], 'l1_ratio': np.linspace(0.1,0.9,9)},
    'Random Forest': {'n_estimators':[100, 500, 1000, 2000, 5000], 'max_depth':list(range(1, 40+2, 5))}
}

y_true = y_test.values.reshape(-1,)
predictions = pd.DataFrame()
tscv = TimeSeriesSplit()
for model in models.keys():
    
    print(f'Model:{model}')

    model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.joblib'
    
    if not os.path.isfile(model_path):
        # Train model
        print('No existing model found, training...')
        grid_search = GridSearchCV(
            models[model], 
            model_param_grids[model], 
            refit=True, 
            cv=tscv, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=n_jobs_grid_search,
            return_train_score=True,
            verbose=0
        )
        
        # Perform grid search and refit with best params
        best_fit = grid_search.fit(X_train.values, y_train.values.reshape(-1,))

        # Save model
        best_params = best_fit.best_params_
        print(f'Model fit complete. Best parameters: {best_params}')
        print('Saving model...')
        dump(best_fit, model_path)

    else:
        print('Model already exists, loading...')
        best_fit = load(model_path)
    
    y_pred = best_fit.predict(X_test.values).reshape(-1,)
    y_pred_df = pd.DataFrame({model: y_pred}, index=y_test.index)
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
    X_train_p = unrate_ar_train.loc[:, 'unrate_tm0':('unrate_tm' + str(p))]
    
    ar_p.fit(X_train_p, y_train)
    
    ar_p_pred = ar_p.predict(X_train_p)
    n = X_train_p.shape[0]
    mse = mean_squared_error(y_train, ar_p_pred)
    n_params = p + 1
    aic = calculate_aic(n, mse, n_params)
    aics['p'].append(p)
    aics['AIC'].append(aic)

best_ar_model_ind = aics['AIC'].index(min(aics['AIC']))
best_p = aics['p'][best_ar_model_ind]

# initialize model
ar_p = LinearRegression()

# Train Model according to best p
X_train_p = unrate_ar_train.loc[:, 'unrate_tm0':('unrate_tm' + str(best_p))]
X_test_p = unrate_ar_test.loc[:, 'unrate_tm0':('unrate_tm' + str(best_p))]
ar_p.fit(X_train_p, y_train)

# save model
dump(ar_p, model_path)

# generate predictions
ar_preds = ar_p.predict(X_test_p).reshape(-1,)

# save predictions
predictions['AR_p'] = ar_preds
predictions['RW'] = rw_preds
predictions[target] = y_test.values.reshape(-1,)

# Save results
print('Saving predictions...')
predictions.to_csv(PRED_FOLDER + 'all_models_' + target + '_' + train_cutoff_year + '_predictions.csv')
print('Complete.')