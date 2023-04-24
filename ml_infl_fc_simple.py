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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, PredefinedSplit
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
from tensorflow import keras
import shap # For interpretting models

# System libraries
import os # For file management
from joblib import dump, load, delayed, Parallel # For model loading and running parallel tasks
import sys
# from statsmodels.tsa.ar_model import AutoReg
# sys.path.insert(0, 'C:/Users/Robpr/OneDrive/Documents/Projects/CodeLib')


# Paths
RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results8/"
MODELS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Models8/"
CV_RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/CVResults8/"
FEATIMP_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/FeatureImportances/"
PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions8/"


############################################
# Helper functions
############################################

def build_nn(architecture, input_shape, lr=0.001, l2=0.0, h_act='elu', kernel_init='he_normal'):
    
    
    """
    Builds a net according to specified architecture and activation function.
    
    Parameters 
    ----------
    architecture: an iterable containing the number of units in each layer. The length of the iterable determines the number of layers to add.
    activation: the activation function to use for each hidden_layer, 'elu' by default.
    
    Returns
    ----------
    model: a keras model object.
    """
    
    model = keras.models.Sequential()
    
    model.add(keras.Input(shape=input_shape))
    batch_standardize = False         
    
    n_layers = len(architecture)
    if n_layers >= 2:
        batch_standardize=True # for nets with two or more layers we apply batch normalization to avoid exploding/vanishing gradients
        
    layer=0
    for units in architecture:
        layer += 1
        model.add(keras.layers.Dense(units, 
                                     activation=h_act, 
                                     kernel_initializer=kernel_init,
                                     bias_initializer='zeros',
                                     kernel_regularizer=keras.regularizers.l2(l2)
                                     ))
        if (batch_standardize) and (layer != len(architecture)):
            model.add(keras.layers.BatchNormalization())
        
            
    model.add(keras.layers.Dense(1, activation='linear'))
        
    optimizer=keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model


def build_lstm(architecture, input_shape, lr=0.1):

    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(architecture[0], input_shape=input_shape))
    for units in architecture[1:]:
        model.add(keras.layers.LSTM(units))

    model.add(keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


def oos_r2(y_true, y_pred, y_train):

    mse = mean_squared_error(y_true, y_pred)
    mse_trainmean = mean_squared_error(y_true, np.full_like(y_true, np.mean(y_train)))
    r2 = 1 - mse/mse_trainmean

    return r2

def mean_absolute_deviation(y_true, y_pred):

    e = y_true - y_pred
    mad = np.median(  np.abs(e - np.median(e) ) )

    return mad

############################################
# Load, split, and scale data
############################################


h = sys.argv[2]
target = 'infl_tp' + str(h)

# IMPORT data
X = pd.read_csv('inflfc_features_lags4_ar.csv')
y = pd.read_csv('inflfc_targets.csv')

# AR(p) features
X_ar = pd.read_csv('ar_terms.csv')

X['date'] = pd.to_datetime(X['date'], format='%Y-%m-%d')
X_ar['date'] = pd.to_datetime(X_ar['date'], format='%Y-%m-%d')
y['date'] = pd.to_datetime(y['date'], format='%Y-%m-%d')

X = X.set_index('date')
X_ar = X_ar.set_index('date')
y = y.set_index('date')

# X = pd.concat([X,X_ar], axis=1)

# Select target variable
y = y[target].to_frame()

# Split data

# Define cutoff dates
validation_years = 1
train_cutoff_year = sys.argv[1]
train_cutoff = train_cutoff_year + '-12-01'
train_start = '1965-01-01'
train_end = datetime.strptime(train_cutoff, '%Y-%m-%d') - relativedelta(years=validation_years)
val_start = train_end + relativedelta(months=1)
val_end = val_start + relativedelta(years=validation_years-1, months=11)
test_start = val_end + relativedelta(months=1)
test_end = test_start + relativedelta(months=11)

print(train_start, train_end, val_start, val_end, test_start, test_end)

# Split data
X_train = X.loc[train_start:train_end]
X_val = X.loc[val_start:val_end]
X_test = X.loc[test_start:test_end]

X_ar_train = X_ar.loc[train_start:val_end]
X_ar_test = X_ar.loc[test_start:test_end]

y_train = y.loc[train_start:train_end]
y_val = y.loc[val_start:val_end]
y_test = y.loc[test_start:test_end]

# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 

# Scale data
scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(scaler.transform(X_test),columns=X_test.columns, index=X_test.index)

# No need to do this for AR(p) data since OLS coefficients are scale invariant.


# Random walk model y_{t} = y_{t-1} + e_{t}, i.e., the best we can do is use last periods y as our forecast
# For each horizon, h, we use the h-period inflation from h periods ago (which we would observe at time t) as the prediction. E.g., To forecast the 12 month inflation, we use the 
# previous month's recorded 12-month inflation we use the 12-mo inflation we would have observed today, which is the inflation 12 rows before
# since y_t = log(P_t+h) - log(P_t)
rw_preds = X_ar_test['infl_tm0'].values.reshape(-1,)

############################################
# Train and evaluate models
############################################

# make sure these sum up to number of cpus per task
n_jobs_models = 2
n_jobs_gridsearch = -1
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso': Lasso(),
    'Elastic Net': ElasticNet(),
    'Random Forest': RandomForestRegressor(random_state=1, max_features=1/3, n_jobs=n_jobs_models),
    'Extremely Randomized Trees': ExtraTreesRegressor(random_state=1,max_features=1/3, n_jobs=n_jobs_models),
    'Gradient Boosted Trees': GradientBoostingRegressor(random_state=1, max_features=1/3)
}

model_param_grids = {
    'Linear Regression': {},
    'Ridge Regression': {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]},
    'Lasso': {'alpha': [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]},
    'Elastic Net': {'alpha': [0.0001, 0.001, 0.1, 1.0, 10.0, 100.0, 1000.0], 'l1_ratio': np.linspace(0.1,0.9,9)},
    'Random Forest': {'n_estimators':[100, 500, 1000, 2000, 5000], 'max_depth':list(range(1, 40+2, 5))},
    'Extremely Randomized Trees': {'n_estimators':[100, 500, 1000, 2000, 5000], 'max_depth':list(range(1, 40+2, 5))},
    'Gradient Boosted Trees': {'learning_rate' : [0.1, 0.01, 0.001],'n_estimators' : [50, 100, 200, 500, 1000],'subsample' : [0.3, 0.4, 0.5, 0.6, 0.7],'max_depth' : list(range(1, 10+2, 2))}
}

# 10 years of validation
# tscv = TimeSeriesSplit(n_splits=10, test_size=12, gap=0)

# Neural Networks
print('Neural networks')

architectures = [
    # [32], 
    # [32,16], 
    # [32,16,8],
    # [32,16,8,4],
    [32,16,8,4,2]
    # [64],
    # [64,32],
    # [64,32,16],
    # [64,32,16,8],
    [64,32,16,8,4],
    [128,64,32,16,8],
    [256,128,64,32,16]
    # [512,256,128,64,32],
    # [512,256,128,64,32,16,8,4]
]

epochs = 500
batch_size=16
input_shape = X_train.shape[1:]
nn_preds_df = pd.DataFrame()
for architecture in architectures:
    str_architecture = '-'.join([str(i) for i in architecture])
    model = 'NN' + str_architecture
    model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.h5'
    
    if not os.path.isfile(model_path):
        print('No existing model found, training...')

        cbs = [
            keras.callbacks.EarlyStopping(monitor='val_mae', min_delta=10**-4, patience=50, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_mae', min_delta=10**-4, patience=10)
        ]
        nn = build_nn(
            input_shape=X_train.shape[1:],
            architecture=architecture,
            lr=0.1,
            h_act='relu',
            kernel_init='glorot_uniform'
        )

        nn.fit(
            X_train, y_train, 
            validation_data=(X_val, y_val),
            epochs = epochs,
            batch_size = batch_size,
            callbacks = cbs,
            verbose=1
        )

        nn.save(model_path)

    else:
        # load model
        print('Existing model found, loading...')
        nn = keras.models.load_model(model_path)

    # Generate predictions
    print('Generating predictions ...')
    nn_pred = nn.predict(X_test, verbose=0).reshape(-1,)
    nn_pred_df = pd.DataFrame({model:nn_pred}, index=X_test.index)
    nn_preds_df = pd.concat([nn_preds_df, nn_pred_df], axis=1)

# Ensembled NN
# n_ensembles = 40
# model = 'NN_Ensemble' + n_ensembles
# model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.h5'
# if not os.path.isfile(model_path):
#     models = []
    
#     for architecture in architectures 
#     # load cv results of NN
#     for i in range(nemsembles):
#         cbs = [
#             keras.callbacks.EarlyStopping(monitor='val_mae', min_delta=10**-4, patience=15, restore_best_weights=True),
#             keras.callbacks.ReduceLROnPlateau(monitor='val_mae', min_delta=10**-4, patience=5)
#         ]
#         nn = build_nn(
#             input_shape=X_train.shape[1:],
#             architecture=architecture,
#             lr=0.1,
#             h_act='relu',
#             kernel_init='glorot_uniform'
#         )

#         nn.fit(
#             X_train, y_train, 
#             validation_data=(X_val, y_val),
#             epochs = epochs,
#             batch_size = batch_size,
#             callbacks = cbs,
#             verbose=0
#         )
#         models.append(nn)
    
#     model_input = keras.Input(shape=input_shape)
#     model_outputs = [model(model_input) for model in models]
#     ensemble_output = keras.layers.Average()(model_outputs)
#     nn = keras.Model(inputs=model_input, outputs=ensemble_output)
#     nn.save( modelpath )
# else:
#     nn_ensemble = keras.load_model(model_path)


# Other ML models
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
split_index = [-1 if date in X_train.index else 0 for date in X_train_full.index]
pds=PredefinedSplit(split_index)
y_true = y_test.values.reshape(-1,)
preds_df = pd.DataFrame()
for model in models.keys():
    
    print(f'Model:{model}')
    model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '.joblib'
    
    if not os.path.isfile(model_path):
        # Train model
        print('No existing model found, training...')
        grid_search = GridSearchCV(
            models[model], 
            model_param_grids[model], 
            refit=False, 
            cv=pds, 
            scoring='neg_root_mean_squared_error', 
            n_jobs=n_jobs_gridsearch,
            return_train_score=True,
            verbose=0
        )
        
        # Perform grid search and refit with best params
        best_fit = grid_search.fit(X_train_full, y_train_full.values.reshape(-1,))

        # Save cv results
        pd.DataFrame(best_fit.cv_results_).to_csv(CV_RESULTS_FOLDER + model + '_' + train_cutoff_year + '_' + target + '_cvresults.csv')

        
        best_params = best_fit.best_params_
        estimator = models[model].set_params(**best_params)

        # refit only on train data
        model_fit = estimator.fit(X_train, y_train)

        # Save model
        print(f'Model fit complete. Best parameters: {best_params}')
        print('Saving model...')
        dump(model_fit, model_path)

    else:
        print('Model already exists, loading...')
        # load model if already exists
        model_fit = load(model_path)
    
    # make predictions
    y_pred = model_fit.predict(X_test).reshape(-1,)
    y_pred_df = pd.DataFrame({model: y_pred}, index=X_test.index)
    preds_df = pd.concat([preds_df, y_pred_df], axis=1)

    # featimp_path = FEATIMP_FOLDER + model + '_' + train_cutoff_year + '_' + target + '_shap.csv'
    # if not os.path.isfile(featimp_path):
    #     print('No existing SHAP values found, generating SHAP values...')
    #     if model == 'Neural Network':
    #         explainer = shap.SamplingExplainer(best_fit.model.predict, X_test)
    #         shap_vals = pd.DataFrame(explainer.shap_values(X_test), columns=X_test.columns, index=X_test.index)
    #         shap_vals.to_csv(featimp_path)
    #     else:
    #         explainer = shap.SamplingExplainer(best_fit.predict, X_test)
    #         shap_vals = pd.DataFrame(explainer.shap_values(X_test), columns=X_test.columns, index=X_test.index)
    #         shap_vals.to_csv(featimp_path)


# Random walk model
preds_df['RW'] = rw_preds

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
    
    ar_p.fit(X_train_p, y_train_full)
    
    ar_p_pred = ar_p.predict(X_train_p)
    n = X_train_p.shape[0]
    mse = mean_squared_error(y_train_full, ar_p_pred)
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
ar_p.fit(X_train_p, y_train_full)

# save model
dump(ar_p, model_path)

# generate predictions
ar_preds = ar_p.predict(X_test_p).reshape(-1,)

preds_df['AR_p'] = ar_preds

# evaluate model
all_preds = pd.concat([preds_df, nn_preds_df], axis=1)

# Save results
print('Saving predictions...')
all_preds.to_csv(PRED_FOLDER + 'all_models_' + target + '_' + train_cutoff_year + '_predictions.csv')
print('Complete.')