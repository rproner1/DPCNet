"""
This is the final iteration of fitting the LSTMs, In this iteration I 
implement grid search and use 12 timesteps.
"""

# ******************************** Imports ********************************

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
from tensorflow import keras
# import shap # For interpretting models
from keras.layers import Dense, LSTM, Input, LayerNormalization
from keras.regularizers import L1
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model, Model, Sequential

# System libraries
import os # For file management
import sys


# ******************************** Module parameters ********************************

RUN_LOCALLY = False
TIME_STEPS = 12
N_ESTIMATORS = 10

# ******************************** Directories ********************************

# Paths
if RUN_LOCALLY:
    MODELS_DIR = "C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Models/"
    PRED_DIR = "C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Predictions/"
    RESULTS_DIR = "C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Results/"
    DATA_DIR = "C:/Users/Robpr/OneDrive/Documents/Data/Inflation/"
else:
    MODELS_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Models3.1/"
    FEATIMP_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/FeatureImportances/"
    PRED_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions3.1/"
    DATA_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Data/"


# Helper functions
def build_lstm(architecture, dense_architecture, input_shape, recurrent_dropout=0.0, l1=0.0, lr=0.001):

    # Init
    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=input_shape, name='input'))
    
    for units in architecture[:-1]:
        model.add(LSTM(units, return_sequences=True, recurrent_dropout=recurrent_dropout, kernel_regularizer=L1(l1) ))

    # Do not return sequences for the last layer
    model.add(LSTM(architecture[-1], recurrent_dropout=recurrent_dropout, kernel_regularizer=L1(l1)))

    if dense_architecture != None:
        for units in dense_architecture:
            model.add(Dense(units, activation='relu', kernel_regularizer=L1(l1)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mae'])
    
    return model


def build_lstm_ensemble(architecture, dense_architecture, input_shape, recurrent_dropout=0.0, l1=0.0, n_estimators=10):


    models = []
    
    for i in range(n_estimators):
        print(f'Now training estimator {i+1} of {n_estimators}')
        cbs = [
            keras.callbacks.EarlyStopping(monitor='val_mae',min_delta=10**(-5), patience=50, restore_best_weights=True, verbose=1)
        ]

        model = build_lstm(architecture, dense_architecture, input_shape, recurrent_dropout, l1)

        model.fit(X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=1500,
                batch_size=4,
                verbose=0,
                callbacks=cbs
        )
        models.append(model)

    model_input = keras.Input(shape=input_shape)
    model_outputs = [model(model_input) for model in models]
    ensemble_output = keras.layers.Average()(model_outputs)
    ensemble_model = keras.Model(inputs=model_input, outputs=ensemble_output)

    return ensemble_model


def DPCNet(input_shape, hardsharing_architecture, taskspecific_architecture, dense_architecture, recurrent_dropout=0.0, l1=0.0, lr=0.001):

    """
    Builds a multitask LSTM.

    Parameters
    ----------
    hardsharing_architecture: the architecture for the hardsharing layers (iterable).
    taskspecific_architectures: the architecture for each task specific group of layers (iterable of iterables).
    dense_architecture: the architecture for the additional dense layers added after LSTM layers and before output.
        If no dense layers are desired, set to None.
    ntasks: The number of tasks.
    tasknames: tasknames
    input_shape: input shape
    """
    inlayer = Input(shape=input_shape, name='input')

    # Create hardsharing layers
    hardsharing_layers = [LSTM(nodes, return_sequences=True, stateful=False, recurrent_dropout=recurrent_dropout, kernel_regularizer=L1(l1)) for nodes in hardsharing_architecture]    
    # hardsharing_layers = [layer for tup in hardsharing_layers for layer in tup]
    hardsharing = Sequential(hardsharing_layers)(inlayer)

    # Task-specific layers

    # Init output list. Each task has its own output
    outputs = []
    tasknames = ['infl', 'unrate']
    for i in range(2):

        # Create task-specific layer. Return sequence unless last LSTM layer which feeds dense layer.
        tasklayers = [ LSTM(nodes, return_sequences=True, stateful=False, recurrent_dropout=recurrent_dropout, kernel_regularizer=L1(l1)) for nodes in taskspecific_architecture[:-1]]
        tasklayers.append(LSTM(taskspecific_architecture[-1], stateful=False, recurrent_dropout=recurrent_dropout, kernel_regularizer=L1(l1)))
        
        # Add dense layers if any
        if dense_architecture != None:
            for nodes in dense_architecture:
                tasklayers.append(Dense(nodes, activation='relu', kernel_regularizer=L1(l1)))
        
        # Add output layer
        tasklayers.append(Dense(1, activation='linear'))

        # Create task-specific net
        tasknet = Sequential(tasklayers, name=tasknames[i])(hardsharing)

        # Append task-specific net to outputs list
        outputs.append(tasknet)

    # Create and compile full model
    model = Model(inputs=inlayer, outputs=outputs)
    
    model.compile(loss='mse', optimizer=Adam(learning_rate=lr), metrics=['mae'])

    return model

class GridSearch:
    
    """
    Performs grid search for a keras model.
    
    Parameters
    ----------
    estimator: estimator object
        Keras Sequential or Functional model.
    param_grid: dict
        A dictionary with hyperparameter names as keys and lists of values as values.
    verbose: bool
        If true, prints progress, current hyperparameter combination, and scores.
    refit: bool
        If true, refits a new estiamtor object with the hyperparameters that 
        acheived the minimum validation loss.
        
    Attributes
    ----------
    best_score: float
        The best validation loss acheived over all fits.
    best_params: dict
        The set of hyperparameters that acheived the best_score
    best_estimator: estimator object
        An estimator fit with the best_params.
    val_results: dict
        A dictionary containing the hyperparameter combinations and their scores.
    
    Methods
    ----------
    fit(X,y)
        
        Returns a fitted estimator object if refit=True, else None.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            training features 
        y: array-like of shape (n_samples, n_output) or (n_samples,)
            the target labels
        fit_params: dict
            a dictionary of kwargs to pass to the keras.Model.fit() method.
            Requires 'validation_data' key for scoring on the validation set.
        
        Returns
        ----------
        self.best_estimator: estimator object
            A model object refit with the best hyperparameters. Only available if
            refit=True, otherwise returns None.
            
    Dependencies:
        - sklearn.model_selection.ParameterGrid

    """
    
    def __init__(self, estimator, param_grid, verbose=False, refit=False):
        self.estimator = estimator
        self.refit = refit
        self.verbose = verbose
        self.param_grid = param_grid
        
        self.best_score = None
        self.best_params = None
        self.best_estimator = None
        self.val_results = None
        
    
    def fit(self, X, y, fit_params):    
        
        hp_combinations = ParameterGrid(self.param_grid)
      
        scores = []
        estimators=[]
        i=0
        for hps in hp_combinations:
            i+=1
            
            if self.verbose: 
                print(f'Fit {i} of {len(list(hp_combinations))}')
                print(f'Hyperparameters: {hps}')
            
            estimator = self.estimator(**hps)
      
            estimator.fit(X, y, **fit_params)
            
            # get validation loss
            score = estimator.evaluate(*fit_params['validation_data'], verbose=0)[0]
            
            scores.append(score)
            
            
            
            if score == min(scores):
                self.best_estimator = estimator
                self.best_score = score
                self.best_params = hps
            
            if self.verbose:
                print(f'Validation loss: {score}')
                print(f'Best score: {self.best_score}')
                print(f'Best score so far: {min(scores)}')
            
        
        val_results = {
            'params': list(hp_combinations),
            'scores': scores
        }
        self.val_results = val_results
      
        if self.refit:
            print('Refitting model with best parameters...')
            estimator = self.estimator(**self.best_params)
            estimator.fit(X, y, **fit_params)
            self.best_estimator = estimator
            
        return self.best_estimator

def multitask_preds_to_df(mtpreds, colnames):

    """
    Converts the output of a multi-task net to a dataframe of predictions.
    The output of the network is a list of arrays. Each array represents the
    predictions for a particular task given X_test.

    Parameters
    ----------
    mtpreds: output of the mt net (list).
    colnames: the column names of the y dataframe, i.e., the target names.

    Returns
    -------
    mtpreds_df: a dataframe where each column is the predictions for a particular target.
    """


    # Convert each preds array to series. Reshape necessary bec data must be 1D
    mtpreds = [pd.Series(pred.reshape((-1,))) for pred in mtpreds]

    # Concat all series in mtpreds list
    mtpreds_df = pd.concat(mtpreds, axis=1)

    # name columns
    mtpreds_df.columns = colnames

    return mtpreds_df


def df_to_multitask_y(df):

    mt_y = [df[col].values for col in df]

    return mt_y


def split_sequences(sequence, n_timesteps):
    
    X, y = [], []
    for i in range(len(sequence)):
        
        end_ind = i + n_timesteps
        
        if end_ind > len(sequence):
            break
        
        X_seq = sequence[i:end_ind, :-1]
        y_seq = sequence[end_ind-1, -1]
        
        X.append(X_seq)
        y.append(y_seq)
    
    return np.array(X),np.array(y)

h = sys.argv[2]
target = 'infl_tp' + str(h)
print(f'Target: {target}')

# Read data
X = pd.read_csv(DATA_DIR + 'inflfc_features.csv', index_col='date')
y = pd.read_csv(DATA_DIR + 'inflfc_targets.csv', index_col='date')
unrate = pd.read_csv(DATA_DIR + 'delta_unrate.csv', index_col='date')

X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
y.index = pd.to_datetime(y.index, format='%Y-%m-%d')
unrate.index = pd.to_datetime(unrate.index)

# Select target variable
y = y[target].to_frame()
unrate = unrate['unrate_tp' + str(h)].to_frame()

# Define cutoff dates
val_years = 5
train_cutoff_year = sys.argv[1]
train_start = '1960-03-01'
train_cutoff = train_cutoff_year + '-12-01'
train_end = datetime.strptime(train_cutoff, '%Y-%m-%d') - relativedelta(years=val_years)
val_start = train_end + relativedelta(months=1)
val_end = val_start + relativedelta(years=val_years-1, months=11)
test_start = val_end + relativedelta(months=1)
test_end = test_start + relativedelta(months=11)
test_ind = pd.date_range(test_start, test_end, freq='MS')

print(train_start, train_end, val_start, val_end, test_start, test_end)

# Split data

X_train = X.loc[train_start:train_end]
X_val = X.loc[val_start:val_end]
X_test = X.loc[test_start:test_end]

# Scale data
s = StandardScaler()
X_train = pd.DataFrame(s.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(s.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(s.transform(X_test),columns=X_test.columns, index=X_test.index)

X_test = X_test.fillna(X_test.mean())

# Reshape data for RNN
X = pd.concat([X_train, X_val, X_test])
y = y.loc[train_start:test_end]

dataset = pd.concat([X,y], axis=1)
train_dataset = dataset.loc[train_start:train_end]
val_dataset = pd.concat([train_dataset.iloc[-(TIME_STEPS-1):], dataset.loc[val_start:val_end]])
test_dataset = pd.concat([val_dataset.iloc[-(TIME_STEPS-1):],dataset.loc[test_start:test_end]])

X_train, y_train = split_sequences(train_dataset.values, TIME_STEPS)
X_val, y_val = split_sequences(val_dataset.values, TIME_STEPS)
X_test, y_test = split_sequences(test_dataset.values, TIME_STEPS)

unrate_dataset = pd.concat([X,unrate], axis=1)
unrate_train_dataset = unrate_dataset.loc[train_start:train_end]
unrate_val_dataset = pd.concat([unrate_train_dataset.iloc[-(TIME_STEPS-1):], unrate_dataset.loc[val_start:val_end]])
unrate_test_dataset = pd.concat([unrate_val_dataset.iloc[-(TIME_STEPS-1):],unrate_dataset.loc[test_start:test_end]])

_, y_unrate_train = split_sequences(unrate_train_dataset.values, TIME_STEPS)
_, y_unrate_val = split_sequences(unrate_val_dataset.values, TIME_STEPS)
_, y_unrate_test = split_sequences(unrate_test_dataset.values, TIME_STEPS)

# Multi-task data
mt_y_train = [y_train, y_unrate_train]
mt_y_val = [y_val, y_unrate_val]
mt_y_test = [y_test, y_unrate_test]


INPUT_SHAPE = X_train.shape[1:]
# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 


# LSTMs
predictions = {}
unrate_predictions = {}

e_architectures = {
    'LSTM1': {'architecture':[[32]], 'dense_architecture': [[8]*2]},
    'LSTM2': {'architecture':[[32,16]], 'dense_architecture': [[8]*2]},
    'LSTM3': {'architecture':[[32,32,16]], 'dense_architecture': [[8]*2]},
    'LSTM4': {'architecture':[[32,32,32,16]], 'dense_architecture': [[8]*2]}
}

hp_grid = {
    'recurrent_dropout': [0.0, 0.1],
    'l1': [0.0, 1e-7],
    'lr': [1e-3, 1e-2],
    'input_shape':[INPUT_SHAPE]
}


stop_early = EarlyStopping(monitor='val_loss', min_delta=10**-7, patience=50, verbose=1, restore_best_weights=True)

fit_params = {
    'validation_data': (X_val, y_val),
    'epochs': 1500,
    'batch_size':4,
    'verbose':0,
    'callbacks':[stop_early]
}

for model_name in e_architectures.keys():
    
    print(f'Training model {model_name}...')
    
    # Set hps
    model_params = e_architectures[model_name]
    hp_grid.update(model_params)
    
    # perform grid search
    search = GridSearch(build_lstm, hp_grid, verbose=True, refit=False)
    search.fit(X_train, y_train, fit_params)
    
    # Get best params
    best_params = search.best_params
    print(f'Best hyperparameters: {best_params}')
    
    
    for i in range(1,N_ESTIMATORS+1):
    
        model_path = MODELS_DIR + model_name + '/' + model_name + '_' + str(i) + '_' + target + '_' + train_cutoff_year + '.h5'
    
        if not os.path.isfile(model_path):
            print(f'Training estimator {i} of {N_ESTIMATORS}...')
            
            print(best_params)
            model = build_lstm(**best_params)
            model.fit(X_train, y_train, **fit_params)
    
            print('Model fit complete, saving model...')
            model.save(model_path)
    
        else:
            print('Existing model found, loading...')
            model = keras.models.load_model(model_path)
        
        print('Generating predictions...')
        y_pred = model.predict(X_test, verbose=0).reshape(-1,)
    
        predictions[model_name] = y_pred

mt_architectures = {
    'DPCNet1': {'hardsharing_architecture':[[32]],'taskspecific_architecture':[[32]],'dense_architecture': [[8]*2]},
    'DPCNet2': {'hardsharing_architecture':[[32]*2],'taskspecific_architecture':[[32]],'dense_architecture': [[8]*2]},
    'DPCNet3': {'hardsharing_architecture':[[32]*3],'taskspecific_architecture':[[32]],'dense_architecture': [[8]*2]},
    # 'DPCNet4': {'hardsharing_architecture':[[32]*4],'taskspecific_architecture':[[32]],'dense_architecture': [[8]*2]}
}

hp_grid = {
    'recurrent_dropout': [0.0, 0.1],
    'l1': [0.0, 1e-7],
    'lr': [1e-3, 1e-2],
    'input_shape':[INPUT_SHAPE]
}

stop_early = EarlyStopping(monitor='val_loss', min_delta=10**-7, patience=50, verbose=1, restore_best_weights=True)

fit_params = {
    'validation_data': (X_val, mt_y_val),
    'epochs': 1500,
    'batch_size':4,
    'verbose':0,
    'callbacks':[stop_early]
}

# DPCNet
for model in mt_architectures.keys():
    
    print(f'Training {model}...')

    estimator_infl_predictions = []
    estimator_unrate_predictions = []
    
    # Set hyperparameter grid
    model_params = mt_architectures[model]
    hp_grid.update(model_params)
    
    # Perform grid search
    search = GridSearch(DPCNet, hp_grid, verbose=True, refit=False)
    search.fit(X_train, mt_y_train, fit_params)
    
    # Get best params
    best_params = search.best_params
    
    # Fit ensemble with best params
    # mtlstm_estimators = []
    for i in range(1, N_ESTIMATORS+1):

        model_path = MODELS_DIR + model + '/' + model + '_' + str(i) + '_' + target + '_' + train_cutoff_year + '.h5'

        if not os.path.isfile(model_path):
            print(f'Training estimator {i} of {N_ESTIMATORS}...')

            # Build model with best params, fit, and save
            mtlstm = DPCNet(**best_params)
            mtlstm.fit(X_train, mt_y_train, **fit_params)
            
            mtlstm.save(model_path)
        
            # mtlstm_estimators.append(mtlstm)

        else:
            print('Loading existing model...')
            mtlstm = load_model(model_path)

            # mtlstm_estimators.append(mtlstm)

        mtlstm_pred = multitask_preds_to_df(mtlstm.predict(X_test, verbose=0), colnames=[target, 'unrate'])
        estimator_infl_predictions.append(mtlstm_pred[target].values.reshape(-1,1))
        estimator_unrate_predictions.append(mtlstm_pred['unrate'].values.reshape(-1,1))
    
    predictions[model] = np.mean(np.concatenate(estimator_infl_predictions, axis=1), axis=1).reshape(-1,)
    unrate_predictions[model] = np.mean(np.concatenate(estimator_unrate_predictions, axis=1), axis=1).reshape(-1,)

    # true_infl_rmse = mean_squared_error(y_test, predictions[model], squared=False)
    # true_unrate_rmse = mean_squared_error(y_unrate_test, unrate_predictions[model], squared=False)

    # # Feature importance
    # increases_infl_rmse = []
    # increases_unrate_rmse = []
    # for j in range(X_train.shape[2]):

    #     print('Computing feature importance...')
    #     # create copy of test feature set
    #     X_test_temp = X_test.copy()
    #     X_test_temp[:,:,j] = 0

    #     infl_temp_pred = []
    #     unrate_temp_pred = []
    #     for estimator in mtlstm_estimators:
    #         # Make forecasts with each estimator in the ensemble
    #         estimator_pred = multitask_preds_to_df(estimator.predict(X_test_temp, verbose=0), colnames=['infl', 'unrate'])
    #         infl_temp_pred.append(estimator_pred['infl'].values.reshape(-1,1))
    #         unrate_temp_pred.append(estimator_pred['unrate'].values.reshape(-1,1))
        
    #     # Take the mean prediction across estimators
    #     print('Computing ensemble predictions...')
    #     infl_temp_pred = np.mean(np.concatenate(infl_temp_pred, axis=1), axis=1).reshape(-1,)
    #     unrate_temp_pred = np.mean(np.concatenate(unrate_temp_pred, axis=1), axis=1).reshape(-1,)

    #     # Compute RMSE with feature j set to 0 (the mean)
    #     print('Recording increase in RMSE...')
    #     temp_infl_rmse = mean_squared_error(y_test, infl_temp_pred, squared=False)
    #     temp_unrate_rmse = mean_squared_error(y_unrate_test, unrate_temp_pred, squared=False)

    #     # Compute increase in RMSE
    #     increase_infl_rmse = temp_infl_rmse - true_infl_rmse
    #     increase_unrate_rmse = temp_unrate_rmse - true_unrate_rmse

    #     increases_infl_rmse.append(increase_infl_rmse)
    #     increases_unrate_rmse.append(increase_unrate_rmse)

    
    # infl_featimp_df = pd.DataFrame(
    #     np.array(increases_infl_rmse).reshape(1,X_train.shape[2]),
    #     columns=X.columns
    # )
    # unrate_featimp_df = pd.DataFrame(
    #     np.array(increases_unrate_rmse).reshape(1,X_train.shape[2]),
    #     columns=X.columns
    # )

    # print('Saving feature importance...')
    # infl_featimp_df.to_csv(FEATIMP_DIR + model_name + '_' + target + '_' + train_cutoff_year + 'inc_rmse.csv')
    # unrate_featimp_df.to_csv(FEATIMP_DIR + model_name + '_unrate_tp' + str(h) + '_' + train_cutoff_year + 'inc_rmse.csv')

# Save predictions
predictions = pd.DataFrame(predictions, index=test_ind)
unrate_predictions = pd.DataFrame(unrate_predictions, index=test_ind)
print('Saving predictions...')
predictions.to_csv(PRED_DIR + 'all_LSTMs_' + target + '_' + train_cutoff_year + '_predictions.csv')
unrate_predictions.to_csv(PRED_DIR + 'all_MT-LSTMs_' + 'unrate_tp' + str(h) + '_' + train_cutoff_year + '_predictions.csv')
print('COMPLETE.')