############################################
# Import Libraries
############################################


# Data Manipulation Libraries
import pandas as pd 
import numpy as np
from numpy.random import seed # for reproducible results
seed(1)
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta # may need to pip install python-dateutil
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
from tensorflow import keras
from keras.regularizers import L1
from keras.layers import LSTM, Dense
import shap # For interpretting models

# System libraries
import os # For file management
import sys



# Paths
MODELS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Models10/"
CV_RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/CVResults10/"
FEATIMP_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/FeatureImportances/"
PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions10/"

# Helper functions
def build_lstm(architecture, dense_architecture, input_shape, recurrent_dropout=0.0, l1=0.0):

    # Init
    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=input_shape, name='input'))
    
    for units in architecture[:-1]:
        model.add(
            LSTM(units, return_sequences=True, recurrent_dropout=recurrent_dropout,kernel_regularizer=L1(l1=l1))
        )

    # Do not return sequences for the last layer
    model.add(LSTM(architecture[-1], recurrent_dropout=recurrent_dropout,kernel_regularizer=L1(l1=l1)))

    if dense_architecture != None:
        for units in dense_architecture:
            model.add(Dense(units, activation='relu', kernel_regularizer=L1(l1=l1)))

    # Output layer
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    return model


def build_lstm_ensemble(architecture, dense_architecture, input_shape, recurrent_dropout=0.0, l1=0.0, n_estimators=10):


    models = []
    
    for i in range(n_estimators):
        print(f'Now training estimator {i+1} of {n_estimators}')
        cbs = [
            keras.callbacks.EarlyStopping(monitor='val_mae',min_delta=10**(-4), patience=50, restore_best_weights=True, verbose=1)
            # keras.callbacks.ReduceLROnPlateau(monitor='val_mae',min_delta=10**(-5), patience=250, restore_best_weights=True, verbose=1)
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


def build_multitask_lstm(hardsharing_architecture, taskspecific_architecture, dense_architecture, ntasks, tasknames, input_shape, recurrent_dropout=0.0, l1=0.0):

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

    inlayer = keras.layers.Input(shape=input_shape, name='input')

    # Create hardsharing layers
    hardsharing_layers = [ LSTM(nodes, return_sequences=True, recurrent_dropout=0.1) for nodes in hardsharing_architecture  ]    

    hardsharing = keras.models.Sequential(hardsharing_layers)(inlayer)

    # Task-specific layers

    # Init output list. Each task has its own output
    outputs = []
    for i in range(ntasks):

        # Create task-specific layer. Return sequence unless last LSTM layer which feeds dense layer.
        tasklayers = [ LSTM(nodes, return_sequences=True, recurrent_dropout=recurrent_dropout) for nodes in taskspecific_architecture[:-1]]
        tasklayers.append(LSTM(taskspecific_architecture[-1], recurrent_dropout=recurrent_dropout))
        
        # Add dense layers if any
        if dense_architecture != None:
            for nodes in dense_architecture:
                tasklayers.append(Dense(nodes, activation='relu', kernel_regularizer=L1(l1=l1)))
        
        # Add output layer
        tasklayers.append(Dense(1, activation='linear'))

        # Create task-specific net
        tasknet = keras.models.Sequential(tasklayers, name=tasknames[i])(hardsharing)

        # Append task-specific net to outputs list
        outputs.append(tasknet)

    # Create and compile full model
    model = keras.models.Model(inputs=inlayer, outputs=outputs)
    
    model.compile(loss=['mse']*ntasks, optimizer='adam', metrics = ['mae'])

    return model


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
X = pd.read_csv('inflfc_features.csv', index_col='date')
y = pd.read_csv('inflfc_targets.csv', index_col='date')
unrate = pd.read_csv('delta_unrate.csv', index_col='date')

X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
y.index = pd.to_datetime(y.index, format='%Y-%m-%d')
unrate.index = pd.to_datetime(unrate.index)

# Select target variable
y = y[target].to_frame()
unrate = unrate['unrate_tp' + str(h)].to_frame()

# Define cutoff dates
val_years = 5
train_cutoff_year = sys.argv[1]
train_start = '1965-01-01'
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

# Reshape data for RNN
X = pd.concat([X_train, X_val, X_test])
y = y.loc[train_start:test_end]

dataset = pd.concat([X,y], axis=1)
train_dataset = dataset.loc[train_start:train_end]
val_dataset = pd.concat([train_dataset.iloc[-11:], dataset.loc[val_start:val_end]])
test_dataset = pd.concat([val_dataset.iloc[-11:],dataset.loc[test_start:test_end]])

X_train, y_train = split_sequences(train_dataset.values, 12)
X_val, y_val = split_sequences(val_dataset.values, 12)
X_test, y_test = split_sequences(test_dataset.values, 12)

unrate_dataset = pd.concat([X,unrate], axis=1)
unrate_train_dataset = unrate_dataset.loc[train_start:train_end]
unrate_val_dataset = pd.concat([unrate_train_dataset.iloc[-11:], unrate_dataset.loc[val_start:val_end]])
unrate_test_dataset = pd.concat([unrate_val_dataset.iloc[-11:],unrate_dataset.loc[test_start:test_end]])

_, y_unrate_train = split_sequences(unrate_train_dataset.values, 12)
_, y_unrate_val = split_sequences(unrate_val_dataset.values, 12)
_, y_unrate_test = split_sequences(unrate_test_dataset.values, 12)

# Multi-task data
mt_y_train = [y_train, y_unrate_train]
mt_y_val = [y_val, y_unrate_val]
mt_y_test = [y_test, y_unrate_test]


# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 


# LSTMs
predictions = {}
unrate_predictions = {}

# # Ensemble LSTM 
# e_architectures = [
#     [32]*4
# ]
# e_dense_architectures = [
#     [8]*2
# ]
# n_estimators=10
# for architecture in e_architectures:
#     str_arch = '-'.join([str(units) for units in architecture])
#     for dense_architecture in e_dense_architectures:
        
#         str_dense_arch = '-'.join([str(units) + 'd' for units in dense_architecture])
#         model_name = 'E-LSTM-V2_' + str_arch + '-' + str_dense_arch
#         model_path = MODELS_FOLDER + model_name + '_' + target + '_' + train_cutoff_year + '.h5'

#         if not os.path.isfile(model_path):
#             print(f'No model found, training model:{model_name}')
#             model = build_lstm_ensemble(
#                 architecture=architecture,
#                 dense_architecture=dense_architecture,
#                 input_shape=X_train.shape[1:]
#             )

#             print('Model fit complete, saving model...')
#             model.save(model_path)

#         else:
#             print('Existing model found, loading...')
#             model = keras.models.load_model(model_path)
        
#         print('Generating predictions...')
#         y_pred = model.predict(X_test, verbose=0).reshape(-1,)

#         predictions[model_name] = y_pred



# MT-LSTM
mt_architectures = {
    'MT-LSTM_32-32-32---32-8d-8d': {'hardsharing_architecture':[32]*3,'taskspecific_architecture':[32],'dense_architecture': [8]*2},
    # 'MT-LSTM_32-32---8-8d': {'hardsharing_architecture':[32]*2,'taskspecific_architecture':[8],'dense_architecture': [8]},
    # 'MT-LSTM_32---8-4d': {'hardsharing_architecture':[32],'taskspecific_architecture':[8],'dense_architecture': [4]},
    # 'MT-LSTM_32---8-8d': {'hardsharing_architecture':[32],'taskspecific_architecture':[8],'dense_architecture': [8]}
}

# E-MT-LSTM
for model in mt_architectures.keys():
    
    model_name = 'E-' + model + '-V2'
    print(f'Training {model_name}...')

    estimator_infl_predictions = []
    estimator_unrate_predictions = []
    mtlstm_estimators = []
    for i in range(1, 10+1):

        model_path = MODELS_FOLDER + model_name + '/' + model_name + '_' + str(i) + '_' + target + '_' + train_cutoff_year + '.h5'

        if not os.path.isfile(model_path):
            print(f'Training estimator {i} of 10...')

            model_params = mt_architectures[model]
            mtlstm = build_multitask_lstm(
                **model_params,
                ntasks=2,
                tasknames=[target,'unrate'],
                input_shape=X_train.shape[1:]
            )

            mt_early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_' + target + '_mae', min_delta=10**-7, patience=100, verbose=1, restore_best_weights=True)

            mtlstm.fit(
                X_train, mt_y_train, 
                validation_data = (X_val, mt_y_val),
                epochs=1500,
                batch_size=4,
                verbose=0,
                callbacks=[mt_early_stopping_monitor]
            )
            mtlstm.save(model_path)
        
            mtlstm_estimators.append(mtlstm)

        else:
            mtlstm = keras.models.load_model(model_path)

            mtlstm_estimators.append(mtlstm)

        mtlstm_pred = multitask_preds_to_df(mtlstm.predict(X_test, verbose=0), colnames=[target, 'unrate'])
        estimator_infl_predictions.append(mtlstm_pred[target].values.reshape(-1,1))
        estimator_unrate_predictions.append(mtlstm_pred['unrate'].values.reshape(-1,1))
    
    predictions[model_name] = np.mean(np.concatenate(estimator_infl_predictions, axis=1), axis=1).reshape(-1,)
    unrate_predictions[model_name] = np.mean(np.concatenate(estimator_unrate_predictions, axis=1), axis=1).reshape(-1,)

    true_infl_rmse = mean_squared_error(y_test, predictions[model_name], squared=False)
    true_unrate_rmse = mean_squared_error(y_unrate_test, unrate_predictions[model_name], squared=False)

    # Feature importance
    increases_infl_rmse = []
    increases_unrate_rmse = []
    for j in range(X_train.shape[2]):

        print('Computing feature importance...')
        # create copy of test feature set
        X_test_temp = X_test.copy()
        X_test_temp[:,:,j] = 0

        infl_temp_pred = []
        unrate_temp_pred = []
        for estimator in mtlstm_estimators:
            # Make forecasts with each estimator in the ensemble
            estimator_pred = multitask_preds_to_df(estimator.predict(X_test_temp, verbose=0), colnames=['infl', 'unrate'])
            infl_temp_pred.append(estimator_pred['infl'].values.reshape(-1,1))
            unrate_temp_pred.append(estimator_pred['unrate'].values.reshape(-1,1))
        
        # Take the mean prediction across estimators
        print('Computing ensemble predictions...')
        infl_temp_pred = np.mean(np.concatenate(infl_temp_pred, axis=1), axis=1).reshape(-1,)
        unrate_temp_pred = np.mean(np.concatenate(unrate_temp_pred, axis=1), axis=1).reshape(-1,)

        # Compute RMSE with feature j set to 0 (the mean)
        print('Recording increase in RMSE...')
        temp_infl_rmse = mean_squared_error(y_test, infl_temp_pred, squared=False)
        temp_unrate_rmse = mean_squared_error(y_unrate_test, unrate_temp_pred, squared=False)

        # Compute increase in RMSE
        increase_infl_rmse = temp_infl_rmse - true_infl_rmse
        increase_unrate_rmse = temp_unrate_rmse - true_unrate_rmse

        increases_infl_rmse.append(increase_infl_rmse)
        increases_unrate_rmse.append(increase_unrate_rmse)

    
    infl_featimp_df = pd.DataFrame(
        np.array(increases_infl_rmse).reshape(1,X_train.shape[2]),
        columns=X.columns
    )
    unrate_featimp_df = pd.DataFrame(
        np.array(increases_unrate_rmse).reshape(1,X_train.shape[2]),
        columns=X.columns
    )

    print('Saving feature importance...')
    infl_featimp_df.to_csv(FEATIMP_FOLDER + model_name + '_' + target + '_' + train_cutoff_year + 'inc_rmse.csv')
    unrate_featimp_df.to_csv(FEATIMP_FOLDER + model_name + '_unrate_tp' + str(h) + '_' + train_cutoff_year + 'inc_rmse.csv')

# Save predictions
# predictions = pd.DataFrame(predictions, index=test_ind)
# unrate_predictions = pd.DataFrame(unrate_predictions, index=test_ind)
# print('Saving predictions...')
# predictions.to_csv(PRED_FOLDER + 'all_LSTMs_' + target + '_' + train_cutoff_year + '_predictions.csv')
# unrate_predictions.to_csv(PRED_FOLDER + 'all_MT-LSTMs_' + 'unrate_tp' + str(h) + '_' + train_cutoff_year + '_predictions.csv')
print('COMPLETE.')