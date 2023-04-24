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
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
from tensorflow import keras
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
def build_lstm(architecture, dense_architecture, input_shape):

    # Init
    model = keras.models.Sequential()

    # Input layer
    model.add(keras.layers.Input(shape=input_shape, name='input'))
    
    for units in architecture[:-1]:
        model.add(keras.layers.LSTM(units, return_sequences=True, recurrent_dropout=0.2))

    # Do not return sequences for the last layer
    model.add(keras.layers.LSTM(architecture[-1], recurrent_dropout=0.2))

    if dense_architecture != None:
        for units in dense_architecture:
            model.add(keras.layers.Dense(units))

    # Output layer
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    return model


def build_lstm_ensemble(architecture, dense_architecture, input_shape, n_estimators=10):


    models = []
    
    for i in range(n_estimators):
        print(f'Now training estimator {i+1} of {n_estimators}')
        cbs = [
            keras.callbacks.EarlyStopping(monitor='val_mae',min_delta=10**(-4), patience=50, restore_best_weights=True, verbose=1)
            # keras.callbacks.ReduceLROnPlateau(monitor='val_mae',min_delta=10**(-5), patience=250, restore_best_weights=True, verbose=1)
        ]
        model = keras.models.Sequential()
        for units in architecture[:-1]:
            # Return sequences for all but last lstm layer
            model.add(keras.layers.LSTM(units, input_shape=input_shape, return_sequences=True, recurrent_dropout=0.2))
        
        # Do not return sequences to dense layer
        model.add(keras.layers.LSTM(architecture[-1], input_shape=input_shape, recurrent_dropout=0.2))

        if dense_architecture != None:
            for units in dense_architecture:
                model.add(keras.layers.Dense(units))

        # Output layer
        model.add(keras.layers.Dense(1))

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

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


def build_multitask_lstm(hardsharing_architecture, taskspecific_architecture, dense_architecture, ntasks, tasknames, input_shape):

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
    hardsharing_layers = [ keras.layers.LSTM(nodes, return_sequences=True, recurrent_dropout=0.1) for nodes in hardsharing_architecture  ]    

    hardsharing = keras.models.Sequential(hardsharing_layers)(inlayer)

    # Task-specific layers

    # Init output list. Each task has its own output
    outputs = []
    for i in range(ntasks):

        # Create task-specific layer. Return sequence unless last LSTM layer which feeds dense layer.
        tasklayers = [ keras.layers.LSTM(nodes, return_sequences=True, recurrent_dropout=0.1) for nodes in taskspecific_architecture[:-1]]
        tasklayers.append(keras.layers.LSTM(taskspecific_architecture[-1], recurrent_dropout=0.1))
        
        # Add dense layers if any
        if dense_architecture != None:
            for nodes in dense_architecture:
                tasklayers.append(keras.layers.Dense(nodes))
        
        # Add output layer
        tasklayers.append(keras.layers.Dense(1))

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

h = sys.argv[2]
target = 'infl_tp' + str(h)
print(f'Target: {target}')

# Read data
X = pd.read_csv('inflfc_features.csv', index_col='date')
X_ar = pd.read_csv('ar_terms_announcement_delay_adj.csv', index_col='date')
y = pd.read_csv('inflfc_targets.csv', index_col='date')
# vol = pd.read_csv('infl_delta_vol.csv', index_col='date')
unrate = pd.read_csv('delta_unrate.csv', index_col='date')

X.index = pd.to_datetime(X.index, format='%Y-%m-%d')
X_ar.index = pd.to_datetime(X_ar.index)
X = pd.concat([X,X_ar], axis=1)
y.index = pd.to_datetime(y.index, format='%Y-%m-%d')
# vol.index = pd.to_datetime(vol.index, format='%Y-%m-%d')
unrate.index = pd.to_datetime(unrate.index)

# Select target variable
y = y[target].to_frame()
unrate = unrate['unrate_tp' + str(h)]

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

print(train_start, train_end, val_start, val_end, test_start, test_end)

# Split data
X_train = X.loc[train_start:train_end]
X_val = X.loc[val_start:val_end]
X_test = X.loc[test_start:test_end]

y_train = y.loc[train_start:train_end]
y_val = y.loc[val_start:val_end]
y_test = y.loc[test_start:test_end]

# Multi-task data
unrate_train = unrate.loc[y_train.index]
unrate_val = unrate.loc[y_val.index]
unrate_test = unrate.loc[y_test.index]

mt_y_train = pd.concat([y_train, unrate_train],axis=1)
mt_y_val = pd.concat([y_val, unrate_val],axis=1)
mt_y_test = pd.concat([y_test, unrate_test],axis=1)

mt_y_train = [mt_y_train[col].values for col in mt_y_train]
mt_y_val = [mt_y_val[col].values for col in mt_y_val]
mt_y_test = [mt_y_test[col].values for col in mt_y_test]


# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape data into (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# X_train_full = np.concatenate([X_train,X_val], axis=0)
# y_train_full = pd.concat([y_train, y_val], axis=0)
# vol_train_full = pd.concat([vol_train, vol_val])
# mt_y_train_full = df_to_multitask_y(pd.concat([y_train_full, vol_train_full], axis=1))


# Train models

# LSTMs
predictions = {}
# vol_predictions = pd.DataFrame()
unrate_predictions = {}


# architectures=[
#     [16],
#     [32],
#     [16,16],
#     [32,32]
# ]
# dense_architectures =[
#     [4],
#     [4,4],
#     [8],
#     [8,8]
# ]

# # LSTM w/ dense
# for architecture in architectures:
#     str_arch = '-'.join([str(units) for units in architecture])
#     for dense_architecture in dense_architectures:
        
#         str_dense_arch = '-'.join([str(units) + 'd' for units in dense_architecture])
#         model_name = 'LSTM_' + str_arch + '-' + str_dense_arch
#         model_path = MODELS_FOLDER + model_name + '_' + target + '_' + train_cutoff_year + '.h5'

#         if not os.path.isfile(model_path):
#             print(f'No model found, training model:{model}')
#             model = build_lstm(
#                 architecture=architecture,
#                 dense_architecture=dense_architecture,
#                 input_shape=X_train.shape[1:]
#             )
#             cbs = [
#                 keras.callbacks.EarlyStopping(
#                     monitor='val_mae', min_delta=10**(-4), patience=100, restore_best_weights=True, verbose=1
#                 )
#             ]
        
#             model.fit(
#                 X_train, y_train,
#                 validation_data=(X_val, y_val),
#                 epochs=1500,
#                 batch_size=4,
#                 verbose=0,
#                 callbacks=cbs
#             )

#             print('Model fit complete, saving model...')
#             model.save(model_path)

#         else:
#             print('Existing model found, loading...')
#             model = keras.models.load_model(model_path)
        
#         print('Generating predictions...')
#         y_pred = model.predict(X_test, verbose=0).reshape(-1,)

#         predictions[model_name] = y_pred



# Ensemble LSTM 
e_architectures = [
    [32],
    [32,32]
]
e_dense_architectures = [
    # [4],
    # [8],
    # [4]*2,
    [8]*2,
    [8,4]
]
n_estimators=10
for architecture in e_architectures:
    str_arch = '-'.join([str(units) for units in architecture])
    for dense_architecture in e_dense_architectures:
        
        str_dense_arch = '-'.join([str(units) + 'd' for units in dense_architecture])
        model_name = 'E-LSTM_' + str_arch + '-' + str_dense_arch
        model_path = MODELS_FOLDER + model_name + '_' + target + '_' + train_cutoff_year + '.h5'

        if not os.path.isfile(model_path):
            print(f'No model found, training model:{model_name}')
            model = build_lstm_ensemble(
                architecture=architecture,
                dense_architecture=dense_architecture,
                input_shape=X_train.shape[1:]
            )

            print('Model fit complete, saving model...')
            model.save(model_path)

        else:
            print('Existing model found, loading...')
            model = keras.models.load_model(model_path)
        
        print('Generating predictions...')
        y_pred = model.predict(X_test, verbose=0).reshape(-1,)

        predictions[model_name] = y_pred



# MT-LSTM
mt_architectures = {
    # 'MT-LSTM_32-32---8-4d': {'hardsharing_architecture':[32]*2,'taskspecific_architecture':[8],'dense_architecture': [4]},
    'MT-LSTM_32-32---8-8d': {'hardsharing_architecture':[32]*2,'taskspecific_architecture':[8],'dense_architecture': [8]},
    # 'MT-LSTM_32---8-4d': {'hardsharing_architecture':[32],'taskspecific_architecture':[8],'dense_architecture': [4]},
    # 'MT-LSTM_32---8-8d': {'hardsharing_architecture':[32],'taskspecific_architecture':[8],'dense_architecture': [8]},
    'MT-LSTM_32-32-32---8-8d-8d': {'hardsharing_architecture':[32]*3,'taskspecific_architecture':[8],'dense_architecture': [8,8]}
}
# for model in mt_architectures.keys():
    
#     print(f'Model: {model}')
#     model_path = MODELS_FOLDER + model + '_' + target + '_' + train_cutoff_year + '.h5'

#     if not os.path.isfile(model_path):
#         print(f'Training model: {model}...')

#         model_params = mt_architectures[model]
#         mtlstm = build_multitask_lstm(
#             **model_params,
#             ntasks=2,
#             tasknames=[target,'unrate'],
#             input_shape=X_train.shape[1:]
#         )

#         mt_early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=10**-7, patience=50, verbose=1, restore_best_weights=True)

#         mtlstm.fit(
#             X_train, mt_y_train, 
#             validation_data = (X_val, mt_y_val),
#             epochs=1500,
#             batch_size=4,
#             verbose=0,
#             callbacks=[mt_early_stopping_monitor]
#         )

#         print('Training finished. Saving model...')
#         mtlstm.save(model_path)
#     else:
#         print('Existing model found, loading...')
#         mtlstm = keras.models.load_model(model_path)

#     mtlstm_pred = multitask_preds_to_df(mtlstm.predict(X_test, verbose=0), colnames=[target, 'unrate'])
#     predictions[model] = mtlstm_pred[target].values.reshape(-1,)
#     unrate_predictions[model] = mtlstm_pred['unrate'].values.reshape(-1,)

# E-MT-LSTM
for model in mt_architectures.keys():
    
    model_name = 'E-' + model
    print(f'Training {model_name}...')

    estimator_infl_predictions = []
    estimator_unrate_predictions = []
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

            mt_early_stopping_monitor = keras.callbacks.EarlyStopping(monitor='val_' + target + '_mae', min_delta=10**-4, patience=100, verbose=1, restore_best_weights=True)

            mtlstm.fit(
                X_train, mt_y_train, 
                validation_data = (X_val, mt_y_val),
                epochs=1500,
                batch_size=4,
                verbose=0,
                callbacks=[mt_early_stopping_monitor]
            )
            mtlstm.save(model_path)
        
        else:
            mtlstm = keras.models.load_model(model_path)

        mtlstm_pred = multitask_preds_to_df(mtlstm.predict(X_test, verbose=0), colnames=[target, 'unrate'])
        estimator_infl_predictions.append(mtlstm_pred[target].values.reshape(-1,1))
        estimator_unrate_predictions.append(mtlstm_pred['unrate'].values.reshape(-1,1))
    
    predictions[model_name] = np.mean(np.concatenate(estimator_infl_predictions, axis=1), axis=1).reshape(-1,)
    unrate_predictions[model_name] = np.mean(np.concatenate(estimator_unrate_predictions, axis=1), axis=1).reshape(-1,)

# Save predictions
predictions = pd.DataFrame(predictions, index=y_test.index)
unrate_predictions = pd.DataFrame(unrate_predictions, index=y_test.index)
print('Saving predictions...')
predictions.to_csv(PRED_FOLDER + 'all_LSTMs_' + target + '_' + train_cutoff_year + '_predictions.csv')
unrate_predictions.to_csv(PRED_FOLDER + 'all_MT-LSTMs_' + 'unrate_tp' + str(h) + '_' + train_cutoff_year + '_predictions.csv')
print('COMPLETE.')