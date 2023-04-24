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
import sys
import os

# Machine Learning libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import tensorflow as tf
tf.random.set_seed(1) # for reproducible results
from tensorflow import keras


# Paths
RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results9/"
MODELS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Models9/"
CV_RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/CVResults9/"
FEATIMP_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/FeatureImportances/"
PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions9/"


# IMPORT data
print('Preparing data...')
# Select the data you want
X = pd.read_csv('inflfc_features.csv')
# X = pd.read_csv('inflfc_features_lags.csv')
# X = pd.read_csv('inflfc_features_lags4_ar.csv')
X_ar = pd.read_csv('ar_terms.csv')
y = pd.read_csv('inflfc_targets.csv')

X['date'] = pd.to_datetime(X['date'], format='%Y-%m-%d')
X_ar['date'] = pd.to_datetime(X_ar['date'], format='%Y-%m-%d')
y['date'] = pd.to_datetime(y['date'], format='%Y-%m-%d')

X = X.set_index('date')
X_ar = X_ar.set_index('date')
y = y.set_index('date')
X = pd.concat([X,X_ar], axis=1)

# Split data

# validation_years = 1
train_cutoff_year = sys.argv[1]
train_cutoff = train_cutoff_year + '-12-01'
train_start = '1965-01-01'
# train_end = datetime.strptime(train_cutoff, '%Y-%m-%d') - relativedelta(years=validation_years)
# val_start = train_end + relativedelta(months=1)
# val_end = val_start + relativedelta(years=validation_years-1, months=11)
test_start = datetime.strptime(train_cutoff, '%Y-%m-%d') + relativedelta(months=1)
test_end = test_start + relativedelta(months=11)

print(f'Training period: {train_start} to {train_cutoff}')
print(f'Testing period: {test_start} to {test_end}')

X_train = X.loc[train_start:train_cutoff]
# X_val = X.loc[val_start:val_end]
X_test = X.loc[test_start:test_end]

y_train = y.loc[train_start:train_cutoff]
# y_val = y.loc[val_start:val_end]
y_test = y.loc[test_start:test_end]

# Note: dates on X and y are matched such that the same date across dataframes has the features at time t and the 
# inflation at time t+h 

# Scale data
s = StandardScaler()

X_train = pd.DataFrame(s.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
# X_val = pd.DataFrame(s.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(s.transform(X_test),columns=X_test.columns, index=X_test.index)

# Helper functions
def build_multitask_nn(architecture, ntasks, tasknames, input_shape, h_act='elu', kernel_init='he_normal', lr=0.001, l2=0.0):

    """
    Builds a multitask nn.

    Parameters
    ----------
    architecture: a string architecture. Number of units in each layer of the hardsharing layers and task specific layers should be 
        seperated by a dash. Hardsharing layers and task specific layers should be seperated by an underscore, e.g., '32-16_8-4'. 
    taskspecific_architectures: the architecture for each task specific group of layers (iterable of iterables).
    ntasks: The number of tasks.
    tasknames: tasknames
    input_shape: input shape
    lr: learning Rate
    l2: degree of l2 regularization. Same for all layers.
    """

    # Convert architecture to list of hidden units for hard sharing and task specific layers
    hardsharing_architecture_str = architecture.split('_')[0].split('-')
    hardsharing_architecture = [int(x) for x in hardsharing_architecture_str]

    taskspecific_architecture_str = architecture.split('_')[1].split('-')
    taskspecific_architecture = [int(x) for x in taskspecific_architecture_str]
    
    # Hard-sharing layers
    batchnorm = False

    inlayer = keras.layers.Input(shape=input_shape, name='input')

    n_layers = len(hardsharing_architecture)

    # Create hardsharing layers
    if n_layers >= 2:
        batchnorm=True # for nets with two or more layers we apply batch normalization to avoid exploding/vanishing gradients

    hardsharing_layers = [ keras.layers.Dense(nodes, activation=h_act, 
                                                kernel_initializer=kernel_init, 
                                                kernel_regularizer=keras.regularizers.l2(l2)) for nodes in hardsharing_architecture]
    
    if batchnorm:
        hardsharing_layers = [ (keras.layers.Dense(nodes, activation=h_act, kernel_initializer=kernel_init, kernel_regularizer=keras.regularizers.l2(l2)), keras.layers.BatchNormalization()) for nodes in hardsharing_architecture ]
        
        hardsharing_layers = [x for tup in hardsharing_layers for x in tup]

    hardsharing = keras.models.Sequential(hardsharing_layers)(inlayer)
      

    # Task-specific layers
    batchnorm = False

    outputs = []
    for i in range(ntasks):

        # Create task-specific layer
        if len(taskspecific_architecture) >= 2:
            batchnorm = True

        if batchnorm:
            tasklayers = [ (keras.layers.Dense(nodes, activation=h_act, 
                                                kernel_initializer=kernel_init, 
                                                kernel_regularizer=keras.regularizers.l2(l2)), keras.layers.BatchNormalization()) for nodes in taskspecific_architecture]
            tasklayers = [x for tup in tasklayers for x in tup]
        else:
            tasklayers = [ keras.layers.Dense(nodes, activation=h_act, 
                                                kernel_initializer=kernel_init, 
                                                kernel_regularizer=keras.regularizers.l2(l2)) for nodes in taskspecific_architecture]
            
        # Add output layer
        tasklayers.append(keras.layers.Dense(1, activation='linear'))

        # Create task-specific net
        tasknet = keras.models.Sequential(tasklayers, name=tasknames[i])(hardsharing)

        # Append task-specific net to outputs list
        outputs.append(tasknet)

    # Create and compile full model
    model = keras.models.Model(inputs=inlayer, outputs=outputs)
    optimizer=keras.optimizers.Nadam(learning_rate=lr, beta_1=0.9, beta_2=0.999)
    model.compile(loss=['mse']*ntasks, optimizer=optimizer, metrics = ['mae'])

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

tscv = TimeSeriesSplit(n_splits=5, gap=0)

architectures = ['32-16_8-4','64-32_16-8', '128-64-32_16-8-4']
epochs=[100,150,200,250]
l2 = [1000.0,100.0,10.0,1.0,0.1]
predictions = pd.DataFrame()

print(f'Writing test dict...')
test_dict = {'a':1, 'b':2}
with open('test.txt', 'w') as f:
    f.write(str(test_dict))

print('Commencing training...')
for a in architectures:
        model = 'MTNN' + a
        model_path = MODELS_FOLDER + model + '_' + train_cutoff_year + '.h5'

        print(f'Model:{model}')
        if not os.path.isfile(model_path):
            print('No model found. Training...')

            best_params={}
            min_rmse = 100.0
            for e in epochs:
                for p in l2:
                    print(f'Epochs:{e}')
                    print(f'l2:{p}')
                    # Initialize avg rmse and n:= number of observations
                    avg_rmse = 0
                    n=0  
                    for train_ind, test_ind in tscv.split(X_train):
                        print(f'Fitting split {n+1} of 5.')
                        
                        X_tr, X_val = X_train.iloc[train_ind], X_train.iloc[test_ind]
                        y_tr, y_val = y_train.iloc[train_ind], y_train.iloc[test_ind]

                        nn = build_multitask_nn(
                            architecture=a, 
                            lr=0.001,
                            l2=p,
                            ntasks=len(y_train.columns), 
                            tasknames=y_train.columns, 
                            input_shape=X_train.shape[1:],
                            h_act='relu',
                            kernel_init='glorot_uniform'
                        )

                        nn.fit(
                            X_tr.values, y_tr.values,
                            batch_size=4, epochs=e, 
                            verbose=0
                        )

                        val_pred = nn.predict(X_val.values, verbose=0)
                        val_pred = multitask_preds_to_df(val_pred, colnames=y.columns)
                        y_val_mat = y_val.values
                        val_pred_mat = val_pred.values
                        rmse = np.sqrt(np.mean(np.square(y_val_mat-val_pred_mat),axis=0))

                        # perform online update of the mean rmse
                        avg_rmse = (1/(n+1))*(n*avg_rmse + rmse)

                        if n==4:
                            print(f'Epochs: {e}. l2: {p}')
                            print(f'Mean RMSE:{avg_rmse}')

                        n += 1

                    sum_avg_rmse = np.sum(avg_rmse)
                    if sum_avg_rmse < min_rmse:
                        min_rmse = sum_avg_rmse
                        best_params['epochs'] = e
                        best_params['l2'] = p

            # Save best parameters
            with open(CV_RESULTS_FOLDER + model + '_' + train_cutoff_year + '_best_params.txt', 'w') as params_file:
                params_file.write(str(best_params))

            # Refit model with best params
            nn = build_multitask_nn(
                architecture=a, 
                lr=0.001,
                l2=best_params['l2'],
                ntasks=len(y_train.columns), 
                tasknames=y_train.columns, 
                input_shape=X_train.shape[1:],
                h_act='relu',
                kernel_init='glorot_uniform'
            )

            nn.fit(
                X_train.values, y_train.values, 
                batch_size=4, epochs=best_params['epochs'], 
                verbose=0
            )

            print('Saving model...')
            nn.save(model_path)
        
        else:
            print('Existing model found, loading...')
            nn = keras.models.load_model(model_path)

        print('Generating predictions...')
        y_pred = nn.predict(X_test.values, verbose=0)
        y_pred_df = multitask_preds_to_df(y_pred, y_test.columns)
        y_pred_df['Model'] = model
        y_pred_df['date'] = y_test.index
        y_pred_df = y_pred_df.set_index('date')
        predictions = pd.concat([predictions, y_pred_df])

print('Saving predictions...')
predictions.to_csv(PRED_FOLDER + 'allmtnets' + '_' + train_cutoff_year + '_.csv')
print('Complete.')