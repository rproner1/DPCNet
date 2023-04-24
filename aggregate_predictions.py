# Data Manipulation Libraries
import pandas as pd 
import numpy as np
from numpy.random import seed # for reproducible results
seed(1)
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
import pandas as pd 
import numpy as np
from numpy.random import seed # for reproducible results
seed(1)
from datetime import datetime
from dateutil.relativedelta import relativedelta # may need to pip install python-dateutil
import warnings
warnings.filterwarnings("ignore")

# Machine Learning libraries
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os
import sys 

# Local paths
# PRED_FOLDER = "Predictions/"
# RESULTS_FOLDER = "Results/"

# Compute Canada paths
PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions10/"
RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results10/"

def oos_r2_score(y_true, y_pred, y_train):

    mse = mean_squared_error(y_true, y_pred)
    mse_trainmean = mean_squared_error(y_true, np.full_like(y_true, np.mean(y_train)))
    r2 = 1 - mse/mse_trainmean

    return r2


def mean_absolute_deviation(y_true, y_pred):

    e = y_true - y_pred
    mad = np.median(  np.abs(e - np.median(e) ) )

    return mad


def score_models(y_true, preds_df, y_train):

    results = {
        'Model': preds_df.columns,
        'RMSE': [],
        'MAE': [],
        'MAD': [],
        'R2':[]
    }

    for model in preds_df:

        y_pred = preds_df[model].values.reshape(-1,)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        mad = mean_absolute_deviation(y_true, y_pred)
        r2 = oos_r2_score(y_true, y_pred, y_train)

        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['MAD'].append(mad)
        results['R2'].append(r2)

    results_df = pd.DataFrame(results).set_index('Model')
    rel_results_df = results_df.apply(lambda row: row / results_df.loc['RW'], axis=1)
    full_results_df = pd.concat([results_df, rel_results_df],axis=0)

    return full_results_df


# Horizon window h
h=sys.argv[1]
target = 'infl_tp' + str(h)
unrate_target = 'unrate_tp' + str(h)

train_start = '1965-01-01'
train_finish = '1989-12-01'
test_start = '1990-01-01'
test_finish = '2019-12-01'

years = [1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018]

y = pd.read_csv('inflfc_targets.csv', index_col=0)
y_unrate = pd.read_csv('delta_unrate.csv', index_col='date')
y = y[target].to_frame()
y_train = y.loc[train_start:train_finish].values.reshape(-1,)
y_test = y.loc[test_start:test_finish].values.reshape(-1,)
y_unrate_train = y_unrate.loc[train_start:train_finish, unrate_target].values.reshape(-1,)
y_unrate_test = y_unrate.loc[test_start:test_finish, unrate_target].values.reshape(-1,)

preds_path = PRED_FOLDER + 'all_models_all_years' + target +  '_predictions.csv'
if not os.path.isfile(preds_path):
    print('Concatenating predictions...')
    preds_all_years = pd.DataFrame()
    for year in years:

        path = PRED_FOLDER + 'all_models_' + target + '_' + str(year) + '_predictions.csv'

        preds = pd.read_csv(path, index_col='date')

        preds_all_years = pd.concat([preds_all_years, preds])

    print('Predictions concatenated. Saving...')
    preds_all_years.to_csv(PRED_FOLDER + 'all_models_all_years' + target +  '_predictions.csv')

else:
    print('Loading predictions...')
    preds_all_years = pd.read_csv(preds_path, index_col='date')

preds_path = PRED_FOLDER + 'all_LSTMs_all_years_' + target +  '_predictions.csv'
if not os.path.isfile(preds_path):
    print('Concatenating LSTM predictions...')
    lstm_preds_all_years = pd.DataFrame()
    for year in years:

        path = PRED_FOLDER + 'all_LSTMs_' + target + '_' + str(year) + '_predictions.csv'

        preds = pd.read_csv(path, index_col=0)

        lstm_preds_all_years = pd.concat([lstm_preds_all_years, preds])

    print('Predictions concatenated. Saving...')
    lstm_preds_all_years.to_csv(preds_path)

else:
    print('Loading LSTM predictions...')
    lstm_preds_all_years = pd.read_csv(preds_path, index_col=0)


# Concatenate unrate predictions
preds_path = PRED_FOLDER + 'all_MT-LSTMs_all_years_' + unrate_target +  '_predictions.csv'
if not os.path.isfile(preds_path):
    print('Concatenating unemployment rate change predictions...')
    mt_unrate_preds_all_years = pd.DataFrame()
    for year in years:

        path = PRED_FOLDER + 'all_MT-LSTMs_' + unrate_target + '_' + str(year) + '_predictions.csv'

        preds = pd.read_csv(path, index_col=0)

        mt_unrate_preds_all_years = pd.concat([mt_unrate_preds_all_years, preds])

    print('Predictions concatenated. Saving...')
    mt_unrate_preds_all_years.to_csv(preds_path)

else:
    print('Loading unemployment rate change predictions...')
    mt_unrate_preds_all_years = pd.read_csv(preds_path, index_col=0)

preds_path = PRED_FOLDER + 'all_models_all_years' + unrate_target +  '_predictions.csv'
if not os.path.isfile(preds_path):
    print('Concatenating predictions...')
    unrate_preds_all_years = pd.DataFrame()
    for year in years:

        path = PRED_FOLDER + 'all_models_' + unrate_target + '_' + str(year) + '_predictions.csv'

        preds = pd.read_csv(path, index_col='date')

        unrate_preds_all_years = pd.concat([unrate_preds_all_years, preds])

    print('Predictions concatenated. Saving...')
    unrate_preds_all_years.to_csv(preds_path)

else:
    print('Loading predictions...')
    unrate_preds_all_years = pd.read_csv(preds_path, index_col='date')

unrate_preds_all_years = pd.concat([unrate_preds_all_years,mt_unrate_preds_all_years], axis=1)
preds_all_years = pd.concat([preds_all_years, lstm_preds_all_years], axis=1)

# Evaluate predictions
# preds_90s = preds_all_years.loc[test_start:'1999-12-01']
# y_test_90s = y.loc[test_start:'1999-12-01'].values.reshape(-1,)

# preds_00s = preds_all_years.loc['2000-01-01':'2009-12-01']
# y_test_00s = y.loc['2000-01-01':'2009-12-01'].values.reshape(-1,)

# preds_10s = preds_all_years.loc['2010-01-01':test_finish]
# y_test_10s =  y.loc['2010-01-01':test_finish].values.reshape(-1,)

# Get results
# results_90s = score_models(y_true=y_test_90s, preds_df=preds_90s, y_train=y_train)
# results_00s = score_models(y_true=y_test_00s, preds_df=preds_00s, y_train=y_train)
# results_10s = score_models(y_true=y_test_10s, preds_df=preds_10s, y_train=y_train)
results_all = score_models(y_true=y_test, preds_df=preds_all_years, y_train=y_train)
unrate_results = score_models(y_true=y_unrate_test, preds_df=unrate_preds_all_years, y_train=y_unrate_train)
# results_unrate = score_models(unrate_test, unrate_preds_all_years, unrate_train)

print('Evaluation done. Saving...')
# results_90s.to_csv(RESULTS_FOLDER + '1990s_results_' +  target + '.csv')
# results_00s.to_csv(RESULTS_FOLDER + '2000s_results_' +  target + '.csv')
# results_10s.to_csv(RESULTS_FOLDER + '2010s_results_' +  target + '.csv')
results_all.to_csv(RESULTS_FOLDER + 'allyears_results_' +  target + '.csv')
unrate_results.to_csv(RESULTS_FOLDER + 'allyears_results_' +  unrate_target + '.csv')
preds_all_years.to_csv(PRED_FOLDER + target + 'master_predictions.csv')
unrate_preds_all_years.to_csv(PRED_FOLDER + unrate_target + 'master_predictions.csv')
print('Complete.')


