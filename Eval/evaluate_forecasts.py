# ******************************** Imports ********************************

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from scipy import stats
import statsmodels.tsa.api as smt
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys 

# ******************************** Parameters ********************************

RUN_LOCALLY = True
CONCAT_PREDS = False

# ******************************** Directories ********************************

if RUN_LOCALLY:
    PRED_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Predictions3/'
    RESULTS_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Results3/'
    DATA_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Data/'
else:
    PRED_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions2/"
    RESULTS_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Results3/"
    DATA_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Data/"


# ******************************** Functions ********************************

def oos_r2_score(y_true, y_pred, hist_means):

    mse = mean_squared_error(y_true, y_pred)
    mse_trainmean = mean_squared_error(y_true, np.full_like(y_true, hist_means))
    r2 = 1 - mse/mse_trainmean

    return r2


def median_absolute_deviation(y_true, y_pred):

    e = y_true - y_pred
    mad = np.median(  np.abs(e - np.median(e) ) )

    return mad

def pttest(y, yhat):
    """Given NumPy arrays with predictions and with true values, 
    return Directional Accuracy Score, Pesaran-Timmermann statistic and its p-value
    """
    size = y.shape[0]
    pyz = np.sum(np.sign(y) == np.sign(yhat))/size
    py = np.sum(y > 0)/size
    qy = py*(1 - py)/size
    pz = np.sum(yhat > 0)/size
    qz = pz*(1 - pz)/size
    p = py*pz + (1 - py)*(1 - pz)
    v = p*(1 - p)/size
    w = ((2*py - 1)**2) * qz + ((2*pz - 1)**2) * qy + 4*qy*qz
    pt = (pyz - p) / (np.sqrt(v - w))
    pval = 1 - stats.norm.cdf(pt, 0, 1)
    return pyz, pt, pval

def diebold_mariano(y_true, y_pred1, y_pred2, loss='mse', verbose=False):
    
    if loss == 'mse':
        loss1 = (y_true - y_pred1)**2
        loss2 = (y_true - y_pred2)**2
    elif loss == 'mae':
        loss1 = np.abs(y_true - y_pred1)
        loss2 = np.abs(y_true - y_pred2)
    else:
        ValueError('Not a valid loss function. Valid loss functions are {"mse", "mae"}.')
        
    d = loss1 - loss2
    
    T = len(d)
    M = round(T**(1/3))
    
    # Truncated heteroskedasticity and autocorrelation variance estimator (HAC)
    # autocov for lags 0 to M. 
    acov = smt.acovf(d,fft=False,nlag=M)
    var_d = acov[0] + 2*np.sum(acov[1:])
    se_d = np.sqrt( (1/T) * var_d )
    d_bar = np.mean(d)
    
    if var_d == np.nan:
        print('Variance estimate is NaN!')
        return None
    
    test_statistic = d_bar / se_d
    
    p_value = norm.sf(abs(test_statistic))*2
    
    return test_statistic, p_value


# def plot_diebold_mariano(y_true, forecasts, loss='squared', save=False, savepath=''):
    
#     tstat_mat = np.empty((forecasts.shape[1],forecasts.shape[1]))
#     pval_mat = np.empty((forecasts.shape[1],forecasts.shape[1]))
#     for i in range(forecasts.shape[1]):
#         for j in range(forecasts.shape[1]):
            
#             verbose=False
#             # if i==4:
#             #     verbose=True
            
#             if i == j:
#                 p_value = 1.0
#                 tstat=0.0
#             else:
                
#                 tstat, p_value = diebold_mariano(
#                     y_true, 
#                     forecasts.iloc[:,i].values.reshape(-1,), 
#                     forecasts.iloc[:,j].values.reshape(-1,),
#                     loss=loss,
#                     verbose=verbose
#                 )
                
#             pval_mat[i, j] = p_value
#             tstat_mat[i, j] = tstat
#     pval_df = pd.DataFrame(pval_mat, columns=forecasts.columns, index=forecasts.columns)
#     tstat_df = pd.DataFrame(tstat_mat, columns=forecasts.columns, index=forecasts.columns)
#     mask_mat = np.triu(pval_df)
    
#     plt.clf()
#     sns.heatmap(tstat_df, annot=tstat_df, center=0.0, cmap='viridis', mask=mask_mat)
#     if save:
#         plt.savefig(savepath, dpi=300)
        
#     plt.show()
    
#     return tstat_df, pval_df

def score_models(y_true, preds_df):

    results = {
        'Model': preds_df.columns,
        'RMSE': [],
        'MAE': [],
        'MAD': [],
        'R2': [],
        'PT': [],
        'PTpval': [],
        'DM' : [],
        'DMpval': []
    }

    hist_mean_preds = preds_df['HistMean'].values.ravel()
    rw_pred = preds_df['RW'].values.ravel()
    
    y_true_diff = np.diff(y_true, n=1)

    for model in preds_df:

        y_pred = preds_df[model].values.reshape(-1,)
        y_pred_diff = np.diff(y_pred, n=1)
        
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        mad = median_absolute_deviation(y_true, y_pred)
        r2 = oos_r2_score(y_true, y_pred, hist_mean_preds)
        
        dm, dm_pval = diebold_mariano(y_true, y_pred, rw_pred)
        _,pt,pval = pttest(y_true_diff, y_pred_diff)
        
        
        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['MAD'].append(mad)
        results['R2'].append(r2)
        results['PT'].append(pt)
        results['PTpval'].append(pval)
        results['DM'].append(dm)
        results['DMpval'].append(pval)
        

    results_df = pd.DataFrame(results).set_index('Model')
    rel_results_df = results_df.apply(lambda row: row / results_df.loc['RW'], axis=1)
    full_results_df = pd.concat([results_df, rel_results_df],axis=0)

    return full_results_df

# ******************************** Data ********************************

# Horizon window h
h=sys.argv[1]
target = 'infl_tp' + str(h)
unrate_target = 'unrate_tp' + str(h)

test_start = '1990-01-01'
test_finish = '2019-12-01'

years = list(range(1989,2019))

y = pd.read_csv(DATA_DIR + 'inflfc_targets.csv', index_col=0)
y_unrate = pd.read_csv(DATA_DIR + 'delta_unrate.csv', index_col='date')
y = y[target].to_frame()

y_test = y.loc[test_start:test_finish].values.reshape(-1,)
y_unrate_test = y_unrate.loc[test_start:test_finish, unrate_target].values.reshape(-1,)

# ******************************** Concat preds ********************************

if CONCAT_PREDS:
    preds_path = PRED_DIR + 'all_models_all_years' + target +  '_predictions.csv'
    print('Concatenating predictions...')
    preds_all_years = pd.DataFrame()
    for year in years:
    
        path = PRED_DIR + 'all_models_' + target + '_' + str(year) + '_predictions.csv'
    
        preds = pd.read_csv(path, index_col='date')
    
        preds_all_years = pd.concat([preds_all_years, preds])
    
    print('Predictions concatenated. Saving...')
    preds_all_years.to_csv(PRED_DIR + 'all_models_all_years' + target +  '_predictions.csv')
    
    preds_path = PRED_DIR + 'all_models_all_years' + unrate_target +  '_predictions.csv'
    print('Concatenating predictions...')
    unrate_preds_all_years = pd.DataFrame()
    for year in years:
    
        path = PRED_DIR + 'all_models_' + unrate_target + '_' + str(year) + '_predictions.csv'
    
        preds = pd.read_csv(path, index_col='date')
    
        unrate_preds_all_years = pd.concat([unrate_preds_all_years, preds])
    
    print('Predictions concatenated. Saving...')
    unrate_preds_all_years.to_csv(preds_path)
    
    
    # Get lstm preds from different folder
    # PRED_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions3/"
    
    
    preds_path = PRED_DIR + 'all_LSTMs_all_years_' + target +  '_predictions.csv'
    print('Concatenating LSTM predictions...')
    lstm_preds_all_years = pd.DataFrame()
    for year in years:
    
        path = PRED_DIR + 'all_LSTMs_' + target + '_' + str(year) + '_predictions.csv'
    
        preds = pd.read_csv(path, index_col=0)
    
        lstm_preds_all_years = pd.concat([lstm_preds_all_years, preds])
    
    print('Predictions concatenated. Saving...')
    lstm_preds_all_years.to_csv(preds_path)
    
    # Concatenate unrate predictions
    preds_path = PRED_DIR + 'all_MT-LSTMs_all_years_' + unrate_target +  '_predictions.csv'
    print('Concatenating unemployment rate change predictions...')
    mt_unrate_preds_all_years = pd.DataFrame()
    for year in years:
    
        path = PRED_DIR + 'all_MT-LSTMs_' + unrate_target + '_' + str(year) + '_predictions.csv'
    
        preds = pd.read_csv(path, index_col=0)
    
        mt_unrate_preds_all_years = pd.concat([mt_unrate_preds_all_years, preds])
    
    print('Predictions concatenated. Saving...')
    mt_unrate_preds_all_years.to_csv(preds_path)



unrate_preds_all_years = pd.concat([unrate_preds_all_years,mt_unrate_preds_all_years], axis=1)
preds_all_years = pd.concat([preds_all_years, mt_unrate_preds_all_years], axis=1)

# ******************************** Evaluate ********************************

preds_90s = preds_all_years.loc[test_start:'1999-12-01']
y_test_90s = y.loc[test_start:'1999-12-01'].values.reshape(-1,)

preds_00s = preds_all_years.loc['2000-01-01':'2009-12-01']
y_test_00s = y.loc['2000-01-01':'2009-12-01'].values.reshape(-1,)

preds_10s = preds_all_years.loc['2010-01-01':test_finish]
y_test_10s =  y.loc['2010-01-01':test_finish].values.reshape(-1,)

# Get results
results_90s = score_models(y_true=y_test_90s, preds_df=preds_90s)
results_00s = score_models(y_true=y_test_00s, preds_df=preds_00s)
results_10s = score_models(y_true=y_test_10s, preds_df=preds_10s)
results_all = score_models(y_true=y_test, preds_df=preds_all_years)
unrate_results = score_models(y_true=y_unrate_test, preds_df=unrate_preds_all_years)

print('Evaluation done. Saving...')
results_90s.to_csv(RESULTS_DIR + '1990s_results_' +  target + '.csv')
results_00s.to_csv(RESULTS_DIR + '2000s_results_' +  target + '.csv')
results_10s.to_csv(RESULTS_DIR + '2010s_results_' +  target + '.csv')
results_all.to_csv(RESULTS_DIR + 'allyears_results_' +  target + '.csv')
unrate_results.to_csv(RESULTS_DIR + 'allyears_results_' +  unrate_target + '.csv')
preds_all_years.to_csv(PRED_DIR + target + 'master_predictions.csv')
unrate_preds_all_years.to_csv(PRED_DIR + unrate_target + 'master_predictions.csv')
print('Complete.')


