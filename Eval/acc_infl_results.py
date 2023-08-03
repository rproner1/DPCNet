# ******************************** Imports ********************************

import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.tsa.api as smt
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

# ******************************** Parameters ********************************

RUN_LOCALLY = True

# ******************************** Directories ********************************

if RUN_LOCALLY:
    PRED_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Predictions3/'
    RESULTS_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Results3/'
    DATA_DIR = 'C:/Users/Robpr/OneDrive/Documents/Projects/InflationForecasting/Data/'
else:
    PRED_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions3/"
    RESULTS_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Results3/"
    DATA_DIR = "/lustre06/project/6070422/rproner/InflationForecasting/Data/"


# ******************************** Functions ********************************
def concat_pred_all_targets(var_name):
    
    """
    Concatenates all dataframes of predictions into a dictionary where each key:value pair is target:pred_df. 
    """

    target_pred_dfs = {}
    targets = [var_name + '_tp' + str(h) for h in range(1,12+1)]
    for target in targets:
        
        if var_name == 'infl':
            # lstm_pred = pd.read_csv(PRED_DIR + 'all_LSTMs_all_years_' + target + '_predictions.csv')
            # other_pred = pd.read_csv(PRED_DIR + 'all_models_all_years' + target + '_predictions.csv')
            # pred = pd.concat([lstm_pred, other_pred], axis=1)
            pred = pd.read_csv(PRED_DIR + target + 'master_predictions.csv', index_col=0)

        elif var_name == 'unrate':
            pred1 = pd.read_csv(PRED_DIR + 'all_MT-LSTMs_all_years_' + target + '_predictions.csv', index_col=0)
            pred2 = pd.read_csv(PRED_DIR + 'all_models_all_years' + target + '_predictions.csv', index_col=0)
            pred = pd.concat([pred2,pred1],axis=1)

        else:
            print(f'{var_name} is not a valid variable name. Valid variable names are "infl" and "unrate".')
            break

        target_pred_dfs[target] = pred

    return target_pred_dfs


def get_accumulated_pred(var_name, models, target_pred_dfs, acc_h=12):

    
    """
    Accumulates the one-month inflation predictions to get predictions for accumulated inflation.

    Paramerters
    -----------
    var_name: str {'infl', 'unrate'}
    models: list of str
        the model names to look at
    target_pred_dfs: dict
        A dictionary of dataframes of predictions, where keys are the targets.
    acc_h: int default=12
        The accumulated horizon.
    Returns
    ----------
    acc_fc_all_models: dataframe
        Accumulated forecasts of all models
    """

    acc_fc_all_models = pd.DataFrame()
    for model in models:
        model_pred_all_targets = pd.DataFrame()
        for target in [var_name + '_tp' + str(h) for h in range(1,acc_h+1)]:
            model_pred_all_targets[target] = target_pred_dfs[target][model].values.reshape(-1,)
        
        model_pred_all_targets.index = target_pred_dfs[target].index
        acc_target = 'acc_'+ str(acc_h)
        model_pred_all_targets[acc_target] = model_pred_all_targets.sum(axis=1)
        acc_fc_all_models[model] = model_pred_all_targets[acc_target].values.reshape(-1,)
        
    acc_fc_all_models.index = model_pred_all_targets.index
    
    return acc_fc_all_models


def diebold_mariano(y_true, y_pred1, y_pred2, loss='squared', verbose=False):
    
    if loss == 'squared':
        loss1 = (y_true - y_pred1)**2
        loss2 = (y_true - y_pred2)**2
    elif loss == 'absolute':
        loss1 = np.abs(y_true - y_pred1)
        loss2 = np.abs(y_true - y_pred2)
    else:
        ValueError('Not a valid loss function')
        
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


def plot_diebold_mariano(y_true, forecasts, loss='squared', save=False, savepath=''):
    
    tstat_mat = np.empty((forecasts.shape[1],forecasts.shape[1]))
    pval_mat = np.empty((forecasts.shape[1],forecasts.shape[1]))
    for i in range(forecasts.shape[1]):
        for j in range(forecasts.shape[1]):
            
            verbose=False
            # if i==4:
            #     verbose=True
            
            if i == j:
                p_value = 1.0
                tstat=0.0
            else:
                
                tstat, p_value = diebold_mariano(
                    y_true, 
                    forecasts.iloc[:,i].values.reshape(-1,), 
                    forecasts.iloc[:,j].values.reshape(-1,),
                    loss=loss,
                    verbose=verbose
                )
                
            pval_mat[i, j] = p_value
            tstat_mat[i, j] = tstat
    pval_df = pd.DataFrame(pval_mat, columns=forecasts.columns, index=forecasts.columns)
    tstat_df = pd.DataFrame(tstat_mat, columns=forecasts.columns, index=forecasts.columns)
    mask_mat = np.triu(pval_df)
    
    plt.clf()
    sns.heatmap(tstat_df, annot=tstat_df, center=0.0, cmap='viridis', mask=mask_mat)
    if save:
        plt.savefig(savepath, dpi=300)
        
    plt.show()
    
    return tstat_df, pval_df


def oos_r2_score(y_true, y_pred, hist_means):

    mse = mean_squared_error(y_true, y_pred)
    mse_trainmean = mean_squared_error(y_true, np.full_like(y_true, hist_means))
    r2 = 1 - mse/mse_trainmean

    return r2

def score_models(y_true, preds_df):

    results = {
        'Model': preds_df.columns,
        'RMSE': [],
        'MAE': [],
        'R2': []
    }

    rw_pred = preds_df['RW'].values.reshape(-1,)
    histmean_pred = preds_df['Mean'].values.ravel()

    for model in preds_df.columns:

        y_pred = preds_df[model].values.reshape(-1,)

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = oos_r2_score(y_true, y_pred, histmean_pred)

        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['R2'].append(r2)
        
    results_df = pd.DataFrame(results).set_index('Model')
    rel_results_df = results_df.copy()
    rel_results_df[['RMSE', 'MAE']] = rel_results_df[['RMSE', 'MAE']].apply(lambda row: row / rel_results_df.loc['RW',['RMSE', 'MAE']], axis=1)

    return results_df, rel_results_df


# Read data
y = pd.read_csv(DATA_DIR + 'inflfc_targets.csv', index_col='date')
y.index = pd.to_datetime(y.index)
rw = pd.read_csv(DATA_DIR + 'acc_rw.csv', index_col='date')
rw.index = pd.to_datetime(rw.index)
ur_rw = pd.read_csv(DATA_DIR + 'acc_ur_rw.csv', index_col='date')
ur_rw.index = pd.to_datetime(ur_rw.index)

# Define y_true for acc inflation

test_start = '1990-01-01'
test_finish = '2019-12-01'

# Define recession dates
recession1 = pd.date_range(start='1990-07-01', end='1991-03-01', freq='MS')
recession2 = pd.date_range(start='2001-03-01', end='2001-11-01', freq='MS')
recession3 = pd.date_range(start='2007-12-01', end='2009-06-01', freq='MS')
recessions = recession1.union(recession2).union(recession3)

recessions_dict = {
    '1990': recession1,
    '2001': recession2,
    '2007': recession3
}


y_test = y.loc[test_start:test_finish]
rw_test = rw.loc[test_start:test_finish]
ur_rw_test = ur_rw.loc[test_start:test_finish]


y_unrate = pd.read_csv(DATA_DIR + 'delta_unrate.csv', index_col='date')
y_unrate.index = pd.to_datetime(y_unrate.index)
y_unrate_test = y_unrate.loc[test_start:test_finish]

# Get targets df
target_pred_dfs_dict = concat_pred_all_targets(var_name='infl')

# Models to compare
models_to_compare = [
    'AR_p',
    'HistMean',
    'Linear Regression',
    'Ridge Regression',
    'Lasso',
    'Elastic Net',
    'Random Forest',
    'Extremely Randomized Trees',
    'Gradient Boosted Trees',
    'DPCNet1',
    'DPCNet2',
    'DPCNet3'
]

model_rename_dict = {
    'HistMean': 'Mean',
    'AR_p': 'AR(p)',
    'Linear Regression': 'LR',
    'Ridge Regression': 'RR',
    'Lasso': 'LAS',
    'Elastic Net': 'EN',
    'Random Forest': 'RF',
    'Extremely Randomized Trees': 'XT',
    'Gradient Boosted Trees': 'GBT',
}

models_order = [
    'RW', 'Mean', 'AR(p)', 'LR', 'RR', 'LAS', 'EN',
    'RF', 'XT', 'GBT', 'DPCNet1', 'DPCNet2', 'DPCNet3'
]

# Inflation results

for acc_h in [1,3,6,12]:
    
    print(f'Evaluating {acc_h}-month inflation forecasts...')

    # Compute true accumulated inflation
    y_true = y_test[['infl_tp' + str(h) for h in range(1,acc_h+1)]].sum(axis=1)

    print('Aggregating monthly forecasts...')
    # Compute accumulated forecasts
    y_pred_df = get_accumulated_pred(
        var_name='infl', 
        models=models_to_compare, 
        target_pred_dfs=target_pred_dfs_dict,
        acc_h=acc_h
    )

    y_pred_df.index = y_test.index
    y_pred_df['RW'] = rw_test['RW' + str(acc_h)].values.reshape(-1,)
    
    y_pred_df = y_pred_df.rename(columns=model_rename_dict)
    
    # Order models by complexity
    y_pred_df = y_pred_df.loc[:,models_order]

    print('Evaluating models...')
    # Evaluate forecasts
    results, relresults = score_models(y_true.values.reshape(-1,), y_pred_df)

    # Save results
    results.to_csv(RESULTS_DIR + 'acc' + str(acc_h) + '_results.csv')
    relresults.to_csv(RESULTS_DIR + 'acc' + str(acc_h) + '_relresults.csv')
    y_pred_df.to_csv(PRED_DIR + 'acc' + str(acc_h) + '_predictions.csv')

    print('Performing Diebold Mariano tests...')
    # Plot DM test matrix
    tstats, pvals = plot_diebold_mariano(
        y_true.values.reshape(-1,), 
        y_pred_df, 
        save=True, 
        savepath='acc' + str(acc_h) + '_DMtest.png')

    tstats.to_csv(RESULTS_DIR + 'dm_test_tstats_' + str(acc_h) + '.csv')
    pvals.to_csv(RESULTS_DIR + 'dm_test_pvals_' + str(acc_h) + '.csv')

    y_true_rec = y_true.loc[recessions]
    y_pred_df_rec = y_pred_df.loc[recessions]

    print('Evaluating models in recessions...')
    # Test in recessions only
    for yr in recessions_dict.keys():

        print(f'Evaluating recession in year: {yr}...')
        date_ind = recessions_dict[yr]

        y_true_rec = y_true.loc[date_ind]
        y_pred_df_rec = y_pred_df.loc[date_ind]

        print('Evaluating models...')
        results, relresults = score_models(y_true_rec.values.reshape(-1,), y_pred_df_rec)

        results.to_csv(RESULTS_DIR + 'acc' + str(acc_h) + '_results_rec' + yr + '.csv')
        relresults.to_csv(RESULTS_DIR + 'acc' + str(acc_h) + '_relresults_rec' + yr + '.csv')
        
        print('Performing Diebold Mariano tests...')
        plot_diebold_mariano(y_true_rec.values.reshape(-1,), 
                              y_pred_df_rec,
                              save=False, 
                              savepath='acc' + str(acc_h) + '_DMtest_rec' + yr + '.png')


# Get all unrate predictions for each unrate forecast horizon
# unrate_pred_dfs_dict = concat_pred_all_targets(var_name='unrate')

# # Unrate results
# for acc_h in [1,3,6,12]:

#     # Compute true accumulated change in the unemployment rate
#     y_true = y_unrate_test[['unrate_tp' + str(h) for h in range(1, acc_h+1)]].sum(axis=1)

#     # Compute accumulated change forecasts
#     y_pred_df = get_accumulated_pred(
#         var_name='unrate', 
#         models=models_to_compare, 
#         target_pred_dfs=unrate_pred_dfs_dict,
#         acc_h=acc_h
#     )

#     y_pred_df.index = y_unrate_test.index
#     y_pred_df['RW'] = ur_rw_test['RW' + str(acc_h)].values.reshape(-1,)

#     # Evaluate forecasts
#     results, relresults = score_models(y_true.values.reshape(-1,), y_pred_df)

#     # Save results
#     results.to_csv(RESULTS_DIR + 'unrate_acc' + str(acc_h) + '_results.csv')
#     relresults.to_csv(RESULTS_DIR + 'unrate_acc' + str(acc_h) + '_relresults.csv')
#     y_pred_df.to_csv(PRED_DIR + 'unrate_acc' + str(acc_h) + '_predictions.csv')

#     # Plot diebold mariano matrix
#     plot_diebold_mariano(y_true.values.reshape(-1,), 
#                          y_pred_df, 
#                          save=True, 
#                          savepath='unrate_' + str(acc_h) + 'm_DMtest.png')

    # Recessions
    # y_true_rec = y_true.loc[recessions]
    # y_pred_df_rec = y_pred_df.loc[recessions]

    
    # for yr in recessions_dict.keys():

    #     date_ind = recessions_dict[yr]

    #     y_true_rec = y_true.loc[date_ind]
    #     y_pred_df_rec = y_pred_df.loc[date_ind]

    #     results, relresults = score_models(y_true_rec.values.reshape(-1,), y_pred_df_rec)

    #     results.to_csv(RESULTS_DIR + 'acc' + str(acc_h) + '_results_rec' + yr + '.csv')
    #     relresults.to_csv(RESULTS_DIR + 'acc' + str(acc_h) + '_relresults_rec' + yr + '.csv')

    #     plot_diebold_mariano(y_true_rec.values.reshape(-1,), 
    #                          y_pred_df_rec, 
    #                          save=True, 
    #                          savepath='acc' + str(acc_h) + '_DMtest_rec' + yr + '.png')
        
print('Complete.')