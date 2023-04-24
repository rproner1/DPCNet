# Libraries

import pandas as pd 
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.tsa.api as smt
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

# Folder paths

PRED_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Predictions10/"
RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results10/"


# Helper functions

def concat_pred_all_targets(var_name):
    
    """
    Concatenates all dataframes of predictions into a dictionary where each key:value pair is target:pred_df. 
    """

    target_pred_dfs = {}
    targets = [var_name + '_tp' + str(h) for h in range(1,12+1)]
    for target in targets:
        
        if var_name == 'infl':
            # lstm_pred = pd.read_csv(PRED_FOLDER + 'all_LSTMs_all_years_' + target + '_predictions.csv')
            # other_pred = pd.read_csv(PRED_FOLDER + 'all_models_all_years' + target + '_predictions.csv')
            # pred = pd.concat([lstm_pred, other_pred], axis=1)
            pred = pd.read_csv(PRED_FOLDER + target + 'master_predictions.csv', index_col=0)

        elif var_name == 'unrate':
            pred1 = pd.read_csv(PRED_FOLDER + 'all_MT-LSTMs_all_years_' + target + '_predictions.csv', index_col=0)
            pred2 = pd.read_csv(PRED_FOLDER + 'all_models_all_years' + target + '_predictions.csv', index_col=0)
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


def diebold_mariano(y_true, y_pred1, y_pred2, loss='squared'):
    
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
    
    test_statistic = d_bar / se_d
    
    p_value = norm.sf(abs(test_statistic))*2
    
    return test_statistic, p_value


def plot_diebold_mariano(y_true, forecasts, loss='squared', save=False, savepath=''):
    
    pval_mat = np.empty((forecasts.shape[1],forecasts.shape[1]))
    for i in range(forecasts.shape[1]):
        for j in range(forecasts.shape[1]):
            if i == j:
                p_value = 1.0
            else:
                _, p_value = diebold_mariano(y_true, 
                                                           forecasts.iloc[:,i].values.reshape(-1,), 
                                                           forecasts.iloc[:,j].values.reshape(-1,),
                                                           loss=loss)
                
            pval_mat[i, j] = p_value
    
    pval_df = pd.DataFrame(pval_mat, columns=forecasts.columns, index=forecasts.columns)
    mask_mat = np.triu(pval_df)
    sns.heatmap(pval_df, center=0.5, cmap='viridis', cbar_kws={"ticks":[0.01,0.05,0.10,0.2,0.5,1.0]}, mask=mask_mat)
    if save:
        plt.savefig(savepath)
        
    plt.show()
    
    return None


def score_models(y_true, preds_df):

    results = {
        'Model': preds_df.columns,
        'RMSE': [],
        'MAE': [],
        'pval':[]
    }

    rw_pred = preds_df['RW'].values.reshape(-1,)

    for model in preds_df.columns:

        y_pred = preds_df[model].values.reshape(-1,)

        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)
        if model == 'RW':
            pval = 1.0
        else:
            _, pval = diebold_mariano(y_true, rw_pred, y_pred)

        results['RMSE'].append(rmse)
        results['MAE'].append(mae)
        results['pval'].append(pval)

    results_df = pd.DataFrame(results).set_index('Model')
    rel_results_df = results_df.copy()
    rel_results_df[['RMSE', 'MAE']] = rel_results_df[['RMSE', 'MAE']].apply(lambda row: row / rel_results_df.loc['RW',['RMSE', 'MAE']], axis=1)

    return results_df, rel_results_df


# Read data
y = pd.read_csv('inflfc_targets.csv', index_col='date')
y.index = pd.to_datetime(y.index)
rw = pd.read_csv('acc_rw.csv', index_col='date')
rw.index = pd.to_datetime(rw.index)
ur_rw = pd.read_csv('acc_ur_rw.csv', index_col='date')
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


y_unrate = pd.read_csv('delta_unrate.csv', index_col='date')
y_unrate.index = pd.to_datetime(y_unrate.index)
y_unrate_test = y_unrate.loc[test_start:test_finish]

# Get targets df
target_pred_dfs_dict = concat_pred_all_targets(var_name='infl')

# Models to compare
models_to_compare = [
    'AR_p',
    'Linear Regression',
    'Ridge Regression',
    'Lasso',
    'Elastic Net',
    'Random Forest',
    'Extremely Randomized Trees',
    'Gradient Boosted Trees',
    'E-LSTM-V2_32-32-32-32-8d-8d',
    'E-MT-LSTM_32-32-32---32-8d-8d-V2'
]

unrate_models = [
    'AR_p',
    'Elastic Net',
    'Random Forest',
    'E-MT-LSTM_32-32-32---32-8d-8d-V2'
]


# Inflation results

for acc_h in [1,3,6,12]:

    # Compute true accumulated inflation
    y_true = y_test[['infl_tp' + str(h) for h in range(1,acc_h+1)]].sum(axis=1)

    # Compute accumulated forecasts
    y_pred_df = get_accumulated_pred(
        var_name='infl', 
        models=models_to_compare, 
        target_pred_dfs=target_pred_dfs_dict,
        acc_h=acc_h
    )

    y_pred_df.index = y_test.index
    y_pred_df['RW'] = rw_test['RW' + str(acc_h)].values.reshape(-1,)

    # Evaluate forecasts
    results, relresults = score_models(y_true.values.reshape(-1,), y_pred_df)

    # Save results
    results.to_csv(RESULTS_FOLDER + 'acc' + str(acc_h) + '_results.csv')
    relresults.to_csv(RESULTS_FOLDER + 'acc' + str(acc_h) + '_relresults.csv')
    y_pred_df.to_csv(PRED_FOLDER + 'acc' + str(acc_h) + '_predictions.csv')

    # Plot DM test matrix
    plot_diebold_mariano(y_true.values.reshape(-1,), 
                         y_pred_df, 
                         save=True, 
                         savepath='acc' + str(acc_h) + '_DMtest.png')

    # y_true_rec = y_true.loc[recessions]
    # y_pred_df_rec = y_pred_df.loc[recessions]

    # Test in recessions only
    # for yr in recessions_dict.keys():

    #     date_ind = recessions_dict[yr]

    #     y_true_rec = y_true.loc[date_ind]
    #     y_pred_df_rec = y_pred_df.loc[date_ind]

    #     results, relresults = score_models(y_true_rec.values.reshape(-1,), y_pred_df_rec)

    #     results.to_csv(RESULTS_FOLDER + 'acc' + str(acc_h) + '_results_rec' + yr + '.csv')
    #     relresults.to_csv(RESULTS_FOLDER + 'acc' + str(acc_h) + '_relresults_rec' + yr + '.csv')

    #     plot_diebold_mariano(y_true_rec.values.reshape(-1,), 
    #                          y_pred_df_rec, 
    #                          save=True, 
    #                          savepath='acc' + str(acc_h) + '_DMtest_rec' + yr + '.png')


# Get all unrate predictions for each unrate forecast horizon
unrate_pred_dfs_dict = concat_pred_all_targets(var_name='unrate')

# Unrate results
for acc_h in [1,3,6,12]:

    # Compute true accumulated change in the unemployment rate
    y_true = y_unrate_test[['unrate_tp' + str(h) for h in range(1, acc_h+1)]].sum(axis=1)

    # Compute accumulated change forecasts
    y_pred_df = get_accumulated_pred(
        var_name='unrate', 
        models=unrate_models, 
        target_pred_dfs=unrate_pred_dfs_dict,
        acc_h=acc_h
    )

    y_pred_df.index = y_unrate_test.index
    y_pred_df['RW'] = ur_rw_test['RW' + str(acc_h)].values.reshape(-1,)

    # Evaluate forecasts
    results, relresults = score_models(y_true.values.reshape(-1,), y_pred_df)

    # Save results
    results.to_csv(RESULTS_FOLDER + 'unrate_acc' + str(acc_h) + '_results.csv')
    relresults.to_csv(RESULTS_FOLDER + 'unrate_acc' + str(acc_h) + '_relresults.csv')
    y_pred_df.to_csv(PRED_FOLDER + 'unrate_acc' + str(acc_h) + '_predictions.csv')

    # Plot diebold mariano matrix
    plot_diebold_mariano(y_true.values.reshape(-1,), 
                         y_pred_df, 
                         save=True, 
                         savepath='unrate_' + str(acc_h) + 'm_DMtest.png')

    # Recessions
    # y_true_rec = y_true.loc[recessions]
    # y_pred_df_rec = y_pred_df.loc[recessions]

    
    # for yr in recessions_dict.keys():

    #     date_ind = recessions_dict[yr]

    #     y_true_rec = y_true.loc[date_ind]
    #     y_pred_df_rec = y_pred_df.loc[date_ind]

    #     results, relresults = score_models(y_true_rec.values.reshape(-1,), y_pred_df_rec)

    #     results.to_csv(RESULTS_FOLDER + 'acc' + str(acc_h) + '_results_rec' + yr + '.csv')
    #     relresults.to_csv(RESULTS_FOLDER + 'acc' + str(acc_h) + '_relresults_rec' + yr + '.csv')

    #     plot_diebold_mariano(y_true_rec.values.reshape(-1,), 
    #                          y_pred_df_rec, 
    #                          save=True, 
    #                          savepath='acc' + str(acc_h) + '_DMtest_rec' + yr + '.png')
        
