# Data Manipulation Libraries
import pandas as pd 
import numpy as np

RESULTS_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/Results10/"

rmse_all_h = pd.DataFrame()
mae_all_h = pd.DataFrame()
mad_all_h = pd.DataFrame()
ur_rmse_all_h = pd.DataFrame()
ur_mae_all_h = pd.DataFrame()
ur_mad_all_h = pd.DataFrame()
for h in range(1,12+1):

    path = RESULTS_FOLDER + 'allyears_results_' + 'infl_tp' + str(h) + '.csv'
    unrate_path = RESULTS_FOLDER + 'allyears_results_' + 'unrate_tp' + str(h) + '.csv'
    pred_df = pd.read_csv(path, index_col='Model')
    unrate_pred_df = pd.read_csv(unrate_path, index_col='Model')

    rmse_all_h[str(h)] = np.round(pred_df['RMSE'].values, decimals=4)
    mae_all_h[str(h)] = np.round(pred_df['MAE'].values, decimals=4)
    mad_all_h[str(h)] = np.round(pred_df['MAD'].values, decimals=4)
    ur_rmse_all_h[str(h)] = np.round(unrate_pred_df['RMSE'].values, decimals=4)
    ur_mae_all_h[str(h)] = np.round(unrate_pred_df['MAE'].values, decimals=4)
    ur_mad_all_h[str(h)] = np.round(unrate_pred_df['MAD'].values, decimals=4)         

models = pred_df.index
rmse_all_h.index = models
mae_all_h.index = models
mad_all_h.index = models
ur_rmse_all_h.index = models
ur_mae_all_h.index = models
ur_mad_all_h.index = models

rmse_all_h.to_csv(RESULTS_FOLDER + 'all_models_rmse_all_horizons.csv')
mae_all_h.to_csv(RESULTS_FOLDER + 'all_models_mae_all_horizons.csv')
mad_all_h.to_csv(RESULTS_FOLDER + 'all_models_mad_all_horizons.csv')
ur_rmse_all_h.to_csv(RESULTS_FOLDER + 'all_models_unrate_rmse_all_horizons.csv')
ur_mae_all_h.to_csv(RESULTS_FOLDER + 'all_models_unrate_mae_all_horizons.csv')
ur_mad_all_h.to_csv(RESULTS_FOLDER + 'all_models_unrate_mad_all_horizons.csv')