import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys

FEATIMP_FOLDER = "/lustre06/project/6070422/rproner/InflationForecasting/FeatureImportances/"

h = sys.argv[1]

years = list(range(1989, 2019))

all_featimp = pd.DataFrame()
for year in years:

    featimp = pd.read_csv(
        FEATIMP_FOLDER + 'E-MT-LSTM_32-32-32---32-8d-8d-V2_infl_tp' + str(h) + '_' + str(year) + 'inc_rmse.csv',
        index_col=0
    )

    all_featimp = pd.concat([all_featimp, featimp])

# all_featimp.index = years.append(2019)

# featimp_90s = all_featimp.loc[1990:1999]
# featimp_00s = all_featimp.loc[2000:2009]
# featimp_10s = all_featimp.loc[2010:2019]

all_featimp_mean = all_featimp.mean().sort_values(ascending=False)
# featimp_90s_mean = featimp_90s.mean().sort_values(ascending=False)
# featimp_00s_mean = featimp_00s.mean().sort_values(ascending=False)
# featimp_10s_mean = featimp_10s.mean().sort_values(ascending=False)

sns.set()

featimp_dict = {
    # '90s': featimp_90s_mean,
    # '00s': featimp_00s_mean,
    # '10s': featimp_10s_mean,
    'all': all_featimp_mean
}
for period in featimp_dict.keys():

    importances = featimp_dict[period]

    plt.clf()
    plt.bar(importances.head(20).index, importances.head(20))
    plt.xticks(rotation=45)
    plt.xlabel('Variable')
    plt.ylabel('Increase in RMSE')
    plt.savefig(FEATIMP_FOLDER + 'E-MT-LSTM_32-32-32---32-8d-8d-V2' + '_' + period + 'infl_tp' + str(h) + 'featimp_plot.png')

    if period == 'all':
        importances.to_csv(FEATIMP_FOLDER + 'E-MT-LSTM_32-32-32---32-8d-8d-V2' + 'infl_tp' + str(h) + 'importance.csv')
