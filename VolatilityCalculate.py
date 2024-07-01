import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import zscore

data_file = 'bond-trade.csv'
data_df = pd.read_csv(data_file)
data_df['cfets-mean'] = data_df['cfets'].mean()
data_df['cfets-std'] = data_df['cfets'].std()
data_df['cfets-z-score-c'] = (data_df['cfets'] - data_df['cfets-mean'])/data_df['cfets-std']
data_df['cfets-z-score'] = stats.zscore(data_df['cfets'])
data_df['broker-z-score'] = stats.zscore(data_df['broker'])
print(data_df.head(3))

pc50, pc70, pc85,pc90, pc100 =np.percentile(data_df['cfets'],[50,70,85,95,100])
pb50, pb70, pb85,pb90, pb100 =np.percentile(data_df['broker'],[50,70,85,95,100])

print(pc50)
print(pc70)
print(pc85)
print(pc90)
print(pc100)

print(pb50)
print(pb70)
print(pb85)
print(pb90)
print(pb100)

def set_percentile(x, p70, p85, p90, p100):
    if x >= p100:
        return 100
    elif x > p90:
        return 90
    elif x > p85:
        return 85
    elif x > p70:
        return 70
    else:
        return 50

def add_p_z(df, col1, col2):
    if df[col1]  >= 100:
        return df[col1]
    else:
        return df[col1] + df[col2]

data_df['cfets-p'] =  data_df['cfets'].apply(set_percentile, args=(pc70, pc85, pc90, pc100))

data_df['broker-p'] = data_df['broker'].apply(set_percentile, args=(pb70, pb85, pb90, pb100))

data_df['cfets-p-z'] = data_df.apply(add_p_z, args=('cfets-p', "cfets-z-score"), axis=1)
data_df['broker-p-z'] = data_df.apply(add_p_z, args=('broker-p', "broker-z-score"), axis=1)
data_df['Vol'] = (data_df['cfets-p-z'] +data_df['broker-p-z'] )/2

data_df['cfets-zs-grp'] = data_df.groupby(['cfets-p']).cfets.transform(lambda x : zscore(x,ddof=1))
data_df['broker-zs-grp'] = data_df.groupby(['broker-p']).broker.transform(lambda x : zscore(x,ddof=1))

data_df['cfets-p-z-grp'] = data_df.apply(add_p_z, args=('cfets-p', "cfets-zs-grp"), axis=1)
data_df['broker-p-z-grp'] = data_df.apply(add_p_z, args=('broker-p', "broker-zs-grp"), axis=1)
data_df['Vol-z-grp'] = (data_df['cfets-p-z-grp'] +data_df['broker-p-z-grp'] )/2

data_df.to_csv(data_file+'_'+"liquidity_rank"+".csv")