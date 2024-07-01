import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import graphviz

from scipy.stats import rankdata
import scipy.stats as stats

from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import baostock as bs
from ta.volume import VolumeWeightedAveragePrice
import statsmodels.api as sm
from scipy.stats.mstats import zscore
import talib

from scipy.stats import spearmanr
import datetime
from numba import jit
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import math
import cloudpickle
import matplotlib.pyplot as plt
#%matplotlib inline
import matplotlib
print(matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

filename = 'gplearn_gx_bond_factors.pkl'
with open(filename, 'rb') as f:
    est_gp = cloudpickle.load(f)
best_programs = est_gp._best_programs
best_programs_dict = {}

for p in best_programs:
    factor_name = 'alpha_' + str(best_programs.index(p) + 1)
    best_programs_dict[factor_name] = {'fitness': p.fitness_, 'expression': str(p), 'depth': p.depth_,
                                       'length': p.length_}

best_programs_dict = pd.DataFrame(best_programs_dict).T
best_programs_dict = best_programs_dict.sort_values(by='fitness')
print(best_programs_dict)

def alpha_factor_graph(num):
    # 打印指定num的表达式图

    factor = best_programs[num - 1]
    print(factor)
    print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))

    dot_data = factor.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('alpha_factor_graph_bond.png', format='png', cleanup=True)

    return graph


#graph1 = alpha_factor_graph(1)
#print(graph1)

factor = best_programs[0]
print(factor)
print('fitness: {0}, depth: {1}, length: {2}'.format(factor.fitness_, factor.depth_, factor.length_))


#@jit
def _factor_backtest(factor_perd, market_price):
    pred = pd.Series(factor_perd.flatten()).fillna(0)
    evaluation = []
    slippage = 4
    shares = 25
    comission = 0.00025

    backtest_data = market_price
    trades = pred

    short_open = 0
    long_open = 0
    held_long = 0
    held_short = 0
    profit = []
    profit_temp = 0
    #GX
    #initial_assets = shares * max(backtest_data.open.values)
    initial_assets = 0
    initial_cash = shares * max(backtest_data.open.values)
    net_worth = [initial_cash]

    for i in range(len(trades)):
        current_pred = trades.iloc[i]
        current_close = backtest_data.iloc[i].open.astype('float')
        # open long
        if current_pred >= 0.75 and held_long == 0 and held_short == 0:
            held_long = 1
            held_short = 0
            long_open = current_close + slippage
            short_open = 0
            # print('open long')

        # hold long
        elif current_pred >= 0.75 and held_long == 1 and held_short == 0:
            # print('hold long')
            pass

            # open long and close short
        elif current_pred >= 0.75 and held_long == 0 and held_short == 1:
            held_long = 1
            # close short and calculate profit
            held_short = 0
            profit_temp = (short_open - (current_close + slippage)) * shares * (1 - comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            # open long
            short_open = 0
            long_open = current_close + slippage
            # print('open long and close short')

        # open short
        elif current_pred <= -0.75 and held_long == 0 and held_short == 0:
            held_long = 0
            held_short = 1
            long_open = long_open
            short_open = current_close + slippage
            profit = profit
            # print('open short')

        # keep short
        elif current_pred <= -0.75 and held_long == 0 and held_short == 1:
            # print('keep short')
            pass

        # close long and open short
        elif current_pred <= -0.75 and held_long == 1 and held_short == 0:
            # close long
            held_long = 0
            held_short = 1
            profit_temp = ((current_close - slippage) - long_open) * shares * (1 - comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            # open short
            long_open = 0
            short_open = current_close - slippage
            profit = profit
            # print('close long and open short')

        # closeout long
        elif current_pred < 0.75 and current_pred > -0.75 and held_long == 1 and held_short == 0:
            held_long = 0
            held_short = 0
            profit_temp = ((current_close - slippage) - long_open) * shares * (1 - comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            short_open = 0
            long_open = 0
            # print('closeout long')

        # closeout short
        elif current_pred < 0.75 and current_pred > -0.75 and held_long == 0 and held_short == 1:
            held_long = 0
            held_short = 0
            profit_temp = (short_open - (current_close + slippage)) * shares * (1 - comission)
            initial_cash += profit_temp
            profit.append(profit_temp)
            net_worth.append(initial_cash)
            short_open = 0
            long_open = 0
            # print('closeout short')

    total_return = (initial_cash - initial_assets) / initial_assets
    print('总收益率', total_return)
    shaprpe_df = pd.Series(profit)
    sharpe_temp = (shaprpe_df - shaprpe_df.shift(1)) / shaprpe_df.shift(1)
    sharpe = sharpe_temp.mean() / sharpe_temp.std() * np.sqrt(len(profit))

    a = np.maximum.accumulate(net_worth)
    l = np.argmax((np.maximum.accumulate(net_worth) - net_worth) / np.maximum.accumulate(net_worth))
    k = np.argmax(net_worth[:l])
    max_draw = (net_worth[k] - net_worth[l]) / (net_worth[l])
    print('最大回撤', max_draw)

    win_count = 0
    loss_count = 0
    initial_profit = 0
    for i in range(len(net_worth)):
        current_profit = net_worth[i]
        if i == 0:
            if current_profit > initial_assets:
                win_count += 1
            else:
                loss_count += 1
        else:
            last_profit = net_worth[i - 1]
            if current_profit > last_profit:
                win_count += 1
            else:
                loss_count += 1
    win_rate = win_count / len(net_worth)
    print('胜率', win_rate)

    total_gain = 0
    total_loss = 0
    for i in range(len(profit)):
        if profit[i] > 0:
            total_gain += profit[i]
        else:
            total_loss += profit[i]
    gain_loss_ratio = (total_gain / win_count) / (abs(total_loss) / loss_count)
    print('盈亏比', gain_loss_ratio)

    result = total_return * np.nan_to_num(sharpe, nan=1) * win_rate * gain_loss_ratio * (1 - max_draw)

    x = np.array(net_worth).reshape(len(net_worth), )
    y = np.arange(len(net_worth))

    #plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    fig = plt.figure(figsize=(16, 9))
    plt.plot(y, x)
    plt.title('因子资金曲线', fontsize=20)
    plt.xlabel('交易次数', fontsize=20)
    plt.ylabel('账户净值', fontsize=20)
    plt.show()

    return result

data_df = pd.read_csv('bond.csv')

fields = ['open','close','high','low','volume','money']
length = []

df = data_df.copy()
df.high = df.high.values.astype('float64')
df.low = df.low.values.astype('float64')
df.close = df.close.values.astype('float64')
df.open = df.open.values.astype('float64')
df.volume = df.volume.values.astype('float64')
df.amount = df.money.values.astype('float64')
#df.amount = df.close * df.volume
print(df.isnull().any())


df['1_day_return'] = df.open.pct_change(1).shift(-1)
df = df.dropna()

train_data=df[(df.date<'2015-12-31')]
train_data=train_data.reset_index(drop=True)

test_data=df[(df.date>='2015-12-31')]
test_data=test_data.reset_index(drop=True)

X_train = train_data.drop(columns=['date','1_day_return']).to_numpy()
y_train = train_data['1_day_return'].values

#X_train, X_train.shape, y_train, y_train.shape

X_test = test_data.drop(columns=['date', '1_day_return']).to_numpy()
y_test = test_data['1_day_return'].values

factors_pred = est_gp.transform(X_train)
print(factors_pred.shape)
print(X_train.shape)

pred_data = pd.DataFrame(factors_pred).T.T
#pred_data, pred_data.iloc[:,[0]]

#quanConnect
new_features = np.hstack((X_train, factors_pred))
model = SymbolicRegressor()
model.fit(new_features, y_train)
# Get next prediction
prediction = model.predict(new_features)
prediction = float(prediction.flatten()[-1])

_factor_backtest(pred_data.iloc[:,[0]].values, train_data)


factors_pred_1 = est_gp.transform(X_test)
print(factors_pred_1.shape)
print(X_test.shape)

pred_data_1 = pd.DataFrame(factors_pred_1).T.T
pred_data_1, pred_data_1.iloc[:,[0]]

_factor_backtest(pred_data_1.iloc[:,[0]].values, test_data)