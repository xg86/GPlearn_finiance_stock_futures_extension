"""
Market or Risky Price = sum(1 to n)((CF_1to_n)/(1+(r+s)/m)^nm))
Author: Oluwaseyi Awoga
IDE: CS50 IDE on Cloud 9/AWS
Topic: zSpread on Risky Bond Versus Benchmark Treasury Bond
Location: Milky-Way Galaxy
"""
from numpy import array
import numpy as np
from scipy.optimize import fsolve

issuePrice = 106.0281
cash_flows = [3.7,
3.7,
3.7,
3.7,
3.7,
3.7,
3.7,
3.7,
103.7]

payment_dates = [0.51,
1.51,
2.51,
3.51,
4.51,
5.51,
6.51,
7.51,
8.51]

rates = [0.019781,
0.021227,
0.022549,
0.023758,
0.024866,
0.025881,
0.026812,
0.027666,
0.028450]

def calc_pv(zs):
    # Calculate the present value of the bond's cash flows
    pv = np.sum([cf / ((1 + rate + zs) ** t) for cf, t, rate in zip(cash_flows, payment_dates, rates)])
    return pv

def optimizationfunc(spread):
    a = issuePrice
    b = calc_pv(spread)
    return b - a

solutions = fsolve(optimizationfunc,[0.4/100],xtol=1.49012e-08,)
spreadtoUse = solutions[0]
print('cash_flows: '+str(len(cash_flows)))
print('payment_dates: '+str(len(payment_dates)))
print('rates: '+str(len(rates)))
print("The zSpread on a Risky Bond is: ")
print(spreadtoUse*10000)
print()









