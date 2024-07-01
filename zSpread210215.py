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

#issuePrice = 975
issuePrice = 102.2437
#maturity = 2
#r = 0.03 #risk-free rate
#c = 0.03  #coupon
#notional = 100
#numberOfTotalPymnts = 8 #(2 years * 4 payments in a year)

#cash_flows = [120, 120, 120, 1120]
cash_flows = [3.12,
3.12,
3.12,
3.12,
3.12,
3.12,
3.12,
3.12,
3.12,
103.12]
#payment_dates = [1, 2, 3, 4]
payment_dates = [0.41,
1.41,
2.41,
3.41,
4.41,
5.41,
6.41,
7.41,
8.41,
9.41]
#rates = [0.05, 0.06, 0.065, 0.07]
rates = [0.019627,
0.021086,
0.022421,
0.023641,
0.024758,
0.025783,
0.026722,
0.027583,
0.028374,
0.0291]

def calc_pv(zs):
    # Calculate the present value of the bond's cash flows
    pv = np.sum([cf / ((1 + rate + zs) ** t) for cf, t, rate in zip(cash_flows, payment_dates, rates)])
    return pv

def optimizationfunc(spread):
    a = issuePrice
    print(">>>>>>>>>>>>optimizationfunc spread %s " % (spread))
    b = calc_pv(spread)
    return b - a

solutions = fsolve(optimizationfunc,[0.4/100],xtol=1.49012e-08,)
spreadtoUse = solutions[0]
print("The zSpread on a Risky Bond is: ")
print(spreadtoUse)
print()









