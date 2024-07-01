from numpy import array
import numpy as np
from scipy.optimize import fsolve

issuePrice = 106.1334
cash_flows = [3.7,
3.7,
3.7,
3.7,
3.7,
3.7,
3.7,
103.7]

payment_dates = [0.46,
1.46,
2.46,
3.46,
4.46,
5.46,
6.46,
7.46]

'''
=YieldCurveZeroRate
10/20/2023
10/19/2024
10/20/2025
10/20/2026
10/20/2027
10/19/2028
10/20/2029
10/20/2030
'''
rates = [0.020120127,
0.021985993,
0.023531336,
0.024481973,
0.025285881,
0.026065438,
0.026810609,
0.027298793]

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
print(spreadtoUse)
print()









