import numpy as np
import scipy.optimize as optimize

# Define the bond's cash flows and payment dates
cash_flows = [120, 120, 120, 1120]
payment_dates = [1, 2, 3, 4]
#rates = [0.05, 0.06, 0.065, 0.07]

def calc_pv(cash_flows, rates):
    # Calculate the present value of the bond's cash flows
    pv = np.sum([cf / ((1 + rate) ** t) for cf, t, rate in zip(cash_flows, payment_dates, rates)])
    return pv

def calc_zspread(cash_flows, payment_dates, bond_price, guess=0.00004):
    # Define the function that we want to minimize
    def spread_func(spread):
        #rates = [spread] * len(cash_flows)
        rates = [0.05, 0.06, 0.065, 0.07]
        pv = calc_pv(cash_flows, rates)
        return bond_price - pv

    # Use the optimize.newton method to solve for the Z-spread
    result = optimize.newton(spread_func, guess)
    return result

# Calculate the Z-spread of a bond with a price of 95.5
bond_price = 975
z_spread = calc_zspread(cash_flows, payment_dates, bond_price)
print("The Z-spread is:", z_spread)