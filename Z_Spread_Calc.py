import numpy as np
import time

start_time = time.time()
debug = True
bond_id = 210215
par_val = 100
cr = .0312
time_to_maturity = 10
market_price = 102.2437
tolerance = 1.48e-8

print('Bond params: '+str(bond_id) +', par:' + str(par_val) + ' coupon:' + str(cr*100) + '% years:' + str(time_to_maturity) + ' mkt_px:' + str(market_price) + ' tolerance:' + str(tolerance))

cf = np.zeros(time_to_maturity)

for i in range(0,time_to_maturity):
    cf[i] = par_val*cr

cf[time_to_maturity-1] += par_val

#TSP Curve

#hardcoded tsp
tsp_curve = np.array([0.019627,
0.021086,
0.022421,
0.023641,
0.024758,
0.025783,
0.026722,
0.027583,
0.028374,
0.029100])

print('curve points:' , tsp_curve)

dcf = np.zeros(cf.size)
for i in range(0,time_to_maturity):
    dcf[i] = cf[i]/((1 + tsp_curve[i])**(i+1))

#print('dcf', dcf)

z_guess = 0.0029

total_Val = np.sum(dcf)
counter=0

while(1>0):

    value_at_z = 0

    for i in range(0,time_to_maturity):
        value_at_z += cf[i] / ((1 + tsp_curve[i] + z_guess)**(i+1))

    #print('value_at_z', value_at_z)

    derivative = 0
    for j in range(0,time_to_maturity):
        derivative += ((j+1 * cf[j]) / ((1 + tsp_curve[j] + z_guess) ** (j + 2)))
        #print('sum partial', derivative)
        
    #print('derivative', derivative)

    value_at_z = market_price - value_at_z
    #print('after diff value_at_z', value_at_z)

    ratio = value_at_z / derivative
    #print('ratio', ratio)

    next_z = z_guess - ratio/100
    
    if debug:
        print('next Z', next_z)
        print('-'*40)
    if (next_z < 0.0018):
        exit(1)
    if abs(next_z - z_guess) < tolerance:
        print('final z_guess:', z_guess, '%')
        print('final z-spread:', next_z, '%')
        print('final z-spread:', next_z*100, '%')
        print('iterations:',counter)
        elapsed_time = time.time() - start_time
        print('elapsed time:',elapsed_time,' seconds')
        break
    
    z_guess = next_z
    counter += 1
########   
