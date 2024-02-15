import numpy as np
from scipy.optimize import curve_fit

# Define the form of the function you want to fit
def func(x, a, b, c, d):
    return a * (b*x)**c + d

# Provide your data as lists or arrays
x_data = np.array([0.19512,0.004901,0.00077,0.0003093,0.0001231])
y_data = np.array([0.0014,0.0000895,0.000035,0.00002721,0.00002071])

# Use curve_fit to find the best fit parameters
popt, pcov = curve_fit(func, x_data, y_data, maxfev=100000)

print("Best fit parameters: {}({}x)^{} + {}".format(*popt))