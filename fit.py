import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit

# # Define the form of the function you want to fit
# def func(x, a, b, c):
#     return a * np.exp(b*x) + c

# # Provide your data as lists or arrays
# # x_data = np.array([0.19512,0.004901,0.00077,0.0003093,0.0001231])
# # y_data = np.array([0.0014,0.0000895,0.000035,0.00002721,0.00002071])

# x_data = np.array([16, 15, 14, 12, 10, 8])
# y_data = np.array([0.2, 0.1, 0.05, 0.0275, 0.02, 0.0215])

# # Use curve_fit to find the best fit parameters
# popt, pcov = curve_fit(func, x_data, y_data, maxfev=100000)

# print("Best fit parameters: {}, {}, {}".format(*[str(round(p, 10)) for p in popt]))

def fluxunc(flux):
    uncertainty = 0.713 * (0.002*flux)**0.8+0.000018 # Canonically accurate noise vs flux # UNCERTAINTY / FLUX / MAGNITUDE / SNR keywords
    return uncertainty
def magunc(flux):
    mag = -2.5 * np.log10(flux / 309.54)
    uncertainty = 2.383*10**(-7)*np.exp(0.8461*mag) + 0.02
    toflux = lambda m: 309.54 * 10**(-m / 2.5)
    return toflux(mag - uncertainty) - toflux(mag + uncertainty)

x = np.linspace(0.0001,4,4000)
y = fluxunc(x)
plt.plot(x, y)
plt.show()
