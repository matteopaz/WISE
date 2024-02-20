import numpy as np

def s(bounds): # sample
    return np.random.uniform(bounds[0], bounds[1])

def g(x, std): # gaussian
    return np.random.normal(x, std)

def baseflux(): # Distribution stuff
    r = s([-6.5,2])
    fluxbound = [0.0002,0.2]
    f = lambda x: (fluxbound[1] - fluxbound[0]) / (1 + np.exp(-3*x)) + fluxbound[0]
    return f(r)

def getstd(flux):
    uncertainty = 0.713 * (0.002*flux)**0.8+0.000018 # Canonically accurate noise vs flux # UNCERTAINTY / FLUX / MAGNITUDE / SNR keywords
    return uncertainty / 2 # Sets uncertainty value to 2 sigma error ~ 95% confidence

shortspacing = 0.115 # days
longspacing = 175 # days between obs groups

def apparitionsbeforegap():
    options = [10, 10, 10, 10, 10, 10, 14, 14, 14, 14, 20, 20, 20, 30, 30, 50, 100, 200, 100000]
    return np.random.choice(np.array(options))
    
days = 4000

buckets = {
    "null": [],
    "nova": [],
    "pulsating_var": [],
    "transit": []
}

def gen_sampling():
    x = [0]
    i = 0
    app = apparitionsbeforegap()
    while x[-1] < days:
        if i % app == 0:
            x.append(x[-1] + longspacing)
        else:
            x.append(x[-1] + shortspacing)
        i += 1
    return np.array(x)

# Nova Model
    
decayrange = [0.01, 0.1]

def get_nova_func(bright, peak, decay):
    pop = int(s([0, days / 180]))*180 # so it pops on a observation period
    small = bool(np.random.random() > 0.2)
    visiblebefore = bool(np.random.random() > 0.5)

    peak = max(1.7 + np.abs(g(0, 0.7)) if small else s([20, 60]), 5*getstd(bright) / bright) # clipped to 3 sigma

    def eval(x):
        if x < pop and (visiblebefore or small):
            return g(bright, getstd(bright))
        elif x >= pop:
            exp = (decay*0.2*(x - pop - 1))
            val = peak * np.exp(-exp) * bright + bright
            std = getstd(val)
            if not visiblebefore and val < 1.5*bright and not small: # If not vis before, dissapear once decay below 3x bg
                return -1.0
            else:
                return g(val, std)
        else:
            return -1.0 # Invisible
    return eval


brightness = baseflux()
decay = s(decayrange)
novafunc = np.vectorize(get_nova_func(brightness, 0, decay))
x = gen_sampling()
y = novafunc(x)
idxer = y > 0 # discard invisible
x = x[idxer]
y = y[idxer]
if len(x) <= 15:
    continue
ex = (x, y)
buckets["nova"].append(ex)
progressbar.update(1)
