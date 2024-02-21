import torch
import numpy as np
import plotly.graph_objects as go
import os
import glob
import tqdm

num_each = int(input("How many of each type of light source?"))
plotting = bool(input("Plotting?") == "y")

def plot(x, y):
    fig = go.Figure()
    ymag = -2.5 * np.log10(y / 309.54) 
    fig.add_trace(go.Scatter(x=x, y=ymag, mode='markers', marker=dict(size=3, opacity=0.7, color="blue")))
    fig.update_layout(
        xaxis_title="Time (days)",
        yaxis_title="Brightness", 
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="RebeccaPurple"
        ))
    fig.update_yaxes(autorange="reversed")  # Reverse y-axis and set range to 5.5 to 17
    return fig

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

# getstd = lambda x: np.arcsinh(x * 70) / 120 # Noise vs Brightness - TOO SIMPLE

progressbar = tqdm.tqdm(total=num_each*4)

# TRANSITS

transitperiodrange = [0.2, 12] # Days 
transitdurationrange = [0.04, 0.15] # Percentage of period domain occupied by transit
transitdepthrange = [0.04, 0.6] # Percentage dip in flux



def get_transit_func(bright, period, duration, depth):
    std = getstd(bright)
    duration = period * duration

    depth = max(bright * depth, 2.5 * std) # Clip to 2 sigma

    transitstart = s([0, period - duration])
    transitend = transitstart + duration

    troughlen = s([0.01,0.5]) * duration 
    troughstart = transitstart + (duration - troughlen) / 2
    troughend = transitend - (duration - troughlen) / 2

    def eval(xr):
        x = xr % period
        if x < transitstart or x > transitend:
            val = bright
            return val
        elif x < troughstart:
            val = bright - depth * (x - transitstart) / (troughstart - transitstart)
            return val
        elif x > troughend:
            val = bright - depth * (1 - (x - troughend) / (transitend - troughend))
            return val
        else: # in trough
            val = bright - depth
            return val
        
    eval.start = transitstart
    eval.end = transitend
        
    def evalwnoise(xr):
        return g(eval(xr), std)
    return evalwnoise, eval


def boxfunc(x, depth, start, end, bright, dur):
    t = ((end - start) - dur) / 2
    
    if x < start or x > end:
        val = bright
        return val
    elif x < start + t:
        return bright - depth * (x - start) / t
    elif x > end - t:
        return bright + depth * (x - end) / t
    else:
        return bright - depth
    

ii = 0
while ii < num_each:
    r = np.random.random()
    if r > 0.75: # 25% LP
        period = s([10, 25])
    elif r > 0.5: # 25% MP
        period = s([2, 10])
    else: # 50% SP
        period = s([0.1, 1])


    duration = s(transitdurationrange)
    brightness = baseflux()
    depth = s(transitdepthrange)
    transitfunc, underlying = get_transit_func(brightness, period, duration, depth)
    ust, uend = underlying.start, underlying.end
    transitfunc = np.vectorize(transitfunc)
    underlying = np.vectorize(underlying)
    x = gen_sampling()
    y = transitfunc(x)
    
    xm = (x % period) / period
    # xpm = np.linspace(0, 1, 1000)
    # ypm = underlying(xpm * period)

    if np.quantile(y, 0.08) > brightness - 2*getstd(brightness):
        continue


    # Lower and upper bounds for each parameter
    # bounds = ([0, ust/period, uend/period- 0.2*duration, brightness- 0.000001, -0.1], 
    #           [depth*brightness, ust/period + 0.2*duration, uend/period, brightness+0.000001, 0.1])
    # popt, pcov = curve_fit(np.vectorize(boxfunc), xm, y, p0=[0.05*depth*brightness, ust/period, uend/period, brightness,0], bounds=bounds)
    # q = popt[0]/(brightness * depth)
    # if q < 0.6 or q > 0.95:
    #     continue

    # model = lambda x: np.vectorize(boxfunc)(x, *popt)
    # tr = go.Scatter(x=xm, y=y, mode="markers", marker=dict(size=3, opacity=0.7, color="blue"))
    # trp = go.Scatter(x=xpm, y=ypm, mode="lines", line=dict(color="red"))
    # trm = go.Scatter(x=xpm, y=model(xpm), mode="lines", line=dict(color="green"))
    # fig = go.Figure()
    # fig.add_trace(tr)
    # fig.add_trace(trp)
    # fig.add_trace(trm)
    # fig.write_image("./temp.png")
    # raise Exception()


    ex = (x, y, period, duration, depth)
    buckets["transit"].append(ex)
    progressbar.update(1)
    ii+=1

# Pulsating Model
ampsrange = [0, 0.5]
periodsrange = [0.1, 500]

def get_period_func(bright, amps, periods, phases):
    amps = bright * np.array(amps)
    freqs = 2 * np.pi / np.array(periods)

    def eval(x):
        val = bright
        for A, F, P in zip(amps, freqs, phases):
            currentval = A * np.sin(F * (x + P))
            try:
                val += currentval
            except:
                print(val, currentval)
                raise Exception()
        return g(val, getstd(val))
    return eval
        
for _ in range(num_each):
    r = np.random.random()
    if r > 0.7:
        periods = [s([0.01, 1]), s([0.01, 1])]
    elif r > 0.2:
        periods = [g(35,30), g(35, 30)]
    else:   
        periods = [s([0.01, 500]), s([0.01, 10])]
    periods = np.clip(periods, periodsrange[0], periodsrange[1])
    brightness = baseflux()
    std = getstd(brightness)
    amps = [g(2*std/brightness, 0.1), 0.2*g(2*std/brightness, 0.1)] # Clip to 3.5 snr
    amps = np.clip(np.abs(amps), 3.5*std / brightness, ampsrange[1])

    phases = [s([0,periods[0]]), s([0,periods[1]])]
    starfunc = np.vectorize(get_period_func(brightness, amps, periods, phases))
    x = gen_sampling()
    y = starfunc(x)
    ex = (x, y, periods, amps)
    buckets["pulsating_var"].append(ex)
    progressbar.update(1)


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

for _ in range(num_each):
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

# Null model

def get_null_func(bright):
    std = getstd(bright)
    def eval(x):
        return g(bright, std)
    return eval

for _ in range(num_each):
    brightness = baseflux()
    transitfunc = np.vectorize(get_null_func(brightness))
    x = gen_sampling()
    y = transitfunc(x)
    ex = (x, y)
    buckets["null"].append(ex)
    progressbar.update(1)

# remove 30% for validation
validbuckets = {
    "null": [],
    "nova": [],
    "pulsating_var": [],
    "transit": []
}

amt_valid = int(0.3*num_each)

for kind in buckets:
    validbuckets[kind] = buckets[kind][:amt_valid]
    buckets[kind] = buckets[kind][amt_valid:]

trainset = PseudoSet(buckets, length_batching=False)
validset = PseudoSet(validbuckets, False, length_batching=False)

with open("../processed_datasets/pseudotrain.pt", "wb") as f:
    torch.save(trainset, f)

with open("../processed_datasets/pseudovalid.pt", "wb") as f:
    torch.save(validset, f)

print("Complete")

# for i, ex in enumerate(buckets["null"]):
#     plt = plot(ex[0], ex[1])
#     plt.write_image(f"./dataset_imgs/pseudo/null/{i}.png")

# for i, ex in enumerate(buckets["nova"]):
#     plt = plot(ex[0], ex[1])
#     plt.write_image(f"./dataset_imgs/pseudo/nova/{i}.png")

# for i, ex in enumerate(buckets["pulsating_var"]):
#     plt = plot(ex[0], ex[1])
#     plt.write_image(f"./dataset_imgs/pseudo/pulsating_var/{i}.png")

#     plt = plot(ex[0] % ex[2][0], ex[1])
#     plt.write_image(f"./dataset_imgs/pseudo/pulsating_var/{i}_fold.png")

# for i, ex in enumerate(buckets["transit"]):
#     plt = plot(ex[0], ex[1])
#     plt.write_image(f"./dataset_imgs/pseudo/transit/{i}.png")
#     plt = plot(ex[0] % ex[2], ex[1])
#     plt.write_image(f"./dataset_imgs/pseudo/transit/{i}_fold.png")
# print("Images saved")

if plotting:
    path = glob.glob("./dataset_imgs/pseudo/null/*")
    for p in path:
        os.remove(p)
    for i, ex in enumerate(buckets["null"]):
        plt = plot(ex[0], ex[1])
        plt.write_image(f"./dataset_imgs/pseudo/null/{i}.png")

    path = glob.glob("./dataset_imgs/pseudo/nova/*")
    for p in path:
        os.remove(p)
    for i, ex in enumerate(buckets["nova"]):
        plt = plot(ex[0], ex[1])
        plt.write_image(f"./dataset_imgs/pseudo/nova/{i}.png")

    path = glob.glob("./dataset_imgs/pseudo/pulsating_var/*")
    for p in path:
        os.remove(p)
    for i, ex in enumerate(buckets["pulsating_var"]):
        plt = plot(ex[0], ex[1])
        plt.write_image(f"./dataset_imgs/pseudo/pulsating_var/{i}.png")

        plt = plot(ex[0] % ex[2][0], ex[1])
        plt.write_image(f"./dataset_imgs/pseudo/pulsating_var/{i}_fold.png")

    path = glob.glob("./dataset_imgs/pseudo/transit/*")
    for p in path:
        os.remove(p)
    for i, ex in enumerate(buckets["transit"]):
        plt = plot(ex[0], ex[1])
        plt.write_image(f"./dataset_imgs/pseudo/transit/{i}.png")
        plt = plot(ex[0] % ex[2], ex[1])
        plt.write_image(f"./dataset_imgs/pseudo/transit/{i}_fold.png")
    print("Images saved")