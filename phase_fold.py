def phase_fold(time, signal, period):
    for i in range(len(time)):
        time[i] = time[i] % period
    return time, signal