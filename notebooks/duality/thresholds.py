import midynet
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import json
import numpy as np

from midynet.utility import display
from functools import partial
from graphinf.utility import seed as gi_seed


def SISSusceptibility(x):
    n = x.sum(0)
    return (np.mean(n**2) - np.mean(n) ** 2) / np.mean(n)


def SISAverage(x):
    return x.mean()


def GlauberSusceptibility(x):
    x[x == 0] = -1
    m = np.abs(x.mean(0))
    return (np.mean(m**2) - np.mean(m) ** 2) / (np.mean(m))


def GlauberAverage(x):
    y = x * 1
    y[x == 0] = -1
    X = np.mean(y, 0)
    return np.mean(np.abs(X))


def CowanSusceptibility(x):
    x[x == 0] = -1
    m = x.mean(-1)
    return np.mean(m**2) - np.mean(np.abs(m)) ** 2


#     return x.mean()


susceptiblityFunctions = {
    "glauber": GlauberSusceptibility,
    "cowan": CowanSusceptibility,
    "sis": SISSusceptibility,
}

averageFunctions = {
    "glauber": GlauberAverage,
    "sis": SISAverage,
    "cowan": SISAverage,
}


def collect(cfg, seed=None):
    if seed is not None:
        gi_seed(seed)
    suscFunc = susceptiblityFunctions[cfg.data_model.name]
    avgFunc = averageFunctions[cfg.data_model.name]
    graph = midynet.config.GraphFactory.build(cfg.prior)
    dynamics = midynet.config.DataModelFactory.build(cfg.data_model)
    dynamics.set_graph_prior(graph)
    dynamics.sample()
    x = np.array(dynamics.get_past_states())
    return suscFunc(x), avgFunc(x)


def searchThresholds(
    cfg,
    paramName,
    paramMin,
    paramMax,
    numPoints=10,
    delta=1,
    tol=1e-3,
    numSamples=10,
    verbose=1,
):
    diff = np.inf

    history = {"thresholds": [], "susceptibility": {}, "averages": {}}
    while diff > tol:
        susceptibility = []
        averages = []
        diff = (paramMax - paramMin) / numPoints
        paramScan = np.linspace(paramMin, paramMax, numPoints)
        if verbose > 0:
            print(f"Scaning params {paramScan}")
            print(f"Current diff: {diff}")
        for p in paramScan:
            cfg.data_model[paramName] = p
            if verbose == 1:
                print(p)
            with mp.Pool(4) as pool:
                f = partial(collect, cfg)
                seeds = int(time.time()) + np.arange(numSamples).astype("int")
                out = pool.map(f, seeds)
                s = [ss for ss, aa in out]
                a = [aa for ss, aa in out]
            susceptibility.append(np.mean(s))
            averages.append(np.mean(a))

        for s, a, p in zip(susceptibility, averages, paramScan):
            if p in history["susceptibility"]:
                history["susceptibility"][p].append(s)
            else:
                history["susceptibility"][p] = [s]
            if p in history["averages"]:
                history["averages"][p].append(a)
            else:
                history["averages"][p] = [a]

        #         plt.plot(paramScan, susceptibility)
        #         plt.show()

        maxIndex = np.argmax(susceptibility)
        paramMin = paramScan[maxIndex] - delta * diff
        paramMax = paramScan[maxIndex] + delta * diff
        history["thresholds"].append(paramScan[maxIndex])
        if verbose > 0:
            print(f"Current history: {history}")
    return history


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1 :] / n)[::n]


def collectCowan(cfg, seed=None):
    if seed is not None:
        gi_seed(seed)
    graph = midynet.config.GraphFactory.build(cfg.prior)
    dynamics = midynet.config.DataModelFactory.build(cfg.data_model)
    dynamics.set_graph_prior(graph)

    dynamics.sample()
    x = np.array(dynamics.get_past_states())
    return np.mean(x)


def searchCowanThresholds(
    cfg,
    paramMin,
    paramMax,
    *,
    delta=2,
    tol=0.01,
    numSamples=10,
    numPoints=10,
    numProcs=4,
    verbose=0,
    parallel=True,
):
    diff = np.inf
    history = {"thresholds": [], "susceptibility": {}, "averages": {}}
    while diff > tol:
        avgx = []
        susceptibility = []
        paramScan = np.linspace(paramMin, paramMax, numPoints)
        diff = (paramMax - paramMin) / numPoints

        if verbose > 0:
            print(f"Scanning params: {paramScan}")
            print(f"Current diff: {diff}")
        for p in paramScan:
            cfg.data_model.nu = p
            f = partial(collectCowan, cfg)
            seeds = int(time.time()) + np.arange(numSamples).astype("int")
            print(p)
            if parallel:
                with mp.Pool(numProcs) as pool:
                    x = pool.map(f, seeds)
            else:
                x = [f(s) for s in seeds]
            x = np.array(x)
            if np.any(x < 0.05):
                x = x[x < 0.05]
            s = np.std(x) / np.mean(x)
            if verbose == 1:
                print(p, s, np.mean(x))
            susceptibility.append(s)
            avgx.append(np.mean(x))
        avgx = np.array(avgx)

        xm = avgx[:-1]
        xp = avgx[1:]

        gap = np.abs(avgx[1:] - avgx[:-1]) / diff
        gap = np.append(gap, 0)
        maxGapIndex = np.argmax(gap)
        #         maxGapIndex = np.argmax(susceptibility)
        paramMin = paramScan[maxGapIndex] - delta * diff
        paramMax = paramScan[maxGapIndex] + delta * diff
        if verbose == 1:
            ax = plt.gca()
            ax.plot(paramScan, avgx, "ro-")
            axx = ax.twinx()
            axx.plot(paramScan, susceptibility, "bs-")
            ax.axvspan(paramMin, paramMax, color="grey", alpha=0.3)
            plt.show()
        for p, x, g in zip(paramScan, avgx, susceptibility):
            if p in history["susceptibility"]:
                history["susceptibility"][p].append(g)
            else:
                history["susceptibility"][p] = [g]

            if p in history["averages"]:
                history["averages"][p].append(x)
            else:
                history["averages"][p] = [x]
        history["thresholds"].append(paramScan[maxGapIndex])
        if verbose > 0:
            print(f"Current history: {history}")
    return history


def plotThresholdSearch(ax, history, paramName=f"Coupling"):
    x = []
    y = []
    z = []

    for k in history["susceptibility"].keys():
        x.append(float(k))
        y.append(np.mean(history["susceptibility"][k]))
        z.append(np.mean(history["averages"][k]))
    indices = np.argsort(x)
    x = np.array(x)[indices]
    y = np.array(y)[indices]
    z = np.array(z)[indices]
    axx = ax.twinx()

    ax.plot(x, y, "o-", color=display.med_colors["blue"])
    axx.plot(x, z, "s-", color=display.med_colors["red"])
    if "thresholds" in history:
        t = history["thresholds"]
        for tt in t[:-1]:
            ax.axvline(tt, linestyle="--", color="grey")
        ax.axvline(t[-1], linestyle="--", color="k", linewidth=2)
    ax.set_xlabel(paramName)
    ax.set_ylabel(r"Susceptibility")
    ax.set_ylim([y.min(), y.max() * 1.1])
    #     ax.set_xlim([x.min(), x.max()])
    axx.set_ylabel(r"Average state")
    axx.set_ylim([z.min(), z.max() * 1.1])
    return ax


def showThresholdSearch(history, paramName=f"Coupling"):
    plotThresholdSearch(plt.gca(), history, paramName=paramName)


if __name__ == "__main__":
    pass
