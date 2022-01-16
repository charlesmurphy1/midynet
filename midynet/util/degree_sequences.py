import numpy as np

from scipy.optimize import bisect
from scipy.special import loggamma
from scipy.interpolate import interp1d
from scipy.stats import poisson, nbinom

__all__ = [
    "generate_degseq",
    "bnbinomial",
    "poisson_degreeseq",
    "scalefree_degreeseq",
    "nbinom_degreeseq",
]


def generate_degseq(xk, pk, size):
    assert xk.shape == pk.shape
    cdf = np.array([np.sum(pk[:k]) for k in range(1, len(pk) + 1)])
    inv_cdf = interp1d(cdf, xk)
    y = np.linspace(cdf.min(), cdf.max(), size + 2)[1:-1]
    degrees = inv_cdf(y).astype("int")
    if sum(degrees) % 2:
        idx = np.random.randint(size)
        degrees[idx] += 1
    return degrees


def logbeta(a, b):
    return loggamma(a) + loggamma(b) - loggamma(a + b)


def bnbinomial(k, r, a, b):
    logp = (
        loggamma(r + k)
        + logbeta(a + r, b + k)
        - loggamma(k + 1)
        - loggamma(r)
        - logbeta(a, b)
    )
    p = np.exp(logp)
    p /= p.sum()
    return p


def poisson_degreeseq(size, avg_deg):
    xk = np.arange(size)
    pk = poisson.pmf(xk, mu=avg_deg)
    return generate_degseq(xk, pk, size)


def nbinom_degreeseq(size, avgk, heterogeneity):
    var = avgk + heterogeneity * avgk ** 2
    q = avgk / var
    n = avgk ** 2 / (var - avgk)
    xk = np.arange(size)
    pk = nbinom.pmf(xk, n, q)
    return generate_degseq(xk, pk, size)


def scalefree_degreeseq(avg, exponent, size, maxiter=1000, kmax=None):
    alpha = exponent + 1
    beta = alpha
    kmax = size if kmax is None else kmax

    def func_to_solve(r):
        xk = np.arange(1, kmax)
        pk = bnbinomial(xk, r, alpha, r * beta)
        k = generate_degseq(xk, pk, size)
        return avg - k.mean()

    r0 = avg * (alpha - 1) / beta

    factor = 2
    i = 0
    success = False
    while not success and i < maxiter:
        a = r0 / factor
        if func_to_solve(a) < 0:
            factor *= 2
            i += 1
        else:
            success = True

    factor = 2
    i = 0
    success = False
    while not success and i < maxiter:
        b = r0 * factor
        if func_to_solve(b) > 0:
            factor *= 2
            i += 1
        else:
            success = True
    r = bisect(func_to_solve, a, b)
    xk = np.arange(size)
    pk = bnbinomial(xk, r, alpha, r * beta)  # exponent - 1)
    k = generate_degseq(xk, pk, size)
    k[k < 1.0] = 1
    return k


if __name__ == "__main__":
    pass
