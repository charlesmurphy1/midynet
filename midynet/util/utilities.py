import numpy as np


def clip(x, xmin=0, xmax=1):
    if isinstance(x, np.ndarray):
        x[x < xmin] = xmin
        x[x > xmax] = xmax
    elif isinstance(x, (int, float)):
        if x < xmin:
            x = xmin
        elif x > xmax:
            x = xmax
    return x


def log_sum_exp(x):
    x = np.array(x)
    b = np.max(x)
    return b + np.log(np.sum(np.exp(x - b)))


def log_mean_exp(x):
    x = np.array(x)
    b = np.max(x)
    return b + np.log(np.mean(np.exp(x - b)))


def to_batch(x, size):
    return zip(*[x[i::size] for i in range(size)])


if __name__ == "__main__":
    pass
