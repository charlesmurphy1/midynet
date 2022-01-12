import numpy as np
import pathlib

__all__ = ["clip", "log_sum_exp", "log_mean_exp", "to_batch"]


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


def delete_path(path: pathlib.Path):
    if not path.exists():
        return
    elif not path.is_dir():
        path.unlink()
        return
    for sub in path.iterdir():
        if sub.is_dir():
            delete_path(sub)
        else:
            sub.unlink()
    path.rmdir()


if __name__ == "__main__":
    pass
