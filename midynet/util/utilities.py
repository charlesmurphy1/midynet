import basegraph
import networkx as nx
import numpy as np
import pathlib

__all__ = [
    "clip",
    "log_sum_exp",
    "log_mean_exp",
    "to_batch",
    "delete_path",
    "convert_basegraph_to_networkx",
]


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


def convert_basegraph_to_networkx(g: basegraph.core.UndirectedMultigraph) -> nx.Graph:
    A = np.array(g.get_adjacency_matrix())
    return nx.from_numpy_array(A)


if __name__ == "__main__":
    pass
