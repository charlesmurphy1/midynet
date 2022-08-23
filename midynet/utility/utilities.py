import itertools
import pathlib
import typing

import basegraph
import numpy as np

__all__ = (
    "clip",
    "log_sum_exp",
    "log_mean_exp",
    "to_batch",
    "delete_path",
    "enumerate_all_graphs",
    "enumerate_all_partitions",
)


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
    for i in range(0, len(x), size):
        yield x[i : i + size]


def logbase(x, base=np.e):
    return np.log(x) / np.log(base)


def to_nary(x, base=2, dim=None):
    if type(x) is int or type(x) is float:
        x = np.array([x])
    if dim is None:
        max_val = base ** np.floor(logbase(np.max(x), base) + 1)
        dim = int(logbase(max_val, base))
    y = np.zeros([dim, *x.shape])
    for idx, xx in np.ndenumerate(x):
        r = np.zeros(dim)
        r0 = xx
        while r0 > 0:
            b = int(np.floor(logbase(r0, base)))
            r[b] += 1
            r0 -= base**b
        y.T[idx] = r[::-1]
    return y


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


def get_all_edges(size: int, allow_self_loops: bool = False) -> list[tuple[int, int]]:
    if allow_self_loops:
        return list(itertools.combinations_with_replacement(range(size), 2))
    else:
        return list(itertools.combinations(range(size), 2))


def generate_all_edge_lists(
    size: int,
    edge_count: int,
    allow_self_loops: bool = False,
    allow_multiedges: bool = False,
) -> typing.Generator[list[tuple[int, int]], None, None]:
    all_edges = get_all_edges(size, allow_self_loops)

    if allow_multiedges:
        edge_list_generator = itertools.combinations_with_replacement(
            all_edges, edge_count
        )
    else:
        edge_list_generator = itertools.combinations(all_edges, edge_count)
    for el in edge_list_generator:
        yield list(el)


def enumerate_all_graphs(
    size: int,
    edge_count: int,
    allow_self_loops: bool = False,
    allow_multiedges: bool = False,
) -> typing.Generator[basegraph.core.UndirectedMultigraph, None, None]:
    all_edge_lists = generate_all_edge_lists(
        size,
        edge_count,
        allow_self_loops=allow_self_loops,
        allow_multiedges=allow_multiedges,
    )
    for edge_list in all_edge_lists:
        g = basegraph.core.UndirectedMultigraph(size)
        for u, v in edge_list:
            g.add_edge_idx(u, v)
        yield g


def enumerate_all_partitions(size, block_count=None):
    block_count = size if block_count is None else block_count
    for i in range(block_count**size):
        yield tuple(to_nary(i, block_count, dim=size).squeeze().astype("int").tolist())


if __name__ == "__main__":
    pass
