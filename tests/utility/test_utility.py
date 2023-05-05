import os
import math
import pytest
import numpy as np
import midynet
import graph_tool.all as gt
from scipy.special import loggamma
from midynet.utility import enumerate_all_graphs


def log_binom(n, k):
    return loggamma(n + 1) - loggamma(n - k + 1) - loggamma(k + 1)


def log_multiset(n, k):
    return log_binom(n + k - 1, k)


def test_enumerate_all_simple_graphs_of_3_vertices_and_3_edges():
    counter = 0
    for g in enumerate_all_graphs(3, 3):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == 1


def test_enumerate_all_simple_graphs_of_3_vertices_and_2_edges():
    counter = 0
    for g in enumerate_all_graphs(3, 2):
        counter += 1
        assert g.get_total_edge_number() == 2

    assert counter == 3


def test_enumerate_all_simple_graphs_of_3_vertices_and_1_edges():
    counter = 0
    for g in enumerate_all_graphs(3, 1):
        counter += 1
        assert g.get_total_edge_number() == 1

    assert counter == 3


def test_enumerate_all_loopy_multigraphs_of_3_vertices_and_3_edges():
    counter = 0
    for g in enumerate_all_graphs(3, 3, allow_self_loops=True, allow_multiedges=True):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == pytest.approx(math.exp(log_multiset(int(3 * (3 + 1) / 2), 3)))


def test_enumerate_all_multigraphs_of_3_vertices_and_3_edges():
    counter = 0
    for g in enumerate_all_graphs(3, 3, allow_self_loops=False, allow_multiedges=True):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == pytest.approx(math.exp(log_multiset(int(3 * (3 - 1) / 2), 3)))


def test_enumerate_all_loopy_graphs_of_3_vertices_and_3_edges():
    counter = 0
    for g in enumerate_all_graphs(3, 3, allow_self_loops=True, allow_multiedges=False):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == pytest.approx(math.exp(log_binom(int(3 * (3 + 1) / 2), 3)))


def test_save_graph():
    g = gt.collection.ns["foodweb_little_rock"]
    g = midynet.utility.convert.convert_graphtool_to_basegraph(g)
    midynet.utility.convert.save_graph(g, "test-little-rock.npy")
    assert os.path.exists("test-little-rock.npy")
    edges = np.load("test-little-rock.npy")
    assert isinstance(edges, np.ndarray)
    assert edges.shape == (g.get_edge_number(), 3)
    assert edges[:, -1].sum() == g.get_total_edge_number()
    os.remove("test-little-rock.npy")


def test_load_graph():
    g = gt.collection.ns["foodweb_little_rock"]
    g = midynet.utility.convert.convert_graphtool_to_basegraph(g)
    midynet.utility.convert.save_graph(g, "test-little-rock.npy")
    assert os.path.exists("test-little-rock.npy")
    loaded = midynet.utility.convert.load_graph("test-little-rock.npy")
    assert g == loaded
    os.remove("test-little-rock.npy")


if __name__ == "__main__":
    pass
