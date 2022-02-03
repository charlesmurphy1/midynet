import math
import pytest
from _midynet.utility import log_binom, log_multiset
from midynet.util import enumerate_all_graphs


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
    for g in enumerate_all_graphs(
        3, 3, allow_self_loops=True, allow_multiedges=True
    ):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == pytest.approx(
        math.exp(log_multiset(int(3 * (3 + 1) / 2), 3))
    )


def test_enumerate_all_multigraphs_of_3_vertices_and_3_edges():
    counter = 0
    for g in enumerate_all_graphs(
        3, 3, allow_self_loops=False, allow_multiedges=True
    ):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == pytest.approx(
        math.exp(log_multiset(int(3 * (3 - 1) / 2), 3))
    )


def test_enumerate_all_loopy_graphs_of_3_vertices_and_3_edges():
    counter = 0
    for g in enumerate_all_graphs(
        3, 3, allow_self_loops=True, allow_multiedges=False
    ):
        counter += 1
        assert g.get_total_edge_number() == 3

    assert counter == pytest.approx(
        math.exp(log_binom(int(3 * (3 + 1) / 2), 3))
    )


if __name__ == "__main__":
    pass
