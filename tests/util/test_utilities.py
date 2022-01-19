import unittest
import math

from midynet.util import enumerate_all_graphs
from _midynet.utility import log_binom, log_multiset


class TestEnumerateAllGraphs(unittest.TestCase):
    def setUp(self):
        pass

    def test_enumerate_all_simple_graphs_of_3_vertices_and_3_edges(self):
        counter = 0
        for g in enumerate_all_graphs(3, 3):
            counter += 1
            self.assertEqual(g.get_total_edge_number(), 3)

        self.assertEqual(counter, 1)

    def test_enumerate_all_simple_graphs_of_3_vertices_and_2_edges(self):
        counter = 0
        for g in enumerate_all_graphs(3, 2):
            counter += 1
            self.assertEqual(g.get_total_edge_number(), 2)

        self.assertEqual(counter, 3)

    def test_enumerate_all_simple_graphs_of_3_vertices_and_1_edges(self):
        counter = 0
        for g in enumerate_all_graphs(3, 1):
            counter += 1
            self.assertEqual(g.get_total_edge_number(), 1)

        self.assertEqual(counter, 3)

    def test_enumerate_all_loopy_multigraphs_of_3_vertices_and_3_edges(self):
        counter = 0
        for g in enumerate_all_graphs(
            3, 3, allow_self_loops=True, allow_multiedges=True
        ):
            counter += 1
            self.assertEqual(g.get_total_edge_number(), 3)

        self.assertAlmostEqual(counter, math.exp(log_multiset(int(3 * (3 + 1) / 2), 3)))

    def test_enumerate_all_multigraphs_of_3_vertices_and_3_edges(self):
        counter = 0
        for g in enumerate_all_graphs(
            3, 3, allow_self_loops=False, allow_multiedges=True
        ):
            counter += 1
            self.assertEqual(g.get_total_edge_number(), 3)

        self.assertAlmostEqual(counter, math.exp(log_multiset(int(3 * (3 - 1) / 2), 3)))

    def test_enumerate_all_loopy_graphs_of_3_vertices_and_3_edges(self):
        counter = 0
        for g in enumerate_all_graphs(
            3, 3, allow_self_loops=True, allow_multiedges=False
        ):
            counter += 1
            self.assertEqual(g.get_total_edge_number(), 3)

        self.assertAlmostEqual(counter, math.exp(log_binom(int(3 * (3 + 1) / 2), 3)))


if __name__ == "__main__":
    unittest.main()
