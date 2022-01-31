from __future__ import annotations
import numpy as np
import typing
import importlib

from basegraph.core import UndirectedMultigraph
from _midynet.mcmc import DynamicsMCMC
from _midynet.utility import get_edge_list


class MCMCConvergenceAnalysis:
    def __init__(
        self,
        mcmc: DynamicsMCMC,
        distance: BaseDistance,
    ):
        if importlib.util.find_spec("netrd") is None:
            message = (
                f"The MCMCConvergenceAnalysis method cannot be used, "
                + "because `netrd` is not installed."
            )
            raise NotImplementedError(message)
        else:
            from netrd.distance import BaseDistance
        self.mcmc = mcmc
        self.distance = distance
        self.collected = []

    def collect(
        self,
        burn=1000,
        num_samples=100,
        numsteps_between_samples=10,
        numsteps_between_resets=None,
    ):

        original_graph = self.convert_basegraph_to_networkx(self.mcmc.get_graph())

        s, f = self.mcmc.do_MH_sweep(burn)
        numsteps = 0
        for i in range(num_samples):
            s, f = self.mcmc.do_MH_sweep(numsteps_between_samples)
            current_graph = self.convert_basegraph_to_networkx(self.mcmc.get_graph())
            self.collected.append(self.distance.dist(original_graph, current_graph))
        return self.collected

    def clear(self):
        self.collected.clear()

    @staticmethod
    def convert_basegraph_to_networkx(
        bs_graph: basegraph.core.UndirectedMultigraph,
    ) -> nx.Graph:
        if importlib.util.find_spec("networkx") is None:
            message = (
                f"The MCMCConvergenceAnalysis method cannot be used, "
                + "because `networkx` is not installed."
            )
            raise NotImplementedError(message)
        else:
            import networkx as nx
        nx_graph = nx.MultiGraph()
        nx_graph.add_nodes_from(range(bs_graph.get_size()))
        nx_graph.add_edges_from(get_edge_list(bs_graph))
        return nx_graph
