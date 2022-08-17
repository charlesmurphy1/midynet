from __future__ import annotations
import importlib
import basegraph
from typing import Callable, Optional
from _midynet.mcmc import _GraphReconstructionMCMC
from _midynet.mcmc.callbacks import CallBack
from _midynet.utility import get_edge_list


__all__ = ("MCMCConvergenceAnalysis",)


class MCMCConvergenceAnalysis:
    def __init__(
        self,
        mcmc: GraphReconstructionMCMC,
        distance: Optional[Callable] = None,
        callbacks: Optional[list[CallBack]] = None,
    ):
        self.mcmc = mcmc
        self.distance = distance
        self.collected = []
        self.callbacks = [] if callbacks is None else callbacks
        for c in self.callbacks:
            self.mcmc.add_callback(c)

    def burn(self, numsteps=1000):
        return self.mcmc.do_MH_sweep(numsteps)

    def collect(
        self,
        num_samples=100,
        numsteps_between_samples=10,
        numsteps_between_resets=None,
    ):

        original_graph = self.convert_basegraph_to_networkx(self.mcmc.get_graph())

        for i in range(num_samples):
            s, f = self.mcmc.do_MH_sweep(numsteps_between_samples)
            current_graph = self.convert_basegraph_to_networkx(self.mcmc.get_graph())
            if self.distance is not None:
                self.collected.append(self.distance(original_graph, current_graph))
        return self.collected

    def clear(self):
        self.collected.clear()

    @staticmethod
    def convert_basegraph_to_networkx(
        bs_graph: basegraph.core.UndirectedMultigraph,
    ):
        if importlib.util.find_spec("networkx") is None:
            message = (
                "The MCMCConvergenceAnalysis method cannot be used, "
                + "because `networkx` is not installed."
            )
            raise NotImplementedError(message)
        else:
            import networkx as nx
        nx_graph = nx.MultiGraph()
        nx_graph.add_nodes_from(range(bs_graph.get_size()))
        nx_graph.add_edges_from(get_edge_list(bs_graph))
        return nx_graph
