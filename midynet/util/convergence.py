from __future__ import annotations
import typing
import importlib

from basegraph.core import UndirectedMultigraph
from _midynet.mcmc import DynamicsMCMC


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

    def collect(self, num_samples=100, numsteps_between_samples=10):
        original_graph = convert_basegraph_to_networkx(self.mcmc.get_graph())
        for i in range(num_samples):
            self.mcmc.do_MH_sweep(numsteps_between_samples)
            current_graph = convert_basegraph_to_networkx(self.mcmc.get_graph())
            self.collected.append(self.distance.dist(original_graph, current_graph))
        return self.collected

    def clear(self):
        self.collected.clear()

    @staticmethod
    def convert_basegraph_to_networkx(
        g: basegraph.core.UndirectedMultigraph,
    ) -> nx.Graph:
        if importlib.util.find_spec("networkx") is None:
            message = (
                f"The MCMCConvergenceAnalysis method cannot be used, "
                + "because `networkx` is not installed."
            )
            raise NotImplementedError(message)
        else:
            import networkx as nx
        A = np.array(g.get_adjacency_matrix())
        return nx.from_numpy_array(A)
