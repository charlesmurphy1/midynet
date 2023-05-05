import os

from basegraph import core
from graphinf.graph import (
    ErdosRenyiModel,
    PoissonModel,
    NegativeBinomialModel,
    ConfigurationModelFamily,
    StochasticBlockModelFamily,
    PlantedPartitionModel,
)

from midynet.config import Config, static
from .factory import Factory

__all__ = ("GraphConfig", "GraphFactory")


@static
class GraphConfig(Config):
    @classmethod
    def karate(cls):
        return cls(
            "karate",
            size=34,
            edge_count=78,
            gt_id="karate/78",
            with_self_loops=False,
            with_parallel_edges=False,
        )

    @classmethod
    def littlerock(cls):
        return cls(
            "littlerock",
            size=183,
            edge_count=2_494,
            gt_id="foodweb_little_rock",
            with_self_loops=True,
            with_parallel_edges=True,
        )

    @classmethod
    def openflights(cls):
        return cls(
            "openflights",
            size=3_214,
            edge_count=66_771,
            gt_id="openflights",
            with_self_loops=False,
            with_parallel_edges=False,
        )

    @classmethod
    def euairlines(cls):
        return cls(
            "euairlines",
            size=450,
            edge_count=3_588,
            gt_id="eu_airlines",
            with_self_loops=False,
            with_parallel_edges=True,
        )

    @classmethod
    def celegans(cls):
        return cls(
            "celegans",
            size=514,
            edge_count=2_363,
            gt_id="celegans_2019/male_gap_junction_synapse",
            with_self_loops=True,
            with_parallel_edges=True,
        )

    @classmethod
    def erdosrenyi(
        cls,
        size: int = 100,
        edge_count: float = 250,
        likelihood_type: str = "uniform",
        canonical: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
    ):
        return cls(
            "erdosrenyi",
            size=size,
            likelihood_type=likelihood_type,
            edge_count=edge_count,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
        )

    @classmethod
    def configuration(
        cls,
        size: int = 100,
        edge_count: float = 250,
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        edge_proposer_type: str = "uniform",
    ):
        return cls(
            "configuration",
            size=size,
            edge_count=edge_count,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            edge_proposer_type=edge_proposer_type,
            with_self_loops=True,
            with_parallel_edges=True,
        )

    @classmethod
    def poisson(cls, size: int = 100, edge_count: int = 250):
        return cls(
            "poisson",
            size=size,
            edge_count=edge_count,
        )

    @classmethod
    def nbinom(
        cls,
        size: int = 100,
        edge_count: int = 250,
        heterogeneity: float = 0,
    ):
        return cls(
            "nbinom",
            size=size,
            edge_count=edge_count,
            heterogeneity=heterogeneity,
            with_self_loops=True,
            with_parallel_edges=True,
        )

    @classmethod
    def stochastic_block_model(
        cls,
        size: int = 100,
        edge_count: float = 250,
        block_count: int = 0,
        likelihood_type: str = "uniform",
        block_prior_type: str = "hyper",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        exact: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
        block_proposer_type: str = "uniform",
        sample_label_count_prob: float = 0.1,
        label_creation_prob: float = 0.5,
        shift: float = 1,
    ):
        return cls(
            "stochastic_block_model",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            likelihood_type=likelihood_type,
            block_prior_type=block_prior_type,
            label_graph_prior_type=label_graph_prior_type,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            exact=exact,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
            block_proposer_type=block_proposer_type,
            sample_label_count_prob=sample_label_count_prob,
            label_creation_prob=label_creation_prob,
            shift=shift,
        )

    @classmethod
    def degree_corrected_stochastic_block_model(cls, **kwargs):
        kwargs["likelihood_type"] = "degree_corrected"
        return cls.stochastic_block_model(**kwargs)

    @classmethod
    def planted_partition(
        cls,
        size: int = 100,
        edge_count: int = 250,
        block_count: int = 3,
        assortativity: float = 0.5,
        stub_labeled: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
    ):
        return cls(
            "planted_partition",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
            stub_labeled=stub_labeled,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
        )


class GraphFactory(Factory):
    @staticmethod
    def load_gtgraph(name: str) -> core.UndirectedMultigraph:
        from graph_tool import collection
        from midynet.utility.convert import convert_graphtool_to_basegraph

        gt_graph = collection.ns[name]
        return convert_graphtool_to_basegraph(gt_graph)

    @staticmethod
    def load_graph(config: GraphConfig) -> core.UndirectedMultigraph:
        try:
            # print("Fetching graph from Network Repo...")
            raise KeyError()
            return GraphFactory.load_gtgraph(config.gt_id)
        except KeyError:
            from midynet.utility.convert import load_graph

            # print("Loading graph locally...")
            path_to_graph = os.path.join(
                __file__.removesuffix("random_graph.py"), config.name + ".npy"
            )
            return load_graph(path_to_graph)

    @staticmethod
    def build_karate(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_littlerock(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_euairlines(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_openflights(config: GraphConfig) -> core.UndirectedMultigraph:
        g = GraphFactory.load_graph(config)
        g.remove_selfloops()
        return g

    @staticmethod
    def build_celegans(config: GraphConfig) -> core.UndirectedMultigraph:
        return GraphFactory.load_graph(config)

    @staticmethod
    def build_erdosrenyi(config: GraphConfig) -> ErdosRenyiModel:
        return ErdosRenyiModel(
            config.size,
            config.edge_count,
            canonical=config.canonical,
            with_self_loops=config.with_self_loops,
            with_parallel_edges=config.with_parallel_edges,
            edge_proposer_type=config.edge_proposer_type,
        )

    @staticmethod
    def build_configuration(config: GraphConfig) -> ConfigurationModelFamily:
        return ConfigurationModelFamily(
            config.size,
            config.edge_count,
            degree_prior_type=config.degree_prior_type,
            canonical=config.canonical,
            edge_proposer_type=config.edge_proposer_type,
        )

    @staticmethod
    def build_poisson(config: GraphConfig) -> PoissonModel:
        return PoissonModel(config.size, config.edge_count)

    @staticmethod
    def build_nbinom(config: GraphConfig) -> NegativeBinomialModel:
        return NegativeBinomialModel(
            config.size, config.edge_count, config.heterogeneity
        )

    @staticmethod
    def build_stochastic_block_model(
        config: GraphConfig,
    ) -> StochasticBlockModelFamily:
        return StochasticBlockModelFamily(
            size=config.size,
            edge_count=config.edge_count,
            block_count=config.block_count,
            likelihood_type=config.likelihood_type,
            block_prior_type=config.block_prior_type,
            label_graph_prior_type=config.label_graph_prior_type,
            degree_prior_type=config.degree_prior_type,
            canonical=config.canonical,
            with_self_loops=config.with_self_loops,
            with_parallel_edges=config.with_parallel_edges,
            edge_proposer_type=config.edge_proposer_type,
            block_proposer_type=config.block_proposer_type,
            sample_label_count_prob=config.sample_label_count_prob,
            label_creation_prob=config.label_creation_prob,
            shift=config.shift,
        )

    @staticmethod
    def build_planted_partition(config: GraphConfig):
        return PlantedPartitionModel(
            size=config.size,
            edge_count=config.edge_count,
            block_count=config.block_count,
            assortativity=config.assortativity,
            stub_labeled=config.stub_labeled,
            with_self_loops=config.with_self_loops,
            with_parallel_edges=config.with_parallel_edges,
        )


if __name__ == "__main__":
    pass
