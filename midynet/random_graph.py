from _midynet import random_graph as _random_graph
from _midynet.random_graph import (
    RandomGraph,
    BlockLabeledRandomGraph,
    NestedBlockLabeledRandomGraph,
    ErdosRenyiModel,
    ConfigurationModel,
    ConfigurationModelFamily,
    StochasticBlockModel,
    PlantedPartitionModel,
)
from midynet.wrapper import Wrapper as _Wrapper

__all__ = (
    "RandomGraph",
    "RandomGraphWrapper",
    "BlockLabeledRandomGraph",
    "NestedBlockLabeledRandomGraph",
    "ErdosRenyiModel",
    "ConfigurationModel",
    "ConfigurationModelFamily",
    "StochasticBlockModel",
    "StochasticBlockModelFamily",
    "PlantedPartitionModel",
)


class RandomGraphWrapper(_Wrapper):
    def __init__(self, graph_model, **kwargs):
        super().__init__(graph_model, params=kwargs)


class ErdosRenyiModel(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        likelihood_type: str = "uniform",
        canonical: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
    ):
        wrapped = _random_graph.ErdosRenyiModel(
            size,
            edge_count,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
        )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            likelihood_type=likelihood_type,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
        )


class ConfigurationModelFamily(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        prior_type: str = "uniform",
        canonical: bool = False,
        edge_proposer_type: str = "uniform",
    ):
        wrapped = _random_graph.ConfigurationModelFamily(
            size,
            edge_count,
            canonical=canonical,
            hyperprior=(prior_type == "hyperprior"),
            edge_proposer_type=edge_proposer_type,
        )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            prior_type=prior_type,
            canonical=canonical,
            edge_proposer_type=edge_proposer_type,
        )


class PlantedPartitionModel(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: int = 250,
        block_count: int = 3,
        assortativity: float = 0.5,
        stub_labeled: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
    ):
        wrapped = _random_graph.PlantedPartitionModel(
            size,
            edge_count,
            block_count=block_count,
            assortativity=assortativity,
            stub_labeled=stub_labeled,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
        )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
            stub_labeled=stub_labeled,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
        )


class StochasticBlockModelFamily(RandomGraphWrapper):
    def __init__(
        self,
        size: int = 100,
        edge_count: float = 250,
        block_count: int = 0,
        likelihood_type: str = "uniform",
        block_prior_type: str = "uniform",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        with_self_loops: bool = True,
        with_parallel_edges: bool = True,
        edge_proposer_type: str = "uniform",
        block_proposer_type: str = "uniform",
        sample_label_count_prob: float = 0.1,
        label_creation_prob: float = 0.5,
        shift: float = 1,
    ):

        if likelihood_type == "degree_corrected":
            if label_graph_prior_type == "nested":
                wrapped = _random_graph.NestedDegreeCorrectedStochasticBlockModelFamily(
                    size,
                    edge_count,
                    degree_hyperprior=(degree_prior_type == "hyper"),
                    canonical=canonical,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
            else:
                wrapped = _random_graph.DegreeCorrectedStochasticBlockModelFamily(
                    size,
                    edge_count,
                    block_count=block_count,
                    block_hyperprior=(block_prior_type == "hyper"),
                    degree_hyperprior=(degree_prior_type == "hyper"),
                    planted=(label_graph_prior_type == "planted"),
                    canonical=canonical,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
        else:
            if label_graph_prior_type == "nested":
                wrapped = _random_graph.NestedStochasticBlockModelFamily(
                    size,
                    edge_count,
                    stub_labeled=(likelihood_type == "stub_labeled"),
                    canonical=canonical,
                    with_self_loops=with_self_loops,
                    with_parallel_edges=with_parallel_edges,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
            else:
                wrapped = _random_graph.StochasticBlockModelFamily(
                    size,
                    edge_count,
                    block_count=block_count,
                    block_hyperprior=(block_prior_type == "hyper"),
                    planted=(label_graph_prior_type == "planted"),
                    stub_labeled=(likelihood_type == "stub_labeled"),
                    canonical=canonical,
                    with_self_loops=with_self_loops,
                    with_parallel_edges=with_parallel_edges,
                    edge_proposer_type=edge_proposer_type,
                    block_proposer_type=block_proposer_type,
                    sample_label_count_prob=sample_label_count_prob,
                    label_creation_prob=label_creation_prob,
                    shift=shift,
                )
        super().__init__(
            wrapped,
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            likelihood_type=likelihood_type,
            block_prior_type=block_prior_type,
            label_graph_prior_type=label_graph_prior_type,
            degree_prior_type=degree_prior_type,
            canonical=canonical,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
            edge_proposer_type=edge_proposer_type,
            block_proposer_type=block_proposer_type,
            sample_label_count_prob=sample_label_count_prob,
            label_creation_prob=label_creation_prob,
            shift=shift,
        )
