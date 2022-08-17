import numpy as np
from math import floor, ceil
from typing import Union, Optional
from itertools import combinations_with_replacement

from basegraph.core import UndirectedMultigraph
from _midynet.random_graph import (
    ErdosRenyiModel,
    ConfigurationModel,
    ConfigurationModelFamily,
    PlantedPartitionModel,
    StochasticBlockModelFamily,
    NestedStochasticBlockModelFamily,
    DegreeCorrectedStochasticBlockModelFamily,
    NestedDegreeCorrectedStochasticBlockModelFamily,
)

from midynet.utility.degree_sequences import poisson_degreeseq, nbinom_degreeseq
from .config import Config
from .factory import Factory, UnavailableOption

__all__ = ("RandomGraphConfig", "RandomGraphFactory")


class RandomGraphConfig(Config):
    requirements: set[str] = {"labeled"}

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
            name="erdosrenyi",
            size=size,
            edge_count=edge_count,
            labeled=False,
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
        prior_type: str = "uniform",
        canonical: bool = False,
        edge_proposer_type: str = "uniform",
    ):
        return cls(
            name="configuration",
            size=size,
            labeled=False,
            edge_count=edge_count,
            prior_type=prior_type,
            canonical=canonical,
            edge_proposer_type=edge_proposer_type,
        )

    @classmethod
    def poisson(cls, size: int = 100, edge_count: int = 250):
        return cls(
            "poisson",
            size=size,
            labeled=False,
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
            labeled=False,
            edge_count=edge_count,
            heterogeneity=heterogeneity,
        )

    @classmethod
    def stochastic_block_model(
        cls,
        size: int = 100,
        edge_count: float = 250,
        block_count: int = 0,
        likelihood_type: str = "uniform",
        block_prior_type: str = "uniform",
        label_graph_prior_type: str = "uniform",
        degree_prior_type: str = "uniform",
        canonical: bool = False,
        with_self_loops=True,
        with_parallel_edges=True,
        edge_proposer_type: str = "uniform",
        block_proposer_type: str = "uniform",
        sample_label_count_prob: float = 0.1,
        label_creation_prob: float = 0.5,
        shift: float = 1,
    ):
        return cls(
            name="stochastic_block_model",
            size=size,
            labeled=True,
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
            name="planted_partition",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
            labeled=True,
            stub_labeled=stub_labeled,
            with_self_loops=with_self_loops,
            with_parallel_edges=with_parallel_edges,
        )


class RandomGraphFactory(Factory):
    @staticmethod
    def build_erdosrenyi(config: RandomGraphConfig) -> ErdosRenyiModel:
        return ErdosRenyiModel(
            config.size,
            config.edge_count,
            canonical=config.canonical,
            with_self_loops=config.with_self_loops,
            with_parallel_edges=config.with_parallel_edges,
            edge_proposer_type=config.edge_proposer_type,
        )

    @staticmethod
    def build_configuration(config: RandomGraphConfig) -> ConfigurationModelFamily:
        return ConfigurationModelFamily(
            config.size,
            config.edge_count,
            hyperprior=(config.prior_type == "hyperprior"),
            canonical=config.canonical,
            edge_proposer_type=config.edge_proposer_type,
        )

    @staticmethod
    def build_poisson(config: RandomGraphConfig) -> ConfigurationModel:
        avgk = 2 * config.edge_count / config.size
        degrees = poisson_degreeseq(config.size, avgk).tolist()
        return ConfigurationModel(degrees)

    @staticmethod
    def build_nbinom(config: RandomGraphConfig) -> ConfigurationModel:
        avgk = 2 * config.edge_count / config.size
        degrees = nbinom_degreeseq(config.size, avgk, config.heterogeneity).tolist()
        return ConfigurationModel(degrees)

    @staticmethod
    def build_stochastic_block_model(
        config: RandomGraphConfig,
    ) -> Union[
        StochasticBlockModelFamily,
        NestedStochasticBlockModelFamily,
        DegreeCorrectedStochasticBlockModelFamily,
        NestedDegreeCorrectedStochasticBlockModelFamily,
    ]:
        if config.likelihood_type == "degree_corrected":
            if config.label_graph_prior_type == "nested":
                return NestedDegreeCorrectedStochasticBlockModelFamily(
                    config.size,
                    config.edge_count,
                    degree_hyperprior=(config.degree_prior_type == "hyper"),
                    canonical=config.canonical,
                    edge_proposer_type=config.edge_proposer_type,
                    block_proposer_type=config.block_proposer_type,
                    sample_label_count_prob=config.sample_label_count_prob,
                    label_creation_prob=config.label_creation_prob,
                    shift=config.shift,
                )
            else:
                return DegreeCorrectedStochasticBlockModelFamily(
                    config.size,
                    config.edge_count,
                    block_count=config.block_count,
                    block_hyperprior=(config.block_prior_type == "hyper"),
                    degree_hyperprior=(config.degree_prior_type == "hyper"),
                    planted=(config.label_graph_prior_type == "planted"),
                    canonical=config.canonical,
                    edge_proposer_type=config.edge_proposer_type,
                    block_proposer_type=config.block_proposer_type,
                    sample_label_count_prob=config.sample_label_count_prob,
                    label_creation_prob=config.label_creation_prob,
                    shift=config.shift,
                )
        else:
            if config.label_graph_prior_type == "nested":
                return NestedStochasticBlockModelFamily(
                    config.size,
                    config.edge_count,
                    stub_labeled=(config.likelihood_type == "stub_labeled"),
                    canonical=config.canonical,
                    with_self_loops=config.with_self_loops,
                    with_parallel_edges=config.with_parallel_edges,
                    edge_proposer_type=config.edge_proposer_type,
                    block_proposer_type=config.block_proposer_type,
                    sample_label_count_prob=config.sample_label_count_prob,
                    label_creation_prob=config.label_creation_prob,
                    shift=config.shift,
                )
            else:
                return StochasticBlockModelFamily(
                    config.size,
                    config.edge_count,
                    block_count=config.block_count,
                    block_hyperprior=(config.block_prior_type == "hyper"),
                    planted=(config.label_graph_prior_type == "planted"),
                    stub_labeled=(config.likelihood_type == "stub_labeled"),
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
    def build_planted_partition(config: RandomGraphConfig):
        return PlantedPartitionModel(
            config.size,
            config.edge_count,
            block_count=config.block_count,
            assortativity=config.assortativity,
            stub_labeled=config.stub_labeled,
            with_self_loops=config.with_self_loops,
            with_parallel_edges=config.with_parallel_edges,
        )


if __name__ == "__main__":
    pass
