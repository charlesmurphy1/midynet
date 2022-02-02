from typing import Union, Optional

from _midynet.prior import sbm
from _midynet.random_graph import (
    ConfigurationModelFamily,
    DegreeCorrectedStochasticBlockModelFamily,
    ErdosRenyiFamily,
    SimpleErdosRenyiFamily,
    StochasticBlockModelFamily,
)

from midynet.util.degree_sequences import poisson_degreeseq, nbinom_degreeseq

from .config import Config
from .factory import Factory, UnavailableOption
from .prior import (
    EdgeCountPriorConfig,
    BlockPriorConfig,
    EdgeMatrixPriorConfig,
    DegreePriorConfig,
    EdgeCountPriorFactory,
    BlockPriorFactory,
    EdgeMatrixPriorFactory,
    DegreePriorFactory,
)
from .proposer import EdgeProposerConfig, BlockProposerConfig
from .wrapper import Wrapper

__all__ = ("RandomGraphConfig", "RandomGraphFactory")


class RandomGraphConfig(Config):
    requirements: set[str] = {"size"}

    @classmethod
    def custom_sbm(
        cls,
        name: str,
        size: int,
        blocks: BlockPriorConfig,
        edge_matrix: EdgeMatrixPriorConfig,
    ):
        obj = cls(name=name, size=size)
        obj.insert("blocks", BlockPriorConfig.auto(blocks))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.auto(edge_matrix))
        if obj.edge_matrix.edge_count.name == "delta":
            obj.insert(
                "edge_proposer", EdgeProposerConfig.hinge_flip_uniform()
            )
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        obj.insert("block_proposer", BlockProposerConfig.peixoto())
        obj.insert("sample_graph_prior_prob", 0.5)

        return obj

    @classmethod
    def uniform_sbm(
        cls,
        size: int = 100,
        edge_count: Union[int, float] = 250,
        block_count_max: Optional[Union[int, float]] = None,
    ):
        blocks = BlockPriorConfig.uniform()
        blocks = BlockPriorConfig.uniform()
        blocks.block_count.set_value("max", block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom_sbm("uniform_sbm", size, blocks, edge_matrix)

    @classmethod
    def hyperuniform_sbm(
        cls,
        size: int = 100,
        edge_count: Union[int, float] = 250,
        block_count_max: Optional[Union[int, float]] = None,
    ):
        blocks = BlockPriorConfig.hyperuniform()
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        return cls.custom_sbm("hyperuniform_sbm", size, blocks, edge_matrix)

    def planted_partition(
        cls, size: int, edge_count: int, block_count: int, assortativity: float
    ):
        obj = cls(
            "planted_partition",
            size=size,
            edge_count=edge_count,
            block_count=block_count,
            assortativity=assortativity,
        )
        obj.insert("blocks", BlockPriorConfig.delta(block_count))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.uniform(edge_count))
        obj.insert("edge_proposer", EdgeProposerConfig.hinge_flip_uniform())
        obj.insert("sample_graph_prior_prob", 0.0)
        return obj

    @classmethod
    def custom_er(cls, name: str, size: int, edge_count: EdgeCountPriorConfig):
        obj = cls(name=name, size=size)
        obj.insert("edge_count", EdgeCountPriorConfig.auto(edge_count))

        if obj.edge_count.name == "delta":
            obj.insert(
                "edge_proposer", EdgeProposerConfig.hinge_flip_uniform()
            )
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        obj.insert("sample_graph_prior_prob", 0.0)
        return obj

    @classmethod
    def er(cls, size: int = 100, edge_count: Union[int, float] = 250):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        return cls.custom_er(name="er", size=size, edge_count=edge_count)

    @classmethod
    def ser(cls, size: int = 100, edge_count: Union[int, float] = 250):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        obj = cls.custom_er(name="ser", size=size, edge_count=edge_count)
        obj.edge_proposer.set_value("allow_self_loops", False)
        obj.edge_proposer.set_value("allow_multiedges", False)
        obj.insert("sample_graph_prior_prob", 0.0)
        return obj

    @classmethod
    def custom_dcsbm(
        cls,
        name: str,
        size: int,
        blocks: BlockPriorConfig,
        edge_matrix: EdgeMatrixPriorConfig,
        degrees: DegreePriorConfig,
    ):
        obj = cls(name=name, size=size)
        obj.insert("blocks", BlockPriorConfig.auto(blocks))
        obj.insert("edge_matrix", EdgeMatrixPriorConfig.auto(edge_matrix))
        obj.insert("degrees", DegreePriorConfig.auto(degrees))
        if obj.degrees.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        elif obj.edge_matrix.edge_count.name == "delta":
            obj.insert(
                "edge_proposer", EdgeProposerConfig.hinge_flip_uniform()
            )
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        obj.insert("block_proposer", BlockProposerConfig.peixoto())
        obj.insert("sample_graph_prior_prob", 0.5)
        return obj

    @classmethod
    def uniform_dcsbm(
        cls,
        size: int = 100,
        edge_count: Union[int, float] = 250,
        block_count_max: Optional[Union[int, float]] = None,
    ):
        blocks = BlockPriorConfig.uniform()
        blocks.block_count.set_value("max", block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        degrees = DegreePriorConfig.uniform()

        return cls.custom_dcsbm(
            "uniform_dcsbm", size, blocks, edge_matrix, degrees
        )

    @classmethod
    def hyperuniform_dcsbm(
        cls,
        size: int = 100,
        edge_count: Union[int, float] = 250,
        block_count_max: Optional[Union[int, float]] = None,
    ):
        blocks = BlockPriorConfig.hyperuniform()
        blocks.block_count.set_value("max", block_count_max)
        edge_matrix = EdgeMatrixPriorConfig.uniform(edge_count)
        degrees = DegreePriorConfig.hyperuniform()
        return cls.custom_dcsbm(
            "hyperuniform_dcsbm", size, blocks, edge_matrix, degrees
        )

    @classmethod
    def custom_cm(
        cls,
        name: str,
        size: int,
        edge_count: EdgeCountPriorConfig,
        degrees: DegreePriorConfig,
    ):
        obj = cls(name=name, size=size)
        obj.insert("edge_count", EdgeCountPriorConfig.auto(edge_count))
        obj.insert("degrees", DegreePriorConfig.auto(degrees))
        if obj.degrees.name == "delta":
            obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        elif obj.edge_count.name == "delta":
            obj.insert(
                "edge_proposer", EdgeProposerConfig.hinge_flip_uniform()
            )
        else:
            obj.insert("edge_proposer", EdgeProposerConfig.single_uniform())
        obj.insert("sample_graph_prior_prob", 0.0)
        return obj

    @classmethod
    def poisson_cm(cls, size: int = 100, edge_count: Union[int, float] = 250):
        obj = cls(
            "poisson_cm",
            size=size,
            edge_count=EdgeCountPriorConfig.auto(edge_count),
        )
        obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        obj.insert("sample_graph_prior_prob", 0.0)

        return obj

    @classmethod
    def nbinom_cm(
        cls,
        size: int = 100,
        edge_count: Union[int, float] = 250,
        heterogeneity: int = 0,
    ):
        obj = cls(
            "nbinom_cm",
            size=size,
            edge_count=EdgeCountPriorConfig.auto(edge_count),
            heterogeneity=heterogeneity,
        )
        obj.insert("edge_proposer", EdgeProposerConfig.double_swap())
        obj.insert("sample_graph_prior_prob", 0.0)

        return obj

    @classmethod
    def uniform_cm(cls, size: int = 100, edge_count: Union[int, float] = 250):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        degrees = DegreePriorConfig.uniform()
        return cls.custom_cm("uniform_cm", size, edge_count, degrees)

    @classmethod
    def hyperuniform_cm(
        cls, size: int = 100, edge_count: Union[int, float] = 250
    ):
        edge_count = EdgeCountPriorConfig.auto(edge_count)
        degrees = DegreePriorConfig.hyperuniform()
        return cls.custom_cm("hyperuniform_cm", size, edge_count, degrees)


class RandomGraphFactory(Factory):
    @staticmethod
    def setUpSBM(graph, blocks, edge_matrix):
        graph.set_block_prior(blocks)
        graph.set_edge_matrix_prior(edge_matrix)

    @staticmethod
    def setUpER(graph, edge_count):
        graph.set_edge_count_prior(edge_count)

    @staticmethod
    def setUpDCSBM(
        graph: DegreeCorrectedStochasticBlockModelFamily,
        blocks: sbm.BlockPrior,
        edge_matrix: sbm.EdgeMatrixPrior,
        degrees: sbm.DegreePrior,
    ):
        graph.set_block_prior(blocks)
        graph.set_edge_matrix_prior(edge_matrix)
        graph.set_degree_prior(degrees)

    @staticmethod
    def setUpCM(
        graph, edge_count: sbm.EdgeCountPrior, degrees: sbm.DegreePrior
    ) -> None:
        graph.set_edge_count_prior(edge_count)
        graph.set_degree_prior(degrees)

    @staticmethod
    def build_custom_sbm(
        config: RandomGraphConfig,
    ) -> StochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        block_wrapper.set_size(config.size)
        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)

        g = StochasticBlockModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpSBM(
                wrap,
                others["blocks"].get_wrap(),
                others["edge_matrix"].get_wrap(),
            ),
            blocks=block_wrapper,
            edge_matrix=edge_matrix_wrapper,
        )

    @staticmethod
    def build_custom_fixed_sbm(
        blocks: list[int], edge_matrix: list[list[int]]
    ) -> StochasticBlockModelFamily:
        UnavailableOption("fixed_sbm")

    @staticmethod
    def build_uniform_sbm(
        config: RandomGraphConfig,
    ) -> StochasticBlockModelFamily:
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_sbm(config)

    @staticmethod
    def build_hyperuniform_sbm(
        config: RandomGraphConfig,
    ) -> StochasticBlockModelFamily:
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_sbm(config)

    @staticmethod
    def build_planted_partition(
        config: RandomGraphConfig,
    ) -> StochasticBlockModelFamily:
        UnavailableOption(config.name)

    @staticmethod
    def build_custom_er(config: RandomGraphConfig) -> ErdosRenyiFamily:
        edge_count = EdgeCountPriorFactory.build(config.edge_count)
        g = ErdosRenyiFamily(config.size, edge_count)

        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpER(
                wrap,
                others["edge_count"],
            ),
            edge_count=edge_count,
        )

    @staticmethod
    def build_er(config: RandomGraphConfig) -> ErdosRenyiFamily:
        return RandomGraphFactory.build_custom_er(config)

    @staticmethod
    def build_ser(config: RandomGraphConfig) -> ErdosRenyiFamily:
        edge_count = EdgeCountPriorFactory.build(config.edge_count)
        g = SimpleErdosRenyiFamily(config.size, edge_count)

        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpER(
                wrap,
                others["edge_count"],
            ),
            edge_count=edge_count,
        )

    @staticmethod
    def build_custom_dcsbm(
        config: RandomGraphConfig,
    ) -> DegreeCorrectedStochasticBlockModelFamily:
        block_wrapper = BlockPriorFactory.build(config.blocks)
        block_wrapper.set_size(config.size)
        edge_matrix_wrapper = EdgeMatrixPriorFactory.build(config.edge_matrix)
        degrees = DegreePriorFactory.build(config.degrees)
        g = DegreeCorrectedStochasticBlockModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpDCSBM(
                wrap,
                others["blocks"].get_wrap(),
                others["edge_matrix"].get_wrap(),
                others["degrees"],
            ),
            blocks=block_wrapper,
            edge_matrix=edge_matrix_wrapper,
            degrees=degrees,
        )

    @staticmethod
    def build_uniform_dcsbm(
        config: RandomGraphConfig,
    ) -> DegreeCorrectedStochasticBlockModelFamily:
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_dcsbm(config)

    @staticmethod
    def build_hyperuniform_dcsbm(
        config: RandomGraphConfig,
    ) -> DegreeCorrectedStochasticBlockModelFamily:
        config.blocks.block_count.max = (
            config.size
            if config.blocks.block_count.max is None
            else config.blocks.block_count.max
        )
        return RandomGraphFactory.build_custom_dcsbm(config)

    @staticmethod
    def build_custom_cm(
        config: RandomGraphConfig,
    ) -> ConfigurationModelFamily:
        edge_count = EdgeCountPriorFactory.build(config.edge_count)
        degrees = DegreePriorFactory.build(config.degrees)
        g = ConfigurationModelFamily(config.size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpCM(
                wrap,
                others["edge_count"],
                others["degrees"],
            ),
            edge_count=edge_count,
            degrees=degrees,
        )

    @staticmethod
    def build_fixed_custom_cm(
        degrees: list[int],
    ) -> ConfigurationModelFamily:
        size, edge_count = len(degrees), int(sum(degrees) / 2)
        degree_prior = sbm.DegreeDeltaPrior(degrees)
        edge_count_prior = sbm.EdgeCountDeltaPrior(edge_count)
        g = ConfigurationModelFamily(size)
        return Wrapper(
            g,
            setup_func=lambda wrap, others: RandomGraphFactory.setUpCM(
                wrap,
                others["edge_count"],
                others["degrees"],
            ),
            edge_count=edge_count_prior,
            degrees=degree_prior,
        )

    @staticmethod
    def build_poisson_cm(
        config: RandomGraphConfig,
    ) -> ConfigurationModelFamily:
        degrees = poisson_degreeseq(
            config.size, 2 * config.edge_count.state / config.size
        )
        return RandomGraphFactory.build_fixed_custom_cm(degrees)

    @staticmethod
    def build_nbinom_cm(config: RandomGraphConfig) -> ConfigurationModelFamily:
        degrees = nbinom_degreeseq(
            config.size,
            2 * config.edge_count.state / config.size,
            config.heterogeneity,
        )
        return RandomGraphFactory.build_fixed_custom_cm(degrees)

    @staticmethod
    def build_uniform_cm(
        config: RandomGraphConfig,
    ) -> ConfigurationModelFamily:
        return RandomGraphFactory.build_custom_cm(config)

    @staticmethod
    def build_hyperuniform_cm(
        config: RandomGraphConfig,
    ) -> ConfigurationModelFamily:
        return RandomGraphFactory.build_custom_cm(config)


if __name__ == "__main__":
    pass
