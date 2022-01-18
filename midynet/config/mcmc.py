from __future__ import annotations

from .config import Config

from .factory import Factory
from .wrapper import Wrapper
from .proposer import *
from _midynet import proposer
from _midynet import mcmc

__all__ = ["RandomGraphMCMCFactory"]


class RandomGraphMCMCFactory(Factory):
    @classmethod
    def build(cls, config: RandomGraphConfig) -> Any:
        if config.unmet_requirements():
            raise MissingRequirementsError(config)
        options = {
            k[6:]: getattr(cls, k) for k in cls.__dict__.keys() if k[:6] == "build_"
        }
        name = "sbmlike" if "block_proposer" in config else "generic"
        if name in options:
            return options[name](config)
        else:
            raise OptionError(actual=name, expected=list(options.keys()))

    @staticmethod
    def setup_sbm_mcmc(mcmc_model, edge_proposer, block_proposer):
        mcmc_model.set_edge_proposer(edge_proposer)
        mcmc_model.set_block_proposer(block_proposer)

    @staticmethod
    def setup_er_mcmc(mcmc_model, edge_proposer):
        mcmc_model.set_edge_proposer(edge_proposer)

    @staticmethod
    def build_generic(config: RandomGraphConfig) -> mcmc.RandomGraphMCMC:
        edge_proposer = EdgeProposerFactory.build(config.edge_proposer)
        mcmc_model = mcmc.RandomGraphMCMC()
        return Wrapper(
            mcmc_model,
            setup_func=lambda wrap, others: RandomGraphMCMCFactory.setup_er_mcmc(
                wrap, others["edge_proposer"]
            ),
            edge_proposer=edge_proposer,
        )

    @staticmethod
    def build_sbmlike(config: RandomGraphConfig) -> mcmc.StochasticBlockGraphMCMC:
        edge_proposer = EdgeProposerFactory.build(config.edge_proposer)
        block_proposer = BlockProposerFactory.build(config.block_proposer)
        mcmc_model = mcmc.StochasticBlockGraphMCMC()
        return Wrapper(
            mcmc_model,
            setup_func=lambda wrap, others: RandomGraphMCMCFactory.setup_sbm_mcmc(
                wrap, others["edge_proposer"], others["block_proposer"]
            ),
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )


if __name__ == "__main__":
    pass
