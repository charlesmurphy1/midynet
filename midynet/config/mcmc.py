from __future__ import annotations

from .config import Config

from .factory import Factory
from .wrapper import Wrapper
from .proposer import *
from _midynet.proposer.block_proposer import BlockGenericProposer
from _midynet import mcmc

__all__ = ["RandomGraphMCMCFactory"]


class RandomGraphMCMCFactory(Factory):
    @classmethod
    def build(cls, config: RandomGraphConfig) -> Any:
        if config.unmet_requirements():
            raise MissingRequirementsError(config)
        edge_proposer = EdgeProposerFactory.build(config.edge_proposer)
        block_proposer = (
            BlockProposerFactory.build(config.block_proposer)
            if "block_proposer" in config
            else BlockGenericProposer()
        )
        mcmc_model = mcmc.RandomGraphMCMC(edge_proposer, block_proposer)
        return Wrapper(
            mcmc_model,
            edge_proposer=edge_proposer,
            block_proposer=block_proposer,
        )


if __name__ == "__main__":
    pass
