import os
import dotenv

from math import ceil
from itertools import product
from midynet.config import (
    ExperimentConfig,
    DataModelConfig,
    GraphConfig,
    MetricsConfig,
)
from midynet.config.metrics import MCMCDataConfig, MCMCGraphConfig
from midynet.config.experiments.util import format_sequence

dotenv.load_dotenv(os.path.join(os.path.expanduser("~"), ".md-env"))


class InfoGainSyntheticGraphsScriptConfig(ExperimentConfig):
    models = {
        "erdosrenyi": DataModelConfig.glauber(length=1000, coupling=0.5),
        "planted_partition": DataModelConfig.glauber(
            length=1000, coupling=0.5
        ),
        "large_erdosrenyi": DataModelConfig.glauber(
            length=2000, coupling=0.5
        ),
        "large_planted_partition": DataModelConfig.glauber(
            length=2000, coupling=0.5
        ),
        "karate": DataModelConfig.glauber(length=1000, coupling=0.5),
        "littlerock": DataModelConfig.glauber(length=2000, coupling=0.03),
        "polblogs": DataModelConfig.glauber(length=2000, coupling=0.013),
        "euairlines": DataModelConfig.glauber(length=1000, coupling=0.02),
        "celegans": DataModelConfig.glauber(length=1000, coupling=0.015),
    }
    priors = {
        "erdosrenyi": GraphConfig.erdosrenyi(
            size=100, edge_count=250, loopy=True, multigraph=True
        ),
        "configuration": GraphConfig.configuration(100, 250),
        "degree_corrected_stochastic_block_model": GraphConfig.degree_corrected_stochastic_block_model(
            size=100,
            edge_count=250,
        ),
        "stochastic_block_model": GraphConfig.stochastic_block_model(
            size=100, edge_count=250
        ),
    }
    targets = {
        "erdosrenyi": GraphConfig.erdosrenyi(size=100, edge_count=250),
        "planted_partition": GraphConfig.planted_partition(
            size=100, edge_count=250, block_count=3
        ),
        "large_erdosrenyi": GraphConfig.erdosrenyi(
            size=1000, edge_count=2500
        ),
        "large_planted_partition": GraphConfig.planted_partition(
            size=1000, edge_count=2500, block_count=10
        ),
        "karate": GraphConfig.karate(
            path=os.path.join(
                os.getenv("MD-DATA_PATH", "."), "graphs/karate.npy"
            )
        ),
        "littlerock": GraphConfig.littlerock(
            path=os.path.join(
                os.getenv("MD-DATA_PATH", "."), "graphs/littlerock.npy"
            )
        ),
        "polblogs": GraphConfig.polblogs(
            path=os.path.join(
                os.getenv("MD-DATA_PATH", "."), "graphs/polblogs.npy"
            ),
        ),
        "euairlines": GraphConfig.euairlines(
            path=os.path.join(
                os.getenv("MD-DATA_PATH", "."), "graphs/euairlines.npy"
            ),
        ),
        "celegans": GraphConfig.euairlines(
            path=os.path.join(
                os.getenv("MD-DATA_PATH", "."), "graphs/celegans.npy"
            ),
        ),
    }

    @staticmethod
    def default(
        save_path,
        prior,
        target,
        n_workers=os.getenv("MD-N_WORKERS", 1),
        n_samples_per_worker=4,
        n_async_jobs=1,
        n_sweeps=1000,
        seed=None,
    ):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_mcmc = MCMCDataConfig.meanfield(
            n_sweeps=n_sweeps, n_gibbs_sweeps=2
        )
        graph_mcmc = MCMCGraphConfig.meanfield(
            n_sweeps=n_sweeps, n_gibbs_sweeps=2
        )
        metrics = [
            MetricsConfig.entropy(
                data_mcmc=data_mcmc,
                graph_mcmc=graph_mcmc,
                reduction="identity",
                n_samples=n_samples_per_worker
                * ceil(n_workers / n_async_jobs),
            )
        ]
        config = ExperimentConfig.default(
            f"infogain-syn-{target}-with-{prior}",
            data_model=InfoGainSyntheticGraphsScriptConfig.models[target],
            prior=InfoGainSyntheticGraphsScriptConfig.priors[prior],
            target=InfoGainSyntheticGraphsScriptConfig.targets[target],
            metrics=metrics,
            path=save_path,
            n_workers=n_workers,
            n_async_jobs=n_async_jobs,
            seed=seed,
        )

        config.lock()
        return config

    @staticmethod
    def all(
        name="infogain-syn",
        **kwargs,
    ):
        path = lambda t, p: os.path.join(
            os.getenv("MD-DATA_PATH", "./"), name, f"recon-{t}-with-{p}"
        )
        t_keys = ["erdosrenyi", "planted_partition", "karate", "littlerock"]
        p_keys = InfoGainSyntheticGraphsScriptConfig.priors.keys()
        exps = [
            InfoGainSyntheticGraphsScriptConfig.default(
                target=t,
                prior=p,
                save_path=path(t, p),
                **kwargs,
            )
            for t, p in product(t_keys, p_keys)
        ]
        return exps

    @staticmethod
    def test(**kwargs):
        return InfoGainSyntheticGraphsScriptConfig.all(
            name="test-infogain-syn",
            n_sweeps=kwargs.pop("n_sweeps", 10),
            n_samples_per_worker=kwargs.pop("n_samples_per_worker", 1),
            **kwargs,
        )
