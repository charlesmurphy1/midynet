import time
from typing import Any, Dict, Tuple

import numpy as np
from graphinf.utility import seed as gi_seed
from midynet.config import Config, DataModelFactory, GraphFactory

from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ("ReconstructionInformation", "ReconstructionInformationMetrics")


class ReconstructionInformationMeasures(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def setup(self, seed: int) -> Tuple[Config, Dict[str, Any]]:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.model)

        model.set_graph_prior(prior)

        prior.sample()
        g0 = prior.get_state()
        model.set_graph(g0)
        if "n_active" in config.model:
            x0 = model.get_random_state(config.model.get("n_active", -1))
            model.sample_state(x0)
        else:
            model.sample_state()

        return config, dict(model=model, prior=prior)

    def gather(self, model, config):
        graph_mcmc = config.metrics.get("graph_mcmc", Config("c")).dict
        data_mcmc = config.metrics.get("data_mcmc", Config("c")).dict
        graph_mcmc.pop("name", None)
        data_mcmc.pop("name", None)

        prior = -model.graph_prior.log_evidence(**graph_mcmc)
        likelihood = -model.get_log_likelihood()
        posterior = -model.get_log_posterior(**data_mcmc)
        evidence = prior + likelihood - posterior
        return dict(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
        )

    def func(self, seed: int) -> float:
        config, model_dict = self.setup(seed)
        model, prior = model_dict["model"], model_dict["prior"]
        out = self.gather(model, config)
        out["mutualinfo"] = out["prior"] - out["posterior"]

        if prior.labeled:
            out["graph_joint"] = prior.get_log_joint()
            out["graph_prior"] = prior.get_label_log_joint()
            out["graph_evidence"] = -out["prior"]
            out["graph_posterior"] = (
                out["graph_joint"] - out["graph_evidence"]
            )
        if config.metrics.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out


class ReconstructionInformationMeasuresMetrics(ExpectationMetrics):
    shortname = "reconinfo"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "mutualinfo",
        "recon",
        "pred",
    ]
    expectation_factory = ReconstructionInformationMeasures

    def postprocess(
        self, samples: list[Dict[str, float]]
    ) -> Dict[str, float]:
        stats = self.reduce(
            samples, self.configs.metrics.get("reduction", "normal")
        )
        stats["recon"] = stats["mutualinfo"] / stats["prior"]
        stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        return self.format(stats)


if __name__ == "__main__":
    pass
