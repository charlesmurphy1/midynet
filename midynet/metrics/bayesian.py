from math import ceil
from typing import Any, Dict, Tuple, List

import numpy as np
from graphinf.utility import seed as gi_seed
from graphinf.graph import RandomGraphWrapper
from midynet.config import Config, DataModelFactory, GraphFactory
from basegraph import core as bg

from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ("ReconstructionInformation", "ReconstructionInformationMetrics")


class BayesianInformationMeasures(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def setup(self, seed: int) -> Tuple[Config, Dict[str, Any]]:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)

        model.set_graph_prior(prior)
        if config.target == "None":
            prior.sample()
            g0 = prior.get_state()
        else:
            target = GraphFactory.build(config.target)
            if isinstance(target, bg.UndirectedMultigraph):
                g0 = target
            else:
                assert issubclass(target.__class__, RandomGraphWrapper)
                g0 = target.get_state()

        model.set_graph(g0)
        if config.metrics.get("resample_graph", False):
            prior.sample()

        if "n_active" in config.data_model:
            n0 = config.data_model.get("n_active", -1)
            n0 = ceil(n0 * g0.get_size()) if 0 < n0 < 1 else n0
            x0 = model.get_random_state(n0)
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
        likelihood = -model.log_likelihood()
        posterior = -model.log_posterior(**data_mcmc)
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
            out["graph_joint"] = prior.log_joint()
            out["graph_prior"] = prior.get_label_log_joint()
            out["graph_evidence"] = -out["prior"]
            out["graph_posterior"] = (
                out["graph_joint"] - out["graph_evidence"]
            )
        if config.metrics.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out


class BayesianInformationMeasuresMetrics(ExpectationMetrics):
    shortname = "bayesian"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "mutualinfo",
        "recon",
        "pred",
    ]
    expectation_factory = BayesianInformationMeasures

    def postprocess(
        self, samples: List[Dict[str, float]]
    ) -> Dict[str, float]:
        stats = self.reduce(
            samples, self.configs.metrics.get("reduction", "normal")
        )
        stats["recon"] = stats["mutualinfo"] / stats["prior"]
        stats["pred"] = stats["mutualinfo"] / stats["evidence"]
        out = self.format(stats)
        print(out)
        return out


if __name__ == "__main__":
    pass
