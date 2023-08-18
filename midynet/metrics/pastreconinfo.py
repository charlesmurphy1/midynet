import time
import numpy as np

from collections import defaultdict
from graphinf.utility import seed as gi_seed
from midynet.config import (
    GraphFactory,
    DataModelFactory,
)

from midynet.config import Config
from ..statistics import Statistics
from .metrics import Metrics
from .multiprocess import Expectation

__all__ = [
    "PastDependentInformationMeasures",
    "PastDependentInformationMeasuresMetrics",
]


class PastDependentInformationMeasures(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> float:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        data_model = DataModelFactory.build(config.data_model)
        metrics_cf = config.metrics.recon_information

        data_model.set_graph_prior(prior)
        if config.target == "None":
            prior.sample()
            g0 = prior.state()
        else:
            g0 = prior.state()
        x0 = data_model.random_state(config.data_model.get("n_active", -1))
        data_model.set_graph(g0)
        data_model.sample_state(x0)
        out = {}

        # computing full
        og = data_model.graph()

        data_model.set_graph(og)
        full = self.gather(data_model, metrics_cf)
        out.update(full)
        out["mutualinfo"] = full["prior"] - full["posterior"]

        # computing past
        past_length = config.data_model.past_length
        if isinstance(past_length, float):
            past_length = int(past_length * config.data_model.length)
        elif past_length < 0:
            past_length = config.data_model.length + past_length
        data_model.set_length(past_length)
        past = self.gather(data_model, metrics_cf)
        data_model.set_length(config.data_model.length)
        out["likelihood_past"] = past["likelihood"]
        out["evidence_past"] = past["evidence"]
        out["posterior_past"] = past["posterior"]
        out["mutualinfo_past"] = past["prior"] - past["posterior"]

        if prior.labeled:
            out["graph_joint"] = prior.log_joint()
            out["graph_prior"] = prior.label_log_joint()
            out["graph_evidence"] = -full["prior"]
            out["graph_posterior"] = (
                out["graph_joint"] - out["graph_evidence"]
            )
        if metrics_cf.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out

    def gather(self, data_model, metrics_cf):
        graph_mcmc = (
            metrics_cf.graph_mcmc
            if metrics_cf.graph_mcmc is not None
            else dict()
        )
        graph_mcmc.pop("reset_to_original")
        prior = -data_model.graph_prior.log_evidence(
            reset_to_original=True, **metrics_cf.graph_mcmc
        )
        likelihood = -data_model.log_likelihood()
        posterior = data_model.log_posterior(**metrics_cf.data_mcmc)
        evidence = prior + likelihood - posterior
        return dict(
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
        )


class PastDependentInformationMeasuresMetrics(Metrics):
    shortname = "pastinfo"
    keys = [
        "prior",
        "likelihood",
        "posterior",
        "evidence",
        "prior_past",
        "likelihood_past",
        "posterior_past",
        "evidence_past",
    ]

    def __init__(self, **kwargs):
        super().__init__("recon_information", **kwargs)

    def eval(self, config: Config):
        metrics = PastDependentInformationMeasures(
            config=config,
            n_workers=config.get("n_workers", 1),
            seed=config.get("seed", int(time.time())),
        )

        samples = metrics.compute(
            config.metrics.recon_information.get("n_samples", 10)
        )
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)

        if config.metrics.recon_information.reduction == "identity":
            return sample_dict
        stats = {}
        for k, v in sample_dict.items():
            s = Statistics.from_array(
                v,
                reduction=config.metrics.recon_information.get(
                    "reduction", "normal"
                ),
            )
            for sk, sv in s.__data__.items():
                stats[k + "_" + sk] = [sv]
        return stats


if __name__ == "__main__":
    pass
