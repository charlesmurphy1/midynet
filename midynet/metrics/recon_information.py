import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from graphinf.utility import seed as gi_seed
from midynet.config import (
    Config,
    RandomGraphFactory,
    DataModelFactory,
)
from .metrics import Metrics
from .multiprocess import Expectation
from .statistics import Statistics
from .util import (
    get_log_posterior_meanfield,
    get_log_evidence,
    get_graph_log_evidence_meanfield,
    get_graph_log_evidence_annealed,
    get_graph_log_evidence_exact,
)
from graphinf.random_graph import ErdosRenyiModel

__all__ = ("MutualInformation", "MutualInformationMetrics")


@dataclass
class ReconstructionInformationMeasures(Expectation):
    config: Config = field(repr=False, default_factory=Config)

    def func(self, seed: int) -> float:
        gi_seed(seed)
        graph_model = RandomGraphFactory.build(self.config.graph_prior)
        data_model = DataModelFactory.build(self.config.data_model)
        metrics_cf = self.config.metrics.recon_information

        data_model.set_graph_prior(graph_model)
        data_model.sample()
        out = {}

        # computing full
        og = data_model.get_graph()

        l = data_model.get_log_likelihood()
        pr = data_model.get_log_prior()
        e = get_log_evidence(data_model, metrics_cf)
        po = l + pr - e

        import midynet

        _hx = []

        for _g in midynet.utility.enumerate_all_graphs(5, 5, False, False):
            data_model.set_graph(_g)
            _hx.append(data_model.get_log_joint())
        data_model.set_graph(og)
        full = self.gather(data_model, metrics_cf)
        out.update(full)
        # computing past
        if self.config.data_model.get_value("past_length", 0) > 0:
            data_model.set_length(self.config.data_model.past_length)
            past = self.gather(data_model, metrics_cf)
            future = {k: full[k] - past[k] for k in past.keys()}
            out.update({k + "_past": v for k, v in past.items()})
            out.update({k + "_future": v for k, v in future.items()})
            data_model.set_length(self.config.data_model.length)
        elif (
            "past_length" in self.config.data_model
            and self.config.data_model.past_length == 0
        ):
            out.update(
                {
                    "likelihood_past": 0,
                    "evidence_past": 0,
                    "posterior_past": full["prior"],
                    "likelihood_future": full["likelihood"],
                    "evidence_future": full["evidence"],
                    "posterior_future": full["prior"],
                }
            )

        if graph_model.labeled:
            out["graph_joint"] = graph_model.get_log_joint()
            out["graph_prior"] = graph_model.get_label_log_joint()
            out["graph_evidence"] = -hg
            out["graph_posterior"] = out["graph_joint"] - out["graph_evidence"]
        if metrics_cf.get_value("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out

    def gather(self, data_model, metrics_cf):
        method = metrics_cf.get_value("method", "meanfield")
        graph_evidence_method = metrics_cf.get_value("graph_evidence_method", method)

        og = data_model.get_graph()

        if not data_model.graph_prior.labeled:
            prior = -data_model.graph_prior.get_log_joint()
        elif graph_evidence_method == "exact":
            prior = -get_graph_log_evidence_exact(data_model.graph_prior, metrics_cf)
        elif graph_evidence_method == "annealed":
            prior = -get_graph_log_evidence_annealed(data_model.graph_prior, metrics_cf)
        else:
            prior = -get_graph_log_evidence_meanfield(
                data_model.graph_prior, metrics_cf
            )
        data_model.set_graph(og)
        likelihood = -data_model.get_log_likelihood()

        if method == "meanfield":
            posterior = -get_log_posterior_meanfield(data_model, metrics_cf)
            evidence = prior + likelihood - posterior
        else:
            evidence = -get_log_evidence(data_model, metrics_cf)
            posterior = prior + likelihood - evidence

        data_model.set_graph(og)
        return dict(
            prior=prior, likelihood=likelihood, posterior=posterior, evidence=evidence
        )


class ReconstructionInformationMeasuresMetrics(Metrics):
    def eval(self, config: Config):
        metrics = ReconstructionInformationMeasures(
            config=config,
            num_procs=config.get_value("num_procs", 1),
            seed=config.get_value("seed", int(time.time())),
        )

        samples = metrics.compute(
            config.metrics.recon_information.get_value("num_samples", 10)
        )
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)
        res = {
            k: Statistics.compute(
                v,
                error_type=config.metrics.recon_information.get_value(
                    "error_type", "std"
                ),
            )
            for k, v in sample_dict.items()
        }

        out = {f"{k}-{kk}": vv for k, v in res.items() for kk, vv in v.items()}
        e = out["evidence-mid"]
        l = out["likelihood-mid"]
        po = out["posterior-mid"]
        pr = out["prior-mid"]
        mi = e - l
        print(f"{e=}, {l=}, {pr=}, {po=}, {mi=}")
        return out


if __name__ == "__main__":
    pass
