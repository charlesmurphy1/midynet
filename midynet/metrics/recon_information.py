import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from graphinf.utility import seed as gi_seed
from midynet.config import (
    GraphFactory,
    DataModelFactory,
)

from midynet.config import Config
from .metrics import Metrics
from .aggregator import Aggregator
from .multiprocess import Expectation
from midynet.statistics import Statistics
from .util import (
    get_log_posterior_meanfield,
    get_log_evidence,
    get_graph_log_evidence_meanfield,
    get_graph_log_evidence_annealed,
    get_graph_log_evidence_exact,
)

__all__ = ("MutualInformation", "MutualInformationMetrics")


class ReconstructionInformationMeasures(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.config = config
        super().__init__(**kwargs)

    def func(self, seed: int) -> float:
        gi_seed(seed)
        prior = GraphFactory.build(self.config.prior)
        data_model = DataModelFactory.build(self.config.data_model)
        metrics_cf = self.config.metrics.recon_information

        data_model.set_graph_prior(prior)
        if self.config.target == "None":
            prior.sample()
            g0 = prior.get_state()
        else:
            g0 = prior.get_state()
        x0 = data_model.get_random_state(
            self.config.data_model.get("num_active", -1)
        )
        data_model.set_graph(g0)
        data_model.sample_state(x0)
        out = {}

        # computing full
        og = data_model.get_graph()

        data_model.set_graph(og)
        full = self.gather(data_model, metrics_cf)
        out.update(full)
        out["mutualinfo"] = full["prior"] - full["posterior"]

        # computing past
        if (
            "past_length" in self.config.data_model
            and self.config.data_model.past_length > 0
        ):
            if self.config.data_model.past_length == 0:
                past = dict(
                    prior=full["prior"],
                    likelihood=0,
                    evidence=0,
                    posterior=full["prior"],
                )
            else:
                past_length = self.config.data_model.past_length
                if isinstance(past_length, float):
                    past_length = int(
                        past_length * self.config.data_model.length
                    )
                elif past_length < 0:
                    past_length = self.config.data_model.length + past_length
                data_model.set_length(past_length)
                past = self.gather(data_model, metrics_cf)
                data_model.set_length(self.config.data_model.length)
            out["likelihood_past"] = past["likelihood"]
            out["evidence_past"] = past["evidence"]
            out["posterior_past"] = past["posterior"]
            out["mutualinfo_past"] = past["prior"] - past["posterior"]

        if prior.labeled:
            out["graph_joint"] = prior.get_log_joint()
            out["graph_prior"] = prior.get_label_log_joint()
            out["graph_evidence"] = -full["prior"]
            out["graph_posterior"] = (
                out["graph_joint"] - out["graph_evidence"]
            )
        if metrics_cf.get("to_bits", True):
            out = {k: v / np.log(2) for k, v in out.items()}
        return out

    def gather(self, data_model, metrics_cf):
        method = metrics_cf.get("method", "meanfield")
        graph_evidence_method = metrics_cf.get(
            "graph_evidence_method", method
        )

        og = data_model.get_graph()

        if not data_model.graph_prior.labeled:
            prior = -data_model.graph_prior.get_log_joint()
        elif graph_evidence_method == "exact":
            prior = -get_graph_log_evidence_exact(
                data_model.graph_prior, metrics_cf
            )
        elif graph_evidence_method == "annealed":
            prior = -get_graph_log_evidence_annealed(
                data_model.graph_prior, metrics_cf
            )
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
            prior=prior,
            likelihood=likelihood,
            posterior=posterior,
            evidence=evidence,
        )


class ReconstructionInformationMeasuresMetrics(Metrics):
    def __init__(self, **kwargs):
        super().__init__("recon_information", **kwargs)

    def eval(self, config: Config):
        metrics = ReconstructionInformationMeasures(
            config=config,
            num_procs=config.get("num_procs", 1),
            seed=config.get("seed", int(time.time())),
        )

        samples = metrics.compute(
            config.metrics.recon_information.get("num_samples", 10)
        )
        sample_dict = defaultdict(list)
        for s in samples:
            for k, v in s.items():
                sample_dict[k].append(v)
        # res = {
        #     k: Statistics.compute(
        #         v,
        #         error_type=config.metrics.recon_information.get(
        #             "error_type", "std"
        #         ),
        #     )
        #     for k, v in sample_dict.items()
        # }
        # out = {
        #     f"{k}-{kk}": vv for k, v in res.items() for kk, vv in v.items()
        # }
        # e = out["evidence-mid"]
        # l = out["likelihood-mid"]
        # po = out["posterior-mid"]
        # pr = out["prior-mid"]
        # mi = out["mutualinfo-mid"]
        # print(f"Full: {e=}, {l=}, {pr=}, {po=}, {mi=}")

        # e_p = out["evidence_past-mid"]
        # l_p = out["likelihood_past-mid"]
        # po_p = out["posterior_past-mid"]
        # mi_p = out["mutualinfo_past-mid"]
        # print(f"Past: {e_p=}, {l_p=}, {pr=}, {po_p=}, {mi_p=}")
        # print(out)
        return {
            k: getattr(Aggregator, config.get("stat_type", "std"))(v)
            for k, v in sample_dict.items()
        }


if __name__ == "__main__":
    pass
