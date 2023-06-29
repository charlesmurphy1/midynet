from copy import deepcopy
from typing import Dict

import numpy as np
from basegraph import core as bs
from graphinf.utility import seed as gi_seed
from graphinf.graph import RandomGraphWrapper
from midynet.config import Config, DataModelFactory, GraphFactory
from midynet.statistics import Statistics

from .heuristics import (
    AverageProbabilityPredictor,
    BayesianReconstructor,
    get_predictor,
    get_reconstructor,
    prepare_training_data,
)
from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ["ErrorProbMetrics"]


class ReconstructionError(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> Dict[str, float]:
        # Data generation
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)
        model.set_graph_prior(prior)

        if config.target != "None":
            prior.sample()
            g0 = prior.get_state()
        else:
            target = GraphFactory.build(config.target)
            if isinstance(target, bs.UndirectedMultigraph):
                g0 = target
            else:
                assert issubclass(target.__class__, RandomGraphWrapper)
                g0 = target.get_state()
        prior.from_graph(g0)

        if "n_active" in config.data_model:
            x0 = model.get_random_state(config.data_model.get("n_active", -1))

            model.sample_state(x0)
            x = np.array(model.get_past_states())
        else:
            model.sample_state()
            x = np.array(model.get_state())

        # Reconstruction
        if config.metrics.get("reconstructor") == "bayesian":
            reconstructor = BayesianReconstructor(config)
            reconstructor.model.set_state_from(model)
            data_mcmc = config.metrics.get("data_mcmc", Config("c")).dict
            reconstructor.fit(g0=g0, **data_mcmc)
        else:
            reconstructor = get_reconstructor(config.metrics)
            reconstructor.fit(x)

        # Evaluation
        out = reconstructor.compare(g0, measures=config.metrics.measures)
        out = {k: v for k, v in out.items() if isinstance(v, (float, int))}
        return out


class ReconstructionErrorMetrics(ExpectationMetrics):
    shortname = "recon_error"
    keys = "recon_error"
    expectation_factory = ReconstructionError

    def postprocess(
        self, samples: list[Dict[str, float]]
    ) -> Dict[str, Statistics]:
        out = super().postprocess(samples)
        return out


class PredictionError(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def func(self, seed: int) -> Dict[str, float]:
        # Data generation
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        model = DataModelFactory.build(config.data_model)
        model.set_graph_prior(prior)
        g0 = model.get_graph()

        if "n_active" in config.data_model:
            x0 = model.get_random_state(config.data_model.get("n_active", -1))

            model.sample_state(x0)
            x = np.array(model.get_past_states()).T
        else:
            model.sample_state()
            x = np.array(model.get_state()).T

        # Prediction
        if config.metrics.get("predictor") == "average_probability":
            predictor = AverageProbabilityPredictor(config)
            predictor.fit(
                inputs=model.get_past_states(),
                targets=model.get_future_states(),
                n_train_samples=config.metrics.get("n_train_samples", 100),
            )
        else:
            predictor = get_predictor(config.metrics)
            x_train, y_train = prepare_training_data(
                config,
                n_train_samples=config.metrics.get("n_train_samples", 100),
            )
            predictor.fit(x_train, y_train, **config.metrics)

        # Evaluation
        targets = np.array(model.get_transition_matrix(out_state=1)).T
        preds = predictor.predict(inputs=x)
        out = predictor.eval(targets, preds, measures=config.metrics.measures)
        out = {k: v for k, v in out.items() if isinstance(v, (float, int))}
        # out["mi"] = model.log_likelihood() - model.log_evidence(
        #     method="exact"
        # )
        return out


class PredictionErrorMetrics(ExpectationMetrics):
    shortname = "pred_error"
    keys = "pred_error"
    expectation_factory = PredictionError

    def postprocess(
        self, samples: list[Dict[str, float]]
    ) -> Dict[str, Statistics]:
        out = super().postprocess(samples)
        return out
