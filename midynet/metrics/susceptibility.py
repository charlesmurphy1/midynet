import time
import numpy as np

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any
from graphinf.utility import seed as gi_seed
from midynet.config import (
    GraphFactory,
    DataModelFactory,
)

from midynet.config import Config
from midynet.statistics import Statistics
from .metrics import ExpectationMetrics
from .multiprocess import Expectation

__all__ = ("Susceptibility", "SusceptibilityMetrics")


def spin_average(x: np.ndarray) -> float:
    y = x.copy()
    y[x == 0] = -1
    y = y.mean(0)
    return np.mean(np.abs(y))


def spin_susceptibility(x: np.ndarray) -> float:
    y = x.copy()
    y[x == 0] = -1
    m = np.abs(y.mean(0))
    return (np.mean(m**2) - np.mean(m) ** 2) / np.mean(m)


def spreading_susceptibility(x: np.ndarray) -> float:
    n = x.sum(0)
    return (np.mean(n**2) - np.mean(n) ** 2) / np.mean(n)


def gap_susceptibility(x: np.ndarray, epsilon=0.05) -> float:
    y = x.copy()
    y = y.mean(0)
    if np.any(y < epsilon):
        y = y[y < epsilon]
    return np.std(y) / np.mean(y)


susceptibility_func = {
    "glauber": spin_susceptibility,
    "sis": spreading_susceptibility,
    "cowan": lambda x: gap_susceptibility
    if x[:, 0].mean() == 1.0
    else spin_susceptibility,
}

average_func = {
    "glauber": spin_average,
    "sis": lambda x: x.mean(),
    "cowan": lambda x: x.mean(),
}


class Susceptibility(Expectation):
    def __init__(self, config: Config, **kwargs):
        self.params = config.dict
        super().__init__(**kwargs)

    def setup(self, seed: int) -> Any:
        gi_seed(seed)
        config = Config.from_dict(self.params)
        prior = GraphFactory.build(config.prior)
        data_model = DataModelFactory.build(config.data_model)

        data_model.set_graph_prior(prior)
        prior.sample()
        g0 = prior.get_state()
        x0 = data_model.get_random_state(config.data_model.get("num_active", -1))
        data_model.set_graph(g0)
        data_model.sample_state(x0)

        return config, prior, data_model

    def func(self, seed: int) -> Dict[str, float]:
        config, prior, data_model = self.setup(seed)
        X = np.array(data_model.get_past_states())
        avg = average_func[config.data_model.name](X)
        susc = susceptibility_func[config.data_model.name](X)
        return dict(average=avg, susceptibility=susc)


class SusceptibilityMetrics(ExpectationMetrics):
    shortname = "susceptibility"
    keys = [
        "average",
        "susceptibility",
    ]
    expectation_factory = Susceptibility


if __name__ == "__main__":
    pass
