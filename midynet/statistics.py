from __future__ import annotations
import copy
import numpy as np
import pandas as pd
import seaborn as sb

from typing import Optional, List, Any, Dict
from scipy.interpolate import interp1d

from .aggregator import Aggregator

__all__ = ("Statistics",)


class Statistics:
    def __init__(
        self,
        data_dict: Dict[str, int or float or np.ndarray],
        name: str = "stat",
    ):
        self.name = name
        self.__data__ = data_dict

    @classmethod
    def from_samples(
        cls,
        samples: List[float] or np.ndarray,
        reduction: str = "normal",
        name: str = "stat",
    ) -> Statistics:
        num_samples = len(samples)
        reduced = Aggregator.reduce(samples, reduction=reduction)
        if "scale" in reduced:
            reduced["scale"] /= np.sqrt(num_samples)
            reduced["scale"] = np.clip(
                reduced["scale"], a_min=1e-15, a_max=np.inf
            )
        return cls(reduced, name=name)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, key: str) -> Statistics:
        subdf = df.filter(regex=f"^{key}")
        data = {}
        for i, col in enumerate(subdf.columns):
            k = col.replace(key, "")
            if not k[0].isalpha():
                k = k[1:]
            data[k] = subdf[col].values
        return cls(data, name=key)

    @property
    def dict(self):
        out = dict()
        for k, v in self.__data__.items():
            out[self.name + "_" + k] = [v] if isinstance(v, float) else v
        return out

    def __repr__(self):
        out = "Statistics("
        for k, v in self.__data__.items():
            out += f"{k}={v}, "

        return out[:-2] + ")"

    def __getattr__(self, key):
        if key == "__data__":
            return self.__data__
        return self.__data__[key]

    def copy(self):
        return Statistics(copy.deepcopy(self.__data__))

    def __contains__(self, key):
        return key in self.__data__

    def clip(self, min=None, max=None):
        if "loc" not in self:
            return
        min = self["loc"].min() if min is None else min
        max = self["loc"].max() if max is None else max
        other = self.copy()
        other["loc"] = np.clip(self["loc"], min, max)
        return other

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.__data__:
            return self.__data__[key]
        else:
            return {k: v[key] for k, v in self.__data__.items()}

    def __setitem__(self, key, value):
        if isinstance(key, str) and key in self.__data__:
            self.__data__[key] = value
        else:
            msg = f"Key {key} not found in Statistics {self}."
            raise LookupError(msg)

    def __add__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            data["loc"] += other.__data__["loc"]
            data["scale"] += other.__data__["scale"]
        else:
            data["loc"] += other
        return Statistics(data)

    def __sub__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            data["loc"] -= other.__data__["loc"]
            data["scale"] += other.__data__["scale"]
        else:
            data["loc"] -= other

        return Statistics(data)

    def __mul__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            data["loc"] *= other.__data__["loc"]
            data["scale"] = data["loc"] * (
                self.__data__["scale"] / self.__data__["loc"]
                + other.__data__["scale"] / other.__data__["loc"]
            )
        else:
            data["loc"] *= other
            data["scale"] *= other
        return Statistics(data)

    def __truediv__(self, other):
        data = self.copy().__data__

        if not isinstance(other, Statistics):
            raise ValueError()
        self_copy = self.copy().__data__
        other_copy = other.copy().__data__
        data["loc"] /= other.__data__["loc"]

        if isinstance(self_copy["scale"], np.ndarray):
            self_copy["scale"][self.__data__["loc"] == 0] = 0
            self_copy["loc"][self.__data__["loc"] == 0] = 1

            other_copy["scale"][other.__data__["loc"] == 0] = 0
            other_copy["loc"][other.__data__["loc"] == 0] = 1
        elif isinstance(self_copy["scale"], (int, float)):
            self_copy["scale"] = (
                0 if self.__data__["loc"] == 0 else self.__data__["loc"]
            )
            self_copy["loc"] = (
                1 if self.__data__["loc"] == 0 else self.__data__["loc"]
            )

            other_copy["scale"] = (
                0 if other.__data__["loc"] == 0 else other.__data__["loc"]
            )
            other_copy["loc"] = (
                1 if other.__data__["loc"] == 0 else other.__data__["loc"]
            )

        data["scale"] = data["loc"] * np.abs(
            self.__data__["scale"] / self_copy["loc"]
            - other.__data__["scale"] / self_copy["loc"]
        )
        return Statistics(data)

    def __ge__(self, other):
        return self["loc"] >= other["loc"]

    def __gt__(self, other):
        return self["loc"] > other["loc"]

    def __le__(self, other):
        return self["loc"] <= other["loc"]

    def __lt__(self, other):
        return self["loc"] < other["loc"]

    def __eq__(self, other):
        return self["loc"] == other["loc"]

    def bootstrap(self, size=1000):
        if isinstance(self.__data__, np.ndarray):
            idx = np.random.randint(len(self.__data__), size=size)
            return self.__data__[idx]
        if isinstance(next(iter(self.__data__.values())), np.ndarray):
            n = len(next(iter(self.__data__.values())))
            samples = np.zeros((n, size))

            for i in range(n):
                params = {k: v[i] for k, v in self.__data__.items()}
                samples[i] = Aggregator.bootstrap(size=size, **params)
            return samples

        return Aggregator.bootstrap(size=size, **self.__data__)

    def lineplot(
        self,
        x: pd.Series or np.ndarray or List[Any],
        aux: Optional[pd.Series or np.ndarray or List[Any]] = None,
        **kwargs,
    ):
        bs = self.bootstrap(kwargs.pop("num_samples", 1000))
        df = pd.DataFrame.from_records(bs)
        if not isinstance(x, pd.Series):
            x = pd.Series(x, name="xaxis")
        if aux is not None and not isinstance(aux, pd.Series):
            aux = pd.series(aux, name="aux")
        df[x.name] = x
        if aux is not None:
            df[aux.name] = aux

        df = df.melt([x.name, aux.name] if aux is not None else x.name)
        return sb.lineplot(
            df,
            y="value",
            x=x.name,
            hue=aux.name if aux is not None else None,
            **kwargs,
        )
