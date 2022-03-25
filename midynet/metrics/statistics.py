import copy
import numpy as np
from scipy.interpolate import interp1d

__all__ = ("Statistics",)


class Statistics:
    def __init__(self, data):
        self.__data__ = data

    def __repr__(self):
        return f"Statistics(mid={self.__data__['mid']})"

    def shape(self):
        return self.__data__["mid"].shape

    def copy(self):
        return Statistics(copy.deepcopy(self.__data__))

    def __contains__(self, key):
        return key in self.__data__

    def clip(self, min=None, max=None):
        if "mid" not in self:
            return
        min = self["mid"].min() if min is None else min
        max = self["mid"].max() if max is None else max
        other = self.copy()
        other["mid"] = np.clip(self["mid"], min, max)
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
            data["mid"] += other.__data__["mid"]
            data["low"] += other.__data__["low"]
            data["high"] += other.__data__["high"]
        else:
            data["mid"] += other
        return Statistics(data)

    def __sub__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            data["mid"] -= other.__data__["mid"]
            data["low"] += other.__data__["low"]
            data["high"] += other.__data__["high"]
        else:
            data["mid"] -= other

        return Statistics(data)

    def __mul__(self, other):
        data = self.copy().__data__
        if isinstance(other, Statistics):
            data["mid"] *= other.__data__["mid"]
            data["low"] = data["mid"] * (
                self.__data__["low"] / self.__data__["mid"]
                + other.__data__["low"] / other.__data__["mid"]
            )
            data["high"] = data["mid"] * (
                self.__data__["high"] / self.__data__["mid"]
                + other.__data__["high"] / other.__data__["mid"]
            )
        else:
            data["mid"] *= other
            data["low"] *= other
            data["high"] *= other
        return Statistics(data)

    def __truediv__(self, other):
        data = self.copy().__data__

        if isinstance(other, Statistics):
            self_copy = self.copy().__data__
            other_copy = other.copy().__data__
            data["mid"] /= other.__data__["mid"]

            self_copy["low"][self.__data__["mid"] == 0] = 0
            self_copy["high"][self.__data__["mid"] == 0] = 0
            self_copy["mid"][self.__data__["mid"] == 0] = 1

            other_copy["low"][other.__data__["mid"] == 0] = 0
            other_copy["high"][other.__data__["mid"] == 0] = 0
            other_copy["mid"][other.__data__["mid"] == 0] = 1

            data["low"] = data["mid"] * (
                self.__data__["low"] / self_copy["mid"]
                - other.__data__["low"] / self_copy["mid"]
            )
            data["high"] = data["mid"] * (
                self.__data__["high"] / self_copy["mid"]
                - other.__data__["high"] / other_copy["mid"]
            )
        else:
            data["mid"] /= other
            data["low"] /= other
            data["high"] /= other
        return Statistics(data)

    def __ge__(self, other):
        return self["mid"] >= other["mid"]

    def __gt__(self, other):
        return self["mid"] > other["mid"]

    def __le__(self, other):
        return self["mid"] <= other["mid"]

    def __lt__(self, other):
        return self["mid"] < other["mid"]

    def __eq__(self, other):
        return self["mid"] == other["mid"]

    @staticmethod
    def plot(
        ax,
        x,
        y,
        fill=True,
        fill_alpha=0.2,
        fill_color=None,
        bar=True,
        spacing=1,
        interpolate=None,
        interp_num_points=1000,
        error_scaling=1,
        **kwargs,
    ):
        c = kwargs.get("color", "grey")
        a = kwargs.get("alpha", 1)
        index = np.argsort(x)
        x = np.array(x)
        marker = kwargs.pop("marker")
        linestyle = kwargs.pop("linestyle", None)
        kwargs.pop("ls", None)
        if interpolate is not None:
            interpF = interp1d(x, y["mid"], kind=interpolate)
            interpX = np.linspace(min(x), max(x), interp_num_points)
            interpY = interpF(interpX)
            ax.plot(
                interpX, interpY, marker="None", linestyle=linestyle, **kwargs
            )
        else:
            ax.plot(
                x[index],
                y["mid"][index],
                marker="None",
                linestyle=linestyle,
                **kwargs,
            )
        ax.plot(
            x[index[::spacing]],
            y["mid"][index[::spacing]],
            marker=marker,
            linestyle="None",
            **kwargs,
        )

        if fill:
            fill_color = c if fill_color is None else fill_color
            ax.fill_between(
                x[index],
                y["mid"][index] - np.abs(y["low"][index]) / error_scaling,
                y["mid"][index] + np.abs(y["high"][index]) / error_scaling,
                color=fill_color,
                alpha=a * fill_alpha,
                linestyle="None",
            )
        if bar:
            ax.errorbar(
                x[index],
                y["mid"][index],
                np.vstack((np.abs(y["low"][index]), np.abs(y["high"][index])))
                / error_scaling,
                ecolor=c,
                marker="None",
                linestyle="None",
                **kwargs,
            )
        return ax

    @staticmethod
    def mid(samples, error_type="std"):
        if error_type == "std":
            return np.mean(samples)
        elif error_type == "percentile":
            return np.median(samples)
        elif error_type == "confidence":
            return np.mean(samples)
        else:
            raise ValueError(
                f"Error_type `{error_type}` is invalid. Valid choices"
                + "are `['std', 'percentile', 'confidence']`."
            )

    @staticmethod
    def low(samples, error_type="std"):
        if error_type == "std":
            return np.std(samples)
        elif error_type == "percentile":
            return np.median(samples) - np.percentile(samples, 16)
        elif error_type == "confidence":
            return np.mean(samples) - np.percentile(samples, 5)
        else:
            raise ValueError(
                f"Error_type `{error_type}` is invalid. Valid choices"
                + "are `['std', 'percentile', 'confidence']`."
            )

    @staticmethod
    def high(samples, error_type="std"):
        if error_type == "std":
            return np.std(samples)
        elif error_type == "percentile":
            return np.percentile(samples, 84) - np.median(samples)
        elif error_type == "confidence":
            return np.percentile(samples, 95) - np.mean(samples)
        else:
            raise ValueError(
                f"Error_type `{error_type}` is invalid. Valid choices"
                + "are `['std', 'percentile', 'confidence']`."
            )

    @classmethod
    def compute(cls, samples, error_type="std"):
        num_samples = len(samples)

        mid = cls.mid(samples, error_type)
        low = cls.low(samples, error_type) / np.sqrt(num_samples)
        high = cls.high(samples, error_type) / np.sqrt(num_samples)
        return dict(mid=mid, low=low, high=high)
