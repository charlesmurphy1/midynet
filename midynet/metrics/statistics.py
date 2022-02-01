import copy

import numpy as np

__all__ = ["Statistics"]


class Statistics:
    def __init__(self, data):
        self.__data__ = data

    def __repr__(self):
        return f"Statistics(mid={self.__data__['mid']})"

    def shape(self):
        return self.__data__["mid"].shape

    def copy(self):
        return Statistics(copy.deepcopy(self.__data__))

    def __getitem__(self, key):
        if isinstance(key, str) and key in self.__data__:
            return self.__data__[key]
        else:
            return {k: v[key] for k, v in self.__data__.items()}

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
                self.__data__["low"] / self.__data__["mid"]
                - other.__data__["low"] / other.__data__["mid"]
            )
            data["high"] = data["mid"] * (
                self.__data__["high"] / self.__data__["mid"]
                - other.__data__["high"] / other.__data__["mid"]
            )
        else:
            data["mid"] /= other
            data["low"] /= other
            data["high"] /= other
        return Statistics(data)

    @staticmethod
    def plot(ax, x, y, fill_alpha=0.2, **kwargs):
        c = kwargs.get("color", "grey")
        a = kwargs.get("alpha", 1)
        index = np.argsort(x)
        x = np.array(x)
        ax.plot(x[index], y["mid"][index], **kwargs)
        ax.fill_between(
            x[index],
            y["mid"][index] - y["low"][index],
            y["mid"][index] + y["high"][index],
            color=c,
            alpha=a * fill_alpha,
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
                f"Error_type `{error_type}` is invalid. Valid choices are `['std', 'percentile', 'confidence']`."
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
                f"Error_type `{error_type}` is invalid. Valid choices are `['std', 'percentile', 'confidence']`."
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
                f"Error_type `{error_type}` is invalid. Valid choices are `['std', 'percentile', 'confidence']`."
            )

    @classmethod
    def compute(cls, samples, error_type="std"):
        num_samples = len(samples)

        mid = cls.mid(samples, error_type)
        low = cls.low(samples, error_type) / np.sqrt(num_samples)
        high = cls.high(samples, error_type) / np.sqrt(num_samples)
        return dict(mid=mid, low=low, high=high)
