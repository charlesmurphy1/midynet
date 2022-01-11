import numpy as np


class Statistics:
    def __call__(self, samples):
        raise NotImplementedError()


class MCStatistics(Statistics):
    def __init__(self, error_type="std", **kwargs):
        if error_type == "std":
            self.mid = lambda s: np.mean(s)
            self.low_error = lambda s: np.std(s)
            self.high_error = lambda s: np.std(s)
        elif error_type == "percentile":
            self.mid = lambda s: np.median(s)
            self.low_error = lambda s: (np.median(s) - np.percentile(s, 16))
            self.high_error = lambda s: (np.percentile(s, 84) - np.median(s))
        elif error_type == "confidence":
            alpha = kwargs.get("alpha", 5)
            self.mid = lambda s: np.mean(s)
            self.low_error = lambda s: (np.mean(s) - np.percentile(s, alpha))
            self.high_error = lambda s: (np.percentile(s, 100 - alpha) - np.mean(s))
        else:
            raise ValueError(
                f"Error_type `{error_type}` is invalid. Valid choices are `['std', 'percentile']`."
            )

    def __call__(self, samples):
        num_samples = len(samples)

        mid = self.mid(samples)
        low = self.low_error(samples) / np.sqrt(num_samples)
        high = self.high_error(samples) / np.sqrt(num_samples)
        return dict(mid=mid, low=low, high=high)
