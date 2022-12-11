import numpy as np


class Aggregator:
    @staticmethod
    def identity(samples):
        return samples

    @staticmethod
    def std(samples):
        std = np.std(samples) / np.sqrt(len(samples))
        return dict(mid=np.mean(samples), low=-std, high=std)

    @staticmethod
    def percentile(samples, alpha=5):
        median = np.median(samples)
        low = (median - np.percentile(samples, alpha)) / np.sqrt(len(samples))
        high = (np.percentile(samples, 100 - alpha) - median) / np.sqrt(
            len(samples)
        )
        return dict(mid=np.median(samples), low=low, high=high)
