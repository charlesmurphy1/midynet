import numpy as np
from scipy.stats import norm, skewnorm


class Aggregator:
    @staticmethod
    def bootstrap(size=1000, **kwargs):

        if kwargs.get("samples") is not None:
            samples = kwargs.get("samples")
            idx = np.random.randint(samples.shape[0], size=size)
            return samples[idx]
        loc = kwargs.get("loc")
        scale = np.abs(kwargs.get("scale", 0))
        if "skewness" in kwargs:
            return skewnorm.rvs(
                kwargs["skewness"], loc=loc, scale=scale, size=size
            )
        return norm.rvs(loc=loc, scale=scale, size=size)

    @staticmethod
    def reduce(samples, reduction="identity"):
        return getattr(Aggregator, reduction)(samples)

    @staticmethod
    def identity(samples):
        return dict(samples=np.array(samples))

    @staticmethod
    def normal(samples):
        loc, scale = norm.fit(samples)
        return dict(loc=loc, scale=scale)

    @staticmethod
    def skewednormal(samples):
        skewness, loc, scale = skewnorm.fit(samples)
        return dict(skewness=skewness, loc=loc, scale=scale)
